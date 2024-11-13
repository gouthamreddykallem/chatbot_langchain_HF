import os
from langchain.document_loaders import TextLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma
from langchain.chains import RetrievalQA
from langchain_huggingface import HuggingFaceEndpoint
import chainlit as cl
import re

# Load environment variables
os.environ["HUGGINGFACEHUB_API_TOKEN"] = "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"

# Load the document
loader = TextLoader("restructured-tng-data.txt")
documents = loader.load()

# Split the document into chunks
text_splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=50)
texts = text_splitter.split_documents(documents)

# Create embeddings
embeddings = HuggingFaceEmbeddings()

# Create a vector store
db = Chroma.from_documents(texts, embeddings)

# Initialize the Hugging Face model
llm = HuggingFaceEndpoint(
    repo_id="mistralai/Mistral-7B-Instruct-v0.2",
    task="text-generation",
    max_new_tokens=100,
    do_sample=True,
    repetition_penalty=1.03,
)

# Create a retrieval chain
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=db.as_retriever(),
    return_source_documents=True
)

def is_valid_email(email):
    pattern = r'^[\w\.-]+@[\w\.-]+\.\w+$'
    return re.match(pattern, email) is not None

def is_valid_mobile(mobile):
    pattern = r'^\+?1?\d{9,15}$'
    return re.match(pattern, mobile) is not None

async def collect_patient_info():
    patient_info = {}

    full_name_response = await cl.AskUserMessage(
        content="Could you please provide your full name?").send()
    patient_info['full_name'] = full_name_response['output']

    while True:
        email_response = await cl.AskUserMessage(content="Please provide your email address:").send()
        email = email_response['output']
        if is_valid_email(email):
            patient_info['email'] = email
            break
        else:
            await cl.Message(content="Invalid email format. Please try again.").send()

    while True:
        mobile_response = await cl.AskUserMessage(content="Please provide your mobile number:").send()
        mobile = mobile_response['output']
        if is_valid_mobile(mobile):
            patient_info['mobile'] = mobile
            break
        else:
            await cl.Message(content="Invalid mobile number format. Please try again.").send()

    query_response = await cl.AskUserMessage(content="Please restate your query:").send()
    patient_info['query'] = query_response['output']

    print("Patient Information:")
    for key, value in patient_info.items():
        print(f"{key.capitalize()}: {value}")

    return patient_info

@cl.on_chat_start
def start():
    cl.user_session.set("qa_chain", qa_chain)

def needs_more_info(response):
    keywords = ["don't have enough information",
                "cannot answer",
                "insufficient information",
                "need more details",
                "No information",
                "I'm sorry",
                "I don't have enough context"
            ]
    return any(keyword.lower() in response.lower() for keyword in keywords)

@cl.on_message
async def main(message: cl.Message):
    qa_chain = cl.user_session.get("qa_chain")

    query = message.content

    response = qa_chain({"query": query})
    print(response)

    if needs_more_info(response["result"]):
        await cl.Message(
            content="I apologize, but I don't have enough information to answer your question accurately. Let me collect some information to better assist you.").send()
        await collect_patient_info()
        await cl.Message(
            content="Thank you for providing your information. A healthcare professional will review your query and get back to you soon.").send()
    else:
        await cl.Message(content=response["result"]).send()

# Run the Chainlit app
# Use: chainlit run this_script.py -w