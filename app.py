import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
import os

# Load Google API Key from Streamlit secrets
GOOGLE_API_KEY = st.secrets['GOOGLE_API_KEY']


def get_pdf_text(pdf_docs):
    """Extract text from uploaded PDF files."""
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text
    if not text.strip():
        raise ValueError("No text could be extracted from the PDF files. Ensure they contain readable content.")
    return text


def get_text_chunks(text):
    """Split text into smaller chunks."""
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=5000, chunk_overlap=200)
    chunks = text_splitter.split_text(text)
    if not chunks:
        raise ValueError("No valid text chunks were generated. Ensure the text content is correct.")
    return chunks


def get_vector_store(text_chunks):
    """Create a FAISS vector store from text chunks."""
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")

    # Test embedding generation
    test_embedding = embeddings.embed_documents(["This is a test"])
    if not test_embedding:
        raise ValueError("Embedding generation failed. Check your Google API key or configuration.")
    
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    vector_store.save_local("faiss_index")


def get_conversational_chain():
    """Load the conversational chain for answering questions."""
    prompt_template = """
    Answer the question as detailed as possible from the provided context, making sure to provide all the details. 
    If the answer is not in the provided context, just say, "The answer is not available in the context."
    Do not provide incorrect or misleading answers.
    
    Context:\n{context}\n
    Question:\n{question}\n
    Answer:
    """
    model = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.3)
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)
    return chain


def user_input(user_question):
    """Process user input and return a response."""
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")

    # Load the FAISS index
    new_db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
    docs = new_db.similarity_search(user_question)
    
    if not docs:
        st.write("No relevant documents found for the question.")
        return

    chain = get_conversational_chain()
    response = chain({"input_documents": docs, "question": user_question}, return_only_outputs=True)

    st.write("Reply:", response["output_text"])


def main():
    """Streamlit app main function."""
    st.set_page_config("Chat PDF")
    st.header("Chat with PDF using Gemini 🤖🗨️")

    user_question = st.text_input("Ask a Question from the PDF Files")

    if user_question:
        try:
            user_input(user_question)
        except Exception as e:
            st.error(f"An error occurred: {str(e)}")

    with st.sidebar:
        st.title("Menu:")
        pdf_docs = st.file_uploader("Upload your PDF Files and Click on the Submit & Process Button", accept_multiple_files=True)
        if st.button("Submit & Process"):
            try:
                with st.spinner("Processing..."):
                    raw_text = get_pdf_text(pdf_docs)
                    text_chunks = get_text_chunks(raw_text)
                    get_vector_store(text_chunks)
                    st.success("Done! Vector store created.")
            except Exception as e:
                st.error(f"An error occurred during processing: {str(e)}")


if __name__ == "__main__":
    main()
