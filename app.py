import os
import dotenv
import streamlit as st
from streamlit_chat import message  # for the chat UI

# LangChain / Chroma / OpenAI imports
from langchain.embeddings import OpenAIEmbeddings
from langchain.document_loaders import DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.chains import RetrievalQA
from langchain.chat_models import ChatOpenAI  # NEW: Replaces langchain.llms.OpenAI

# Load environment variables
dotenv.load_dotenv(".env", override=True)

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    st.warning("OPENAI_API_KEY environment variable not set. Please set it in your .env file.")

# 2. Create (or load) the Chroma vector store
def create_or_load_vectorstore(persist_directory: str):
    """
    If 'documents' are provided, create a new Chroma store from them and persist it.
    Otherwise, load from the existing persist_directory.
    """

    embedding_function = OpenAIEmbeddings(openai_api_key=os.getenv("OPENAI_API_KEY"))

    # Create/update the store from the documents
    vectorstore = Chroma(
            collection_name="documents",
            embedding_function=embedding_function,
            persist_directory=persist_directory
        )
    #vectorstore.persist()

    return vectorstore

# 3. Build the retrieval-augmented QA chain using the new ChatOpenAI model
def build_qa_chain(vectorstore):
    """
    Builds a retrieval-augmented QA chain using the provided vectorstore.
    Uses ChatOpenAI (compatible with openai >= 1.0).
    """
    # Use GPT-4 by specifying model_name="gpt-4"
    llm = ChatOpenAI(
        openai_api_key=OPENAI_API_KEY,
        temperature=0.5,
        model_name="gpt-4"
    )

    retriever = vectorstore.as_retriever(search_kwargs={"k": 10})


    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever,
        return_source_documents=True
    )
    return qa_chain

def main():
    st.title("Psychic AI")


    with st.spinner("Creating or loading Chroma vector store..."):
        vectorstore = create_or_load_vectorstore(persist_directory="chromadb")

    with st.spinner("Building QA chain..."):
        qa_chain = build_qa_chain(vectorstore)

    # Initialize chat history in session_state
    if "messages" not in st.session_state:
        # Start with a welcome message from the assistant
        st.session_state.messages = [
            {"role": "assistant", "content": "Ciao!"}
        ]



    # Streamlit's chat input for new user messages
    user_input = st.chat_input("Ponimi una domanda od un cosiglio.")
    
    if user_input:
        # Add the user's message to the session state
        st.session_state.messages.append({"role": "user", "content": user_input})

        for i, msg in enumerate(st.session_state.messages):
            if msg["role"] == "assistant":
                message(msg["content"], key=str(i))
            else:
                message(msg["content"], is_user=True, key=str(i))

        # RAG retrieval + LLM response
        with st.spinner("Retrieving and generating an answer..."):
            result = qa_chain({"query": user_input})
            answer = result["result"]
            source_docs = result["source_documents"]

        for i, doc in enumerate(source_docs):
            source_info = doc.metadata.get("source", f"Document {i+1}")
            print(source_info)

        # Create a full chatbot answer (answer + sources)
        full_response = f"{answer}"

        # Assistant (chatbot) message
        st.session_state.messages.append({"role": "assistant", "content": full_response})
        st.experimental_rerun()

            # Display existing messages
    
    for i, msg in enumerate(st.session_state.messages):
            if msg["role"] == "assistant":
                message(msg["content"], key=str(i))
            else:
                message(msg["content"], is_user=True, key=str(i))
        # Format sources
        # sources_text = "\n\n**Sources:**\n"
        # for i, doc in enumerate(source_docs):
        #     source_info = doc.metadata.get("source", f"Document {i+1}")
        #     sources_text += f"- {source_info}\n"



if __name__ == "__main__":
    main()
