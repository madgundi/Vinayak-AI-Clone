# Install required packages (Uncomment the line below if running locally)
!pip install streamlit langchain langchain-community langchain-groq sentence-transformers faiss-cpu unstructured PyPDF2 python-docx pymupdf tiktoken

import streamlit as st
import os
import tempfile
import logging
from typing import List, Dict
from langchain_groq import ChatGroq
from langchain_community.document_loaders import (
    PyPDFLoader, Docx2txtLoader, TextLoader, CSVLoader,
    UnstructuredHTMLLoader, UnstructuredMarkdownLoader
)
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain.prompts import PromptTemplate

# ✅ Set API Key (Replace with your actual API key)
GROQ_API_KEY = "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"
os.environ["GROQ_API_KEY"] = GROQ_API_KEY

# ✅ Initialize Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ✅ MultiFormatRAG Class
class MultiFormatRAG:
    def __init__(self):
        self.loader_map = {
            '.pdf': PyPDFLoader,
            '.docx': Docx2txtLoader,
            '.txt': TextLoader,
            '.csv': CSVLoader,
            '.html': UnstructuredHTMLLoader,
            '.md': UnstructuredMarkdownLoader
        }
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000, chunk_overlap=200, length_function=len
        )
        self.embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")

    def load_documents(self, directory_path: str) -> List[Dict]:
        documents = []
        for file in os.listdir(directory_path):
            file_path = os.path.join(directory_path, file)
            file_extension = os.path.splitext(file)[1].lower()
            if file_extension in self.loader_map:
                try:
                    loader = self.loader_map[file_extension](file_path)
                    docs = loader.load()
                    if docs:
                        logger.info(f"Loaded {file}")
                        documents.extend(docs)
                    else:
                        logger.warning(f"No content found in {file}")
                except Exception as e:
                    logger.error(f"Error loading {file}: {str(e)}")
        return documents

    def process_documents(self, documents: List[Dict]) -> FAISS:
        if not documents:
            raise ValueError("No valid documents found for processing.")
        texts = self.text_splitter.split_documents(documents)
        vectorstore = FAISS.from_documents(texts, self.embeddings)
        return vectorstore

    def create_qa_chain(self, vectorstore: FAISS) -> ConversationalRetrievalChain:
        system_prompt = (
           """**Goal**
I want a chatbot that references a PDF about Vinayak Madgundi's background, skills, projects, and certifications. 
It should provide short answers (with emojis) for simple questions, and **detailed explanations** (referencing PDF
insights) for more complex questions. 

**Behavioral Directives**
- Present yourself as a knowledgeable human expert.
- Use emojis for warmth and clarity.
"""
        )
        prompt_template = PromptTemplate(
            input_variables=["chat_history", "context", "question"],
            template=f"{system_prompt}\n\nChat History:\n{{chat_history}}\n\nContext:\n{{context}}\n\nQuestion:\n{{question}}\n\nAnswer:"
        )
        memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
        llm = ChatGroq(model="llama3-70b-8192", temperature=0)
        return ConversationalRetrievalChain.from_llm(
            llm=llm,
            retriever=vectorstore.as_retriever(search_kwargs={"k": 3}),
            memory=memory,
            combine_docs_chain_kwargs={"prompt": prompt_template}
        )

    def query(self, qa_chain: ConversationalRetrievalChain, question: str) -> str:
        try:
            response = qa_chain.invoke({"question": question, "chat_history": st.session_state.chat_history})
            answer = response.get("answer", "No response generated.")
            st.session_state.chat_history.append({"role": "assistant", "content": answer})
            return answer
        except Exception as e:
            logger.error(f"Query Error: {str(e)}")
            return "Error processing query."

# ✅ Streamlit Page Configuration
st.set_page_config(page_title="AI Clone Chat Interface", page_icon="🤖", layout="wide")
st.title("🤖 AI Chatbot Of Vinayak")
st.write("Ask me anything!")

# ✅ Initialize Streamlit Session State
def initialize_session_state():
    if 'rag_system' not in st.session_state:
        st.session_state.rag_system = None
    if 'qa_chain' not in st.session_state:
        st.session_state.qa_chain = None
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []

initialize_session_state()

# ✅ Sidebar for Configuration
with st.sidebar:
    st.title("Configuration")
    uploaded_files = st.file_uploader("Upload Training Documents", accept_multiple_files=True, type=['pdf', 'docx', 'txt', 'csv', 'html', 'md'])
    if uploaded_files and st.button("Initialize System"):
        with st.spinner("Initializing AI Clone..."):
            try:
                temp_dir = tempfile.mkdtemp()
                for file in uploaded_files:
                    file_path = os.path.join(temp_dir, file.name)
                    with open(file_path, "wb") as f:
                        f.write(file.getvalue())
                st.session_state.rag_system = MultiFormatRAG()
                documents = st.session_state.rag_system.load_documents(temp_dir)
                if not documents:
                    st.error("No valid documents found. Please upload valid files.")
                else:
                    vectorstore = st.session_state.rag_system.process_documents(documents)
                    st.session_state.qa_chain = st.session_state.rag_system.create_qa_chain(vectorstore)
                    st.success("System initialized successfully!")
            except Exception as e:
                st.error(f"Initialization Error: {str(e)}")

# ✅ Display Chat History
for message in st.session_state.chat_history:
    role = "🤖 AI" if message["role"] == "assistant" else "🧑‍💻 You"
    st.write(f"**{role}:** {message['content']}")

# ✅ User Input Section
if st.session_state.qa_chain is not None:
    user_input = st.chat_input("Type your message...")
    if user_input:
        st.session_state.chat_history.append({"role": "user", "content": user_input})
        st.chat_message("user").write(user_input)
        with st.spinner("Thinking..."):
            try:
                response = st.session_state.rag_system.query(st.session_state.qa_chain, user_input)
                st.chat_message("assistant").write(response)
            except Exception as e:
                st.error(f"Response Error: {str(e)}")
else:
    st.info("Please initialize the system using the sidebar.")

st.markdown("---")
st.markdown("AI Clone powered by GROQ and LangChain")
