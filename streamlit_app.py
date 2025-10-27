import streamlit as st
import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.llms import Ollama
from langchain_core.prompts import PromptTemplate

# Page config
st.set_page_config(page_title="📚 Test Assistant", page_icon="📚", layout="wide")

# Initialize session state
if "qa_chain" not in st.session_state:
    st.session_state.qa_chain = None
if "is_loaded" not in st.session_state:
    st.session_state.is_loaded = False
if "vectordb" not in st.session_state:
    st.session_state.vectordb = None

# Sidebar
with st.sidebar:
    st.title("📚 Setup")
    st.markdown("---")
    
    # Check if DB exists
    db_exists = os.path.exists("db")
    
    if db_exists:
        st.success("✅ Database Ready!")
        st.info("Press 'Initialize' to load the Q&A system")
    else:
        st.warning("⚠️ Database not found")
        st.info("Press 'Load PDFs' to process your files")
    
    st.markdown("---")
    
    # Model selection
    model = st.selectbox(
        "🤖 Select LLM Model",
        ["mistral", "743xp", "llama3:latest", "phi3"],
        index=0
    )
    
    st.markdown("---")
    st.markdown("### 📖 Usage Tips")
    st.markdown("""
    1. Include all MCQ options
    2. Be specific in your question
    3. Mention topic if needed
    4. First time takes 5-10 min
    """)

# Main area
st.title("📚 Test Assistant - MCQ Helper")
st.markdown("Upload your PDFs and get instant answers used during tests!")

col1, col2 = st.columns([3, 1])

with col1:
    st.markdown("### 💬 Ask Your Question")
    
with col2:
    if st.button("🔄 Reset", use_container_width=True):
        st.session_state.qa_chain = None
        st.session_state.is_loaded = False
        st.rerun()

# Auto-load existing database on startup
if not st.session_state.is_loaded and os.path.exists("db") and len(os.listdir("db")) > 0:
    with st.spinner("🔍 Loading existing database..."):
        try:
            embeddings = OllamaEmbeddings(model="nomic-embed-text")
            vectordb = Chroma(persist_directory="db", embedding_function=embeddings)
            
            llm = Ollama(model=model)
            prompt = PromptTemplate(
                template="Use the following context to answer the question. If you don't know the answer, just say you don't know.\n\nContext: {context}\n\nQuestion: {question}\n\nAnswer:",
                input_variables=["context", "question"]
            )
            
            st.session_state.llm = llm
            st.session_state.prompt = prompt
            st.session_state.retriever = vectordb.as_retriever(search_kwargs={"k": 3})
            st.session_state.is_loaded = True
        except Exception as e:
            st.error(f"❌ Error loading database: {str(e)}")

# Initialize system
if not st.session_state.is_loaded:
    if st.button("🚀 Initialize System", type="primary", use_container_width=True):
        with st.spinner("🔍 Loading PDFs and creating embeddings... (first time takes 5-10 minutes)"):
            try:
                # 1. Load PDFs
                folder_path = "pdf"
                docs = []
                pdf_files = [f for f in os.listdir(folder_path) if f.endswith(".pdf")]
                
                if not pdf_files:
                    st.error("❌ No PDFs found in 'pdf' folder! Please add your PDF files.")
                else:
                    progress_bar = st.progress(0)
                    for i, file in enumerate(pdf_files):
                        try:
                            loader = PyPDFLoader(os.path.join(folder_path, file))
                            docs.extend(loader.load())
                            progress_bar.progress((i + 1) / len(pdf_files))
                        except Exception as e:
                            st.warning(f"⚠️ Error loading {file}: {str(e)}")
                    
                    st.success(f"✅ Loaded {len(docs)} pages from {len(pdf_files)} PDFs")
                    
                    # 2. Split text
                    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
                    chunks = splitter.split_documents(docs)
                    st.info(f"📄 Created {len(chunks)} text chunks")
                    
                    # 3. Create embeddings
                    st.info("🧠 Creating embeddings (this may take a while)...")
                    embeddings = OllamaEmbeddings(model="nomic-embed-text")
                    vectordb = Chroma.from_documents(chunks, embeddings, persist_directory="db")
                    st.session_state.vectordb = vectordb
                    
                    # 4. Initialize LLM and prompt
                    llm = Ollama(model=model)
                    prompt = PromptTemplate(
                        template="Use the following context to answer the question. If you don't know the answer, just say you don't know.\n\nContext: {context}\n\nQuestion: {question}\n\nAnswer:",
                        input_variables=["context", "question"]
                    )
                    
                    # 5. Store components
                    st.session_state.llm = llm
                    st.session_state.prompt = prompt
                    st.session_state.retriever = vectordb.as_retriever(search_kwargs={"k": 3})
                    st.session_state.is_loaded = True
                    st.success("🎉 Ready to answer questions!")
                    st.rerun()
                    
            except Exception as e:
                st.error(f"❌ Error: {str(e)}")
                st.info("💡 Make sure you have:")
                st.info("- Ollama installed and running")
                st.info("- Models downloaded (ollama pull mistral)")
                st.info("- PDFs in the 'pdf' folder")

# If system is loaded, show Q&A interface
if st.session_state.is_loaded:
    # Question input
    question = st.text_area(
        "Enter your MCQ question:",
        placeholder="Example: What is the main cause of climate change?\n\nOptions:\nA) Solar cycles\nB) CO2 emissions\nC) Ocean currents\nD) Clouds",
        height=150
    )
    
    if st.button("🔍 Get Answer", type="primary", use_container_width=True):
        if question.strip():
            with st.spinner("🤔 Thinking..."):
                try:
                    # Retrieve relevant documents
                    docs = st.session_state.retriever.invoke(question)
                    context = "\n\n".join([doc.page_content for doc in docs])
                    
                    # Get answer from LLM
                    answer = st.session_state.llm.invoke(st.session_state.prompt.format(context=context, question=question))
                    
                    # Display answer
                    st.markdown("### 💡 Answer:")
                    st.markdown(f"**{answer}**")
                    
                    # Display sources
                    if docs:
                        with st.expander("📚 View Sources"):
                            for i, doc in enumerate(docs[:3]):
                                st.markdown(f"**Source {i+1}:**")
                                st.text(f"Document: {doc.metadata.get('source', 'Unknown')}")
                                st.text(f"Page: {doc.metadata.get('page', 'N/A')}")
                                st.text(f"Content: {doc.page_content[:300]}...")
                                st.markdown("---")
                except Exception as e:
                    st.error(f"❌ Error getting answer: {str(e)}")
        else:
            st.warning("⚠️ Please enter a question")
    
    # Example questions
    st.markdown("---")
    st.markdown("### 💡 Example Questions")
    example_cols = st.columns(3)
    
    with example_cols[0]:
        if st.button("📝 Example 1", use_container_width=True):
            st.code("What is photosynthesis?")
    
    with example_cols[1]:
        if st.button("📝 Example 2", use_container_width=True):
            st.code("Explain the water cycle")
    
    with example_cols[2]:
        if st.button("📝 Example 3", use_container_width=True):
            st.code("What are the types of cells?")

# Footer
st.markdown("---")
st.markdown(
    "<div style='text-align: center; color: gray;'>🔒 100% Offline | No Data Sent to Internet | Powered by Ollama</div>",
    unsafe_allow_html=True
)

