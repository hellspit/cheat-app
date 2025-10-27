import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.llms import Ollama
from langchain_core.prompts import PromptTemplate

# 1️⃣ Load all PDFs
folder_path = "pdf"
docs = []
for file in os.listdir(folder_path):
    if file.endswith(".pdf"):
        loader = PyPDFLoader(os.path.join(folder_path, file))
        docs.extend(loader.load())

print(f"Loaded {len(docs)} PDF pages from {len(os.listdir(folder_path))} files.")

# 2️⃣ Split text into chunks
splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
chunks = splitter.split_documents(docs)

# 3️⃣ Create embeddings and store in Chroma
print("🔍 Creating or loading vector database (first time may take a few minutes)...")
embeddings = OllamaEmbeddings(model="nomic-embed-text")
vectordb = Chroma.from_documents(chunks, embeddings, persist_directory="db")
vectordb.persist()

# 4️⃣ Initialize LLM
llm = Ollama(model="mistral")  # or 'phi3', 'llama3:latest'

# 5️⃣ Create prompt template
prompt = PromptTemplate(
    template="Use the following context to answer the question. If you don't know the answer, just say you don't know.\n\nContext: {context}\n\nQuestion: {question}\n\nAnswer:",
    input_variables=["context", "question"]
)

print("\n✅ Ready! Type your question below. Type 'exit' to quit.\n")

# 6️⃣ Ask questions interactively
retriever = vectordb.as_retriever(search_kwargs={"k": 3})

while True:
    query = input("❓ Question: ")
    if query.lower() in ["exit", "quit", "q"]:
        break
    
    # Retrieve relevant documents
    docs = retriever.invoke(query)
    context = "\n\n".join([doc.page_content for doc in docs])
    
    # Get answer from LLM
    answer = llm.invoke(prompt.format(context=context, question=query))
    
    print(f"\n💡 Answer:\n{answer}\n")
    if docs:
        print(f"📚 Found in: {docs[0].metadata.get('source', 'Unknown')}\n")
