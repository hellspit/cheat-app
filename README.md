# Test Assistant - PDF Q&A System

## ✅ Yes, Your Idea is Possible!

This system uses Ollama (free, local LLM) to answer MCQ questions from your PDF study material during tests.

## 🚀 Setup Instructions

### 1. Install Ollama
Download and install from: https://ollama.ai/

### 2. Download Required Models (run in terminal):
```bash
ollama pull mistral
ollama pull nomic-embed-text
```

### 3. Install Python Dependencies:
```bash
pip install -r requirements.txt
```

### 4. Run the System:

**Option A: Beautiful Web Interface (Recommended):**
```bash
streamlit run streamlit_app.py
```

**Option B: Command Line:**
```bash
python main.py
```

The first run will:
- Load all PDFs from the `pdf` folder
- Create embeddings (takes 5-10 minutes for 1000 pages)
- Store everything locally in the `db` folder
- Streamlit will open automatically in your browser

## 💡 How to Use During Test

### Using Streamlit (Recommended):
1. Run `streamlit run streamlit_app.py`
2. Click "Initialize System" (first time only)
3. Paste your MCQ question in the text box
4. Click "Get Answer" - instant response!
5. View sources to verify information

### Using Command Line:
1. Run `python main.py`
2. Paste your MCQ question
3. Get the answer with source reference
4. Type `exit` to quit

### Example:
```
❓ Question: What is the main cause of climate change?
Options: A) Solar cycles B) CO2 emissions C) Ocean currents D) Clouds

💡 Answer:
B) CO2 emissions. Human activities, especially burning fossil fuels, release carbon dioxide which traps heat in the atmosphere...

📚 Found in: pdf\1.pdf
```

## ⚠️ Important Notes

- **First run is slow** (embeddings generation) but later runs are instant
- Works **100% offline** - nothing sends to internet
- The `db` folder stores your knowledge base
- Delete `db` folder to rebuild from scratch

## 🎯 Tips for MCQ Success

1. Include all 4 options in your question for better accuracy
2. Be specific about what you're asking
3. Mention the topic/subject if multiple topics in PDFs

Good luck on your test! 🍀
