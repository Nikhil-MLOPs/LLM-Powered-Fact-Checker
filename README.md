# 🔍 PIB Fact Checker

A **100% offline**, fast, and lightweight fact-checking tool that verifies claims against official Press Information Bureau (PIB) fact-check data.

Use case - Perfect for journalists, researchers, WhatsApp fact-checkers, or anyone tired of viral misinformation.

## Features

- ✅ **Fully offline** after initial setup (embedding model + ChromaDB)
- ✅ Uses official PIB fact-check CSV as the knowledge base
- ✅ Semantic search with `all-MiniLM-L6-v2` embeddings
- ✅ Automatic claim extraction using **Google Flan-T5-base**
- ✅ Vague claim detection (e.g., "many people say", "sometimes happens")
- ✅ Confidence scoring + evidence from original PIB records
- ✅ Beautiful Gradio web interface
- ✅ Works on CPU or GPU

## How It Works (High-Level)

1. Input text → split into verifiable factual claims
2. Filter out vague claims (no dates/numbers/names)
3. Embed each claim → retrieve most similar chunks from PIB vector DB
4. Compare similarity → return **True / False / Unverifiable** with evidence

## Requirements

- Python 3.10+
- Runs fine on CPU.

## Installation & Setup

# 1. Clone or download this repository
- git clone https://github.com/Nikhil-MLOPs/pib-fact-checker.git
- cd pib-fact-checker

# 2. Install dependencies
pip install -r requirements.txt

# 3. Add csv file
Add your PIB data in csv format with one column name being statement in the same folder containing main.py

# 4. Run the app
python main.py

Gradio will open in your browser