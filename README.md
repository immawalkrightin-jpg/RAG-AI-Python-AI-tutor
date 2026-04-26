# 🎓 Python & AI Programming Tutor

An adaptive RAG-powered tutoring chatbot that quizzes you on Python 
and AI concepts. Built entirely with local models — no paid APIs required.

## How It Works

1. **Scraping** : Fetches Wikipedia articles based on keywords from 
`keywords.xlsx` using the free Wikipedia API
2. **Embedding & Ingestion** : Chunks text and embeds it into a 
ChromaDB vector store using `mxbai-embed-large`
3. **Chat** : LLaMA 3.2 3b retrieves relevant context and quizzes 
the user adaptively based on their level and performance

## Features

- Adaptive difficulty based on accuracy (beginner/advanced)
- MCQ and open-ended questions
- Heuristic fallback scoring when model skips structured tags
- Weak area detection and tracking
- Multi-model support: Ollama (local), OpenAI, Anthropic
- Zero paid APIs required in default configuration

## Stack

| Component | Tool |
|---|---|
| Embeddings | mxbai-embed-large via Ollama |
| LLM | LLaMA 3.2 3b via Ollama |
| Vector Store | ChromaDB |
| RAG Framework | LangChain |
| Frontend | Streamlit |
| Data Source | Wikipedia Free API |

## Setup

### 1. Install Ollama and pull models

ollama pull mxbai-embed-large
ollama pull llama3.2:3b


### 2. Install dependencies

Create a virtual environment (recommended):

**Windows**

```bash
python -m venv venv
```
Or
```bash
venv\Scripts\activate
```
**macOS / Linux**

```bash
python3 -m venv venv
```
Or
```bash
source venv/bin/activate
```
Then:

```bash
pip install -r requirements.txt
```

### 3. Configure environment

cp .env.example .env
# Edit .env with your settings


### 4. Run the pipeline

# Step 1: Scrape Wikipedia
python scraping_the_wikis.py

# Step 2: Embed and ingest
python ingestion.py

# Step 3: Launch the tutor
streamlit run chatbot.py


## Customization

Edit `keywords.xlsx` to add your own topics and control 
how many Wikipedia pages are fetched per keyword.

## Model Alternatives

See `.env` for switching between Ollama, OpenAI, and Anthropic models.

Conclusion: Unlike typical chatbots, this system actively tests the user through quizzes, evaluates answers, and adapts based on performance!