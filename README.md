# CBSE Class XII Learning Buddy

A RAG-powered study assistant built on official NCERT Class XII textbooks. Ask questions from your syllabus and get accurate, cited answers.

**Live Demo**: https://huggingface.co/spaces/balaprasannav2009/cbse-12th-buddy

## Features

- **RAG pipeline** — Retrieves relevant textbook passages from a Pinecone vector store and answers using GPT-4o
- **12 subjects supported** — Biology, Chemistry, Physics, Mathematics, Computer Science, English, Accountancy, Business Studies, Political Science, Psychology, Biotechnology, and Informatics Practices
- **Subject filtering** — Narrow your search to a specific subject or search across all
- **Chapter & page filters** — Mention a chapter number or page in your question for targeted results
- **Source citations** — Every answer includes subject, chapter, and page references
- **Token usage tracking** — Monitor embedding, input, and output token costs per query

## Tech Stack

- **LLM**: OpenAI GPT-4o
- **Embeddings**: text-embedding-ada-002
- **Vector DB**: Pinecone
- **UI**: Gradio
- **Deployment**: Hugging Face Spaces

## Setup

1. Clone the repo:
   ```bash
   git clone https://github.com/balauae/cbse-learning-buddy.git
   cd cbse-learning-buddy
   ```

2. Install dependencies:
   ```bash
   pip install -r cbse-12th-buddy/requirements.txt
   ```

3. Copy the env template and add your keys:
   ```bash
   cp cbse-12th-buddy/.env.local cbse-12th-buddy/.env
   ```
   Edit `.env` with your OpenAI and Pinecone API keys.

4. Run the app:
   ```bash
   cd cbse-12th-buddy
   python app.py
   ```

## Project Structure

```
├── cbse-12th-buddy/       # Gradio web app
│   ├── app.py             # Main application
│   ├── app.local.py       # Local development version
│   ├── requirements.txt   # Python dependencies
│   └── .env.local         # Environment variable template
└── cbse_buddy_v4.ipynb    # Data ingestion & embedding notebook
```
