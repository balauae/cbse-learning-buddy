# CBSE Class XII Learning Buddy

A RAG-powered chatbot that answers questions from official NCERT Class XII textbooks using Pinecone vector search and GPT-4o.

## Deploy to Hugging Face Spaces

### Prerequisites

- A [Hugging Face](https://huggingface.co) account
- API keys for **OpenAI** and **Pinecone**
- A Pinecone index with NCERT textbook embeddings already ingested (see `cbse_buddy_v4.ipynb` in the root of this repo for the ingestion pipeline)
- `huggingface_hub` Python package installed:
  ```bash
  pip install huggingface_hub
  ```

### Step 1: Log in to Hugging Face

```python
from huggingface_hub import login
login()
```

This opens a browser prompt to authenticate with your HF token.

### Step 2: Create a Gradio Space

```python
from huggingface_hub import HfApi

api      = HfApi()
username = api.whoami()["name"]
repo_id  = f"{username}/cbse-12th-buddy"

api.create_repo(
    repo_id=repo_id,
    repo_type="space",
    space_sdk="gradio",
    exist_ok=True,
)
print(f"https://huggingface.co/spaces/{repo_id}")
```

### Step 3: Add secrets

Add your API keys as Space secrets so they're available as environment variables at runtime:

```python
api.add_space_secret(repo_id=repo_id, key="OPENAI_API_KEY",   value="sk-...")
api.add_space_secret(repo_id=repo_id, key="PINECONE_API_KEY", value="pcsk_...")
api.add_space_secret(repo_id=repo_id, key="PINECONE_INDEX",   value="cbse-class12-from-pc2")
```

### Step 4: Upload app files

```python
api.upload_file(
    path_or_fileobj="app.py",
    path_in_repo="app.py",
    repo_id=repo_id,
    repo_type="space",
)
api.upload_file(
    path_or_fileobj="requirements.txt",
    path_in_repo="requirements.txt",
    repo_id=repo_id,
    repo_type="space",
)
print("Files uploaded")
```

The Space will automatically rebuild and start serving the app.

### Step 5: Verify

Visit `https://huggingface.co/spaces/<username>/cbse-12th-buddy` to confirm the app is running.

## Run locally

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

2. Copy `.env.local` to `.env` and fill in your API keys:
   ```bash
   cp .env.local .env
   ```

3. Run:
   ```bash
   python app.local.py
   ```

   `app.local.py` uses `python-dotenv` to load keys from `.env`. `app.py` reads directly from environment variables (as set by HF Spaces secrets).

## Deleting the Space

```python
api.delete_repo(repo_id=repo_id, repo_type="space")
```

## Files

| File | Description |
|------|-------------|
| `app.py` | Production app (reads env vars set by HF Spaces) |
| `app.local.py` | Local dev version (loads `.env` via dotenv) |
| `requirements.txt` | Python dependencies |
| `.env.local` | Environment variable template |
