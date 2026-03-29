import os, re, gradio as gr
from openai import OpenAI
from pinecone import Pinecone

# ── Config ────────────────────────────────────────────────────
OPENAI_API_KEY   = os.environ["OPENAI_API_KEY"]
PINECONE_API_KEY = os.environ["PINECONE_API_KEY"]
PINECONE_INDEX   = os.environ.get("PINECONE_INDEX", "cbse-class12-from-pc2")
EMBED_MODEL = "text-embedding-ada-002"
LLM_MODEL   = "gpt-4o"
TOP_K       = 8

ALL_NAMESPACES = [
    "accountancy", "biology", "biotechnology", "business-studies",
    "chemistry", "computer-science", "english", "informatics-practices",
    "mathematics", "physics", "political-science", "psychology"
]

SUBJECT_NAMESPACE_MAP = {
    "All Subjects": None, "Biology": "biology",
    "Biotechnology": "biotechnology", "Chemistry": "chemistry",
    "Physics": "physics", "Mathematics": "mathematics",
    "Computer Science": "computer-science",
    "Informatics Practices": "informatics-practices",
    "English": "english", "Accountancy": "accountancy",
    "Business Studies": "business-studies",
    "Political Science": "political-science",
    "Psychology": "psychology",
}

# ── Token tracking ────────────────────────────────────────────
token_usage = {
    "embed_tokens": 0,
    "input_tokens": 0,
    "output_tokens": 0,
    "total_cost": 0.0,
    "requests": 0,
}
query_log = []

COST_PER_1M_INPUT  = 2.50
COST_PER_1M_OUTPUT = 10.00
COST_PER_1M_EMBED  = 0.10

# ── Clients ───────────────────────────────────────────────────
oai_client = OpenAI(api_key=OPENAI_API_KEY)
index      = Pinecone(api_key=PINECONE_API_KEY).Index(PINECONE_INDEX)

# ── Warmup ────────────────────────────────────────────────────
print("🔍 Warming up Pinecone...")
try:
    stats = index.describe_index_stats()
    print(f"✅ {stats.total_vector_count} vectors ready across {len(stats.namespaces)} namespaces")
except Exception as e:
    print(f"⚠️  Pinecone warmup error: {e}")

# ── RAG functions ─────────────────────────────────────────────
def parse_filters(question):
    filters = {}
    m = re.search(r"chapter\s*(\d+)", question.lower())
    if m: filters["chapter"] = m.group(1)
    m = re.search(r"(?:page|pg)\.?\s*(\d+)", question.lower())
    if m: filters["pages_spanned"] = {"$in": [m.group(1)]}
    return filters if filters else None

def retrieve(query, namespace=None, top_k=TOP_K, filters=None):
    embed_resp = oai_client.embeddings.create(input=[query], model=EMBED_MODEL)
    vec = embed_resp.data[0].embedding
    token_usage["embed_tokens"] += embed_resp.usage.total_tokens
    token_usage["_last_embed_tokens"] = embed_resp.usage.total_tokens
    base = {"content_type": {"$eq": "text"}}
    f    = {"$and": [base, filters]} if filters else base
    kwargs = {"vector": vec, "top_k": top_k, "include_metadata": True, "filter": f}
    if namespace:
        kwargs["namespace"] = namespace
        return index.query(**kwargs).matches
    all_m = []
    for ns in ALL_NAMESPACES:
        kwargs["namespace"] = ns
        all_m.extend(index.query(**kwargs).matches)
    return sorted(all_m, key=lambda x: x.score, reverse=True)[:top_k]

def format_context(matches):
    if not matches: return "No relevant content found."
    return "\n\n---\n\n".join(
        f"[{i}] {m.metadata.get('subject')} | {m.metadata.get('chapter_title')} | Page {m.metadata.get('page')}\n{m.metadata.get('text','')}"
        for i, m in enumerate(matches, 1)
    )

def format_sources(matches):
    seen, lines = set(), []
    for m in matches:
        k = f"📖 {m.metadata.get('subject')} | {m.metadata.get('chapter_title')} | Page {m.metadata.get('page')}"
        if k not in seen: seen.add(k); lines.append(k)
    return "\n".join(lines)

def build_usage_table():
    rows = ""
    for i, q in enumerate(query_log, 1):
        rows += (
            f"| {i} | {q['query']} | {q['embed']:,} | {q['input']:,} "
            f"| {q['output']:,} | ${q['cost']:.4f} |\n"
        )
    totals_row = (
        f"| | **Total** | **{token_usage['embed_tokens']:,}** "
        f"| **{token_usage['input_tokens']:,}** | **{token_usage['output_tokens']:,}** "
        f"| **${token_usage['total_cost']:.4f}** |"
    )
    return (
        "| # | Query | Embed | Input | Output | Cost |\n"
        "|---|-------|------:|------:|-------:|-----:|\n"
        + rows + totals_row
    )

SYSTEM_PROMPT = """You are a dedicated CBSE Class XII study assistant.
Answer ONLY from the retrieved NCERT textbook content.
Always cite subject, chapter name and page number.
Never use knowledge outside the provided context."""

# ── Chat function ─────────────────────────────────────────────
def respond(message, history, subject):
    ns      = SUBJECT_NAMESPACE_MAP.get(subject)
    filters = parse_filters(message)
    matches = retrieve(message, namespace=ns, filters=filters)
    context = format_context(matches)
    sources = format_sources(matches)

    clean_history = []
    for h in history:
        if isinstance(h, dict):
            clean_history.append(h)
        elif isinstance(h, (list, tuple)) and len(h) == 2:
            if h[0]: clean_history.append({"role": "user",      "content": h[0]})
            if h[1]: clean_history.append({"role": "assistant", "content": h[1]})

    resp = oai_client.chat.completions.create(
        model=LLM_MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            *clean_history,
            {"role": "user", "content": f"Question: {message}\n\nContext:\n{context}"},
        ],
        temperature=0.1,
        max_tokens=700,
    )

    usage = resp.usage
    embed_used = token_usage.get("_last_embed_tokens", 0)
    token_usage["input_tokens"]  += usage.prompt_tokens
    token_usage["output_tokens"] += usage.completion_tokens
    token_usage["requests"] += 1
    token_usage["total_cost"] = (
        token_usage["input_tokens"]  * COST_PER_1M_INPUT  / 1_000_000
        + token_usage["output_tokens"] * COST_PER_1M_OUTPUT / 1_000_000
        + token_usage["embed_tokens"]  * COST_PER_1M_EMBED  / 1_000_000
    )
    query_cost = (
        usage.prompt_tokens      * COST_PER_1M_INPUT  / 1_000_000
        + usage.completion_tokens * COST_PER_1M_OUTPUT / 1_000_000
        + embed_used              * COST_PER_1M_EMBED  / 1_000_000
    )
    query_log.append({
        "query":  message[:50] + ("..." if len(message) > 50 else ""),
        "embed":  embed_used,
        "input":  usage.prompt_tokens,
        "output": usage.completion_tokens,
        "cost":   query_cost,
    })

    answer = resp.choices[0].message.content.strip()
    yield f"{answer}\n\n---\n{sources}" if sources else answer, build_usage_table()

# ── UI ────────────────────────────────────────────────────────
SUBJECTS = [
    "All Subjects", "Biology", "Biotechnology", "Chemistry", "Physics",
    "Mathematics", "Computer Science", "Informatics Practices", "English",
    "Accountancy", "Business Studies", "Political Science", "Psychology",
]

with gr.Blocks(title="📚 CBSE Class XII Learning Buddy") as demo:
    gr.Markdown("# 📚 CBSE Class XII Learning Buddy\nPowered by official NCERT Class XII textbooks · Ask anything from your syllabus")

    token_display = gr.Markdown(value="*No requests yet*", render=False)

    chat = gr.ChatInterface(
        respond,
        chatbot=gr.Chatbot(height=850),
        additional_inputs=[
            gr.Dropdown(choices=SUBJECTS, value="All Subjects", label="Filter by Subject"),
        ],
        additional_outputs=[token_display],
        examples=[
            ["What is meiosis?"],
            ["Explain double fertilisation"],
            ["What is Coulomb's law?"],
            ["What is a stack in Python?"],
            ["Explain cash flow statement"],
            ["What is the Cold War?"],
        ],
    )

    gr.Markdown("---")
    gr.Markdown("### 📊 Token Usage")
    token_display.render()

if __name__ == "__main__":
    demo.launch()   # ← HF Spaces handles port/host automatically