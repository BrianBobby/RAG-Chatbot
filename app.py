# app.py
# Flask app for chat UI
# Requirements:
# pip install flask python-dotenv langchain_openai pinecone-client

import os
from dotenv import load_dotenv
from flask import Flask, render_template, request, jsonify
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_core.prompts import PromptTemplate
from pinecone import Pinecone

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
INDEX_NAME = os.getenv("PINECONE_INDEX", "citibusiness-manual")
NAMESPACE = os.getenv("PINECONE_NAMESPACE", "")  # must match ingest (or empty)

if not OPENAI_API_KEY or not PINECONE_API_KEY:
    raise RuntimeError("OPENAI_API_KEY and PINECONE_API_KEY required in .env")

# Settings (tweak if needed)
TOP_K = 8
CONFIDENCE_THRESHOLD = 0.35
EMB_MODEL = "text-embedding-3-small"
LLM_MODEL = "gpt-4o-mini"

# initialize models and index client once on startup
os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY
emb = OpenAIEmbeddings(model=EMB_MODEL)
llm = ChatOpenAI(model=LLM_MODEL, temperature=0)

pc = Pinecone(api_key=PINECONE_API_KEY)
index = pc.Index(INDEX_NAME)

PROMPT = PromptTemplate.from_template("""
You are a banking compliance assistant.

Answer using ONLY the CONTEXT below.
If the answer is not present in the context, reply exactly:
"Not found in the provided document."

CONTEXT:
{context}

QUESTION:
{question}

Answer clearly and concisely.
""")

app = Flask(__name__, static_folder="static", template_folder="templates")


def retrieve_raw_matches(query: str, top_k: int = TOP_K):
    q_emb = emb.embed_query(query)
    resp = index.query(vector=q_emb, top_k=top_k, include_metadata=True, namespace=NAMESPACE or None)
    matches = resp.get("matches", []) if isinstance(resp, dict) else getattr(resp, "matches", [])
    return matches


def build_context_from_matches(matches):
    blocks = []
    unique_sections = []
    for m in matches:
        meta = m.get("metadata", {}) if isinstance(m, dict) else m.metadata
        section = meta.get("section", "Unknown")
        text = meta.get("text", meta.get("text_preview", ""))
        blocks.append(f"[Section: {section}]\n{text}")
        if section not in unique_sections:
            unique_sections.append(section)
    context = "\n\n".join(blocks)
    return context, unique_sections


def answer_question(query: str):
    matches = retrieve_raw_matches(query, top_k=TOP_K)
    if not matches:
        return "No relevant information found.", []

    scores = [m.get("score", 0.0) if isinstance(m, dict) else getattr(m, "score", 0.0) for m in matches]
    max_score = max(scores) if scores else 0.0

    if max_score < CONFIDENCE_THRESHOLD:
        return "Not found in the provided document.", matches

    context, unique_sections = build_context_from_matches(matches)
    final_prompt = PROMPT.format(context=context, question=query)
    resp = llm.invoke(final_prompt)
    answer_text = resp.content if hasattr(resp, "content") else str(resp)
    # append sources footer
    sources_line = "Sources: " + "; ".join(unique_sections) if unique_sections else ""
    if sources_line:
        answer_text = answer_text.strip() + "\n\n" + sources_line

    return answer_text, matches


@app.route("/")
def index_page():
    return render_template("index.html")


@app.route("/ask", methods=["POST"])
def ask():
    data = request.json
    question = data.get("question", "").strip()
    if not question:
        return jsonify({"error": "empty question"}), 400

    try:
        answer, matches = answer_question(question)
        # Create compact debug info (optional)
        debug = []
        for m in matches:
            meta = m.get("metadata", {}) if isinstance(m, dict) else m.metadata
            score = m.get("score", None) if isinstance(m, dict) else m.score
            debug.append({
                "id": m.get("id", getattr(m, "id", None)),
                "score": score,
                "section": meta.get("section")
            })
        return jsonify({"answer": answer, "debug": debug})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    # Use debug=False in production
    app.run(host="0.0.0.0", port=5000, debug=True)
