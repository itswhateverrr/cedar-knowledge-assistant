# app.py
# RAG backend for the NGO knowledge assistant.
# Retrieval: ChromaDB + sentence-transformers. Generation: Claude.
# Served via FastAPI.

import os
import glob
import ast
import operator
import time
import hmac
import json
import logging
import httpx
from fastapi import FastAPI, Header, HTTPException
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import chromadb
from sentence_transformers import SentenceTransformer
import anthropic
import pdfplumber
from mock_docs import NGO_DOCUMENTS  # still used for title/category metadata


from dotenv import load_dotenv

load_dotenv()

# Own handler instead of logging.basicConfig: libraries imported above may
# already have configured the root logger, which turns basicConfig into a no-op.
logger = logging.getLogger("agent")
logger.setLevel(logging.INFO)
logger.propagate = False
_handler = logging.StreamHandler()
_handler.setFormatter(logging.Formatter("%(message)s"))
logger.addHandler(_handler)

ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY")
if not ANTHROPIC_API_KEY:
    raise RuntimeError(
        "ANTHROPIC_API_KEY is not set. Create a .env file in this directory "
        "with a line: ANTHROPIC_API_KEY=your-key-here"
    )

BRAVE_API_KEY = os.environ.get("BRAVE_API_KEY")
if not BRAVE_API_KEY:
    raise RuntimeError("BRAVE_API_KEY is not set. Add it to your .env file.")

WEBHOOK_SECRET = os.environ.get("WEBHOOK_SECRET")
if not WEBHOOK_SECRET:
    raise RuntimeError("WEBHOOK_SECRET is not set. Add it to your .env file.")

DOCUMENTS_DIR = "documents"

app = FastAPI()


def extract_text_from_pdf(path: str) -> str:
    """Extract and return all text content from a PDF file."""
    text_parts = []
    with pdfplumber.open(path) as pdf:
        for page in pdf.pages:
            page_text = page.extract_text()
            if page_text:
                text_parts.append(page_text)
    return "\n".join(text_parts)


print("Initializing local embedding model (all-MiniLM-L6-v2)...")
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

print("Initializing ChromaDB (local vector store)...")
chroma_client = chromadb.Client()
collection_name = "ngo_knowledge_base"
try:
    chroma_client.delete_collection(name=collection_name)
except Exception:
    pass
collection = chroma_client.create_collection(name=collection_name)

# Build a lookup of id -> {title, category} from mock_docs.py metadata
meta_lookup = {doc["id"]: {"title": doc["title"], "category": doc["category"]} for doc in NGO_DOCUMENTS}

pdf_paths = sorted(glob.glob(os.path.join(DOCUMENTS_DIR, "*.pdf")))
if not pdf_paths:
    raise RuntimeError(
        f"No PDFs found in '{DOCUMENTS_DIR}/'. Run 'python make_pdfs.py' first "
        f"to generate the sample policy documents."
    )

print(f"Reading and extracting text from {len(pdf_paths)} PDF documents...")
ids, texts, metadatas = [], [], []
for path in pdf_paths:
    doc_id = os.path.splitext(os.path.basename(path))[0]
    extracted_text = extract_text_from_pdf(path)
    meta = meta_lookup.get(doc_id, {"title": doc_id, "category": "Uncategorized"})

    ids.append(doc_id)
    texts.append(extracted_text)
    metadatas.append({"title": meta["title"], "category": meta["category"], "source_file": os.path.basename(path)})

print("Embedding extracted text and indexing in ChromaDB...")
embeddings = embedding_model.encode(texts).tolist()
collection.add(ids=ids, embeddings=embeddings, documents=texts, metadatas=metadatas)
print(f"Ready — {len(ids)} PDF documents ingested and indexed.\n")

anthropic_client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)

# --- Tool: retrieval wrapped as a callable tool -----------------------------
# Claude decides whether/when to call this based on `description` below —
# it never runs automatically the way it did in the old pure retrieve-then-generate flow.

RETRIEVE_TOOL = {
    "name": "retrieve_documents",
    "description": (
        "Search Cedar Health Alliance's internal policy and program documents for "
        "passages relevant to a query. Use this for any question about the "
        "organization's policies, procedures, or programs — do not rely on general "
        "knowledge for those. Not needed for greetings or small talk."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "A focused search query capturing the specific information needed, e.g. 'maternal health referral procedure'.",
            }
        },
        "required": ["query"],
    },
}

# --- Tool: calculator -------------------------------------------------------
# LLMs are unreliable at precise arithmetic, so real math is offloaded to
# actual code. Deliberately NOT using eval() — that would let arbitrary
# expressions execute arbitrary Python. Instead we parse to an AST and only
# allow a whitelisted set of numeric operators to walk it.

CALCULATOR_TOOL = {
    "name": "calculate",
    "description": (
        "Evaluate a basic arithmetic expression — e.g. for budget math, staff "
        "ratios, or thresholds mentioned in policy documents. Supports "
        "+, -, *, /, **, and parentheses. Always use this for arithmetic "
        "rather than computing it yourself."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "expression": {
                "type": "string",
                "description": "A basic arithmetic expression, e.g. '1500 * 0.15' or '(200 + 340) / 4'.",
            }
        },
        "required": ["expression"],
    },
}

_ALLOWED_OPERATORS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.Pow: operator.pow,
    ast.USub: operator.neg,
    ast.UAdd: operator.pos,
}


def _eval_node(node):
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return node.value
    if isinstance(node, ast.BinOp) and type(node.op) in _ALLOWED_OPERATORS:
        return _ALLOWED_OPERATORS[type(node.op)](_eval_node(node.left), _eval_node(node.right))
    if isinstance(node, ast.UnaryOp) and type(node.op) in _ALLOWED_OPERATORS:
        return _ALLOWED_OPERATORS[type(node.op)](_eval_node(node.operand))
    raise ValueError(f"unsupported expression: {ast.dump(node)}")


def calculate(expression: str) -> str:
    """Safely evaluate a basic arithmetic expression (AST whitelist, no eval())."""
    try:
        tree = ast.parse(expression, mode="eval")
        return str(_eval_node(tree.body))
    except Exception as e:
        return f"Error: could not evaluate expression ({e})"


# --- Tool: web search (Brave Search API) ------------------------------------
# First tool that calls a service we don't control: needs auth, a timeout,
# and (next pieces) rate-limit and failure handling.

WEB_SEARCH_TOOL = {
    "name": "web_search",
    "description": (
        "Search the public web for current information that is NOT in "
        "Cedar Health Alliance's internal documents, e.g. public health "
        "guidance, news, or external statistics. Never use it for questions "
        "about the organization's own policies; use retrieve_documents."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "A concise web search query, e.g. 'WHO guidance on cholera outbreak response'.",
            }
        },
        "required": ["query"],
    },
}

BRAVE_SEARCH_URL = "https://api.search.brave.com/res/v1/web/search"


def _seconds_until_reset(headers) -> float:
    # Brave sends "per-second, per-month" pairs, e.g. "1, 979532". We only
    # need the first number: how long until the per-second window resets.
    try:
        return float(headers.get("x-ratelimit-reset", "1").split(",")[0])
    except ValueError:
        return 1.0


def web_search(query: str, count: int = 5) -> str:
    """Call the Brave Search API and return results as plain text."""
    # Each failure type gets its own message telling Claude what to do next,
    # because Claude only sees text and can't tell which errors are worth retrying.
    try:
        for attempt in range(2):
            response = httpx.get(
                BRAVE_SEARCH_URL,
                params={"q": query, "count": count},
                headers={
                    "X-Subscription-Token": BRAVE_API_KEY,
                    "Accept": "application/json",
                },
                timeout=10.0,
            )
            # 429 = "too many requests". Wait briefly (capped at 2s) and retry once.
            if response.status_code == 429 and attempt == 0:
                time.sleep(min(_seconds_until_reset(response.headers), 2.0))
                continue
            break

        if response.status_code == 429:
            return (
                "Error: web search is rate limited right now. Do not call it "
                "again; answer without web results or tell the user."
            )
        response.raise_for_status()

    except httpx.TimeoutException:
        return (
            "Error: web search timed out. You may retry once with a simpler "
            "query, otherwise answer without web results."
        )
    except httpx.HTTPStatusError as e:
        status = e.response.status_code
        if status in (401, 403):
            return (
                "Error: web search is misconfigured (authentication failed). "
                "Do not retry; tell the user web search is unavailable."
            )
        if status >= 500:
            return (
                "Error: the search service is temporarily down. Do not "
                "retry; answer without web results."
            )
        return f"Error: web search request was rejected (HTTP {status}). Do not retry."
    except httpx.RequestError:
        return (
            "Error: could not reach the search service (network problem). "
            "Do not retry; answer without web results."
        )

    results = response.json().get("web", {}).get("results", [])
    if not results:
        return "No web results found."
    return "\n\n".join(
        f"{r['title']}\n{r['url']}\n{r.get('description', '')}" for r in results
    )


# System prompts live in prompts/<version>.txt so each iteration is a
# separate, reviewable file. PROMPT_VERSION picks which one the app uses.
PROMPT_VERSION = os.environ.get("PROMPT_VERSION", "v2")


def load_system_prompt(version: str) -> str:
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "prompts", f"{version}.txt")
    if not os.path.exists(path):
        raise RuntimeError(f"Prompt version '{version}' not found at {path}")
    with open(path) as f:
        return f.read().strip()


SYSTEM_PROMPT = load_system_prompt(PROMPT_VERSION)


def retrieve_documents(query: str, n_results: int = 3) -> dict:
    """Embed a query and search the local ChromaDB vector store."""
    query_embedding = embedding_model.encode([query]).tolist()
    results = collection.query(query_embeddings=query_embedding, n_results=n_results)
    return {
        "documents": results["documents"][0],
        "metadatas": results["metadatas"][0],
    }


# Hard cap on plan->act->observe iterations per request. Without this, a
# model stuck retrying a failing tool call (or just being indecisive) could
# loop forever, burning API calls and latency on a single user request.
MAX_ITERATIONS = 5

# Output limit per Claude call. Too low and long answers get cut off mid-sentence.
MAX_OUTPUT_TOKENS = 1024


class ChatRequest(BaseModel):
    query: str


class RetrievedSource(BaseModel):
    title: str
    category: str
    snippet: str
    source_file: str


class ToolCall(BaseModel):
    name: str
    input: dict
    ok: bool
    duration_ms: int


class ChatResponse(BaseModel):
    answer: str
    sources: list[RetrievedSource]
    tool_calls: list[ToolCall] = []
    truncated: bool = False


def run_agent(query: str) -> ChatResponse:
    messages = [{"role": "user", "content": query}]
    sources: list[RetrievedSource] = []
    tool_calls: list[ToolCall] = []
    tools = [RETRIEVE_TOOL, CALCULATOR_TOOL, WEB_SEARCH_TOOL]

    # The agent loop: plan (Claude calls a tool) -> act (we run it) ->
    # observe (we feed the result, or the error, back) -> repeat. Bounded by
    # MAX_ITERATIONS so a stuck loop fails loudly instead of running forever.
    response = None
    for _ in range(MAX_ITERATIONS):
        response = anthropic_client.messages.create(
            model="claude-sonnet-5",
            max_tokens=MAX_OUTPUT_TOKENS,
            system=SYSTEM_PROMPT,
            tools=tools,
            messages=messages,
        )

        if response.stop_reason != "tool_use":
            break  # Claude produced a final answer — no more tools needed.

        messages.append({"role": "assistant", "content": response.content})

        tool_result_blocks = []
        for block in response.content:
            if block.type != "tool_use":
                continue

            started = time.perf_counter()
            try:
                if block.name == "retrieve_documents":
                    result = retrieve_documents(block.input["query"])
                    tool_output = ""
                    for meta, doc_text in zip(result["metadatas"], result["documents"]):
                        tool_output += f"\n--- Source: {meta['title']} ---\n{doc_text}\n"
                        sources.append(RetrievedSource(
                            title=meta["title"],
                            category=meta["category"],
                            snippet=doc_text[:140] + ("…" if len(doc_text) > 140 else ""),
                            source_file=meta.get("source_file", "")
                        ))

                elif block.name == "calculate":
                    tool_output = calculate(block.input["expression"])

                elif block.name == "web_search":
                    tool_output = web_search(block.input["query"])

                else:
                    tool_output = f"Error: unknown tool '{block.name}'"

            except Exception as e:
                # Observed as data, not raised: Claude sees the failure in the
                # next iteration and can retry with corrected input, instead
                # of the request crashing with a 500.
                tool_output = f"Error: tool '{block.name}' failed ({e})"

            # Our tools report failures as text starting with "Error", whether
            # they raised or returned it, so this catches both.
            call = ToolCall(
                name=block.name,
                input=block.input,
                ok=not tool_output.startswith("Error"),
                duration_ms=int((time.perf_counter() - started) * 1000),
            )
            tool_calls.append(call)
            logger.info(json.dumps({"event": "tool_call", **call.model_dump()}))

            tool_result_blocks.append({
                "type": "tool_result",
                "tool_use_id": block.id,
                "content": tool_output,
            })

        messages.append({"role": "user", "content": tool_result_blocks})
    else:
        # Loop ran MAX_ITERATIONS times without Claude ever stopping on its
        # own (the `break` above never fired) — the bounded failure path.
        return ChatResponse(
            answer=(
                "I wasn't able to finish researching this within my step "
                "limit. Please try rephrasing your question."
            ),
            sources=sources,
            tool_calls=tool_calls,
        )

    answer_text = "".join(block.text for block in response.content if block.type == "text")

    # "max_tokens" means Claude hit our output limit and was cut off mid-answer.
    truncated = response.stop_reason == "max_tokens"
    if truncated:
        logger.warning(json.dumps({"event": "answer_truncated", "max_tokens": MAX_OUTPUT_TOKENS}))

    return ChatResponse(answer=answer_text, sources=sources, tool_calls=tool_calls, truncated=truncated)


# One JSON log line per HTTP request (method, path, status, duration), in the
# same format as the tool-call logs. The health check is skipped: Docker polls
# it every 30 seconds and it would drown out real traffic.
@app.middleware("http")
async def log_requests(request, call_next):
    started = time.perf_counter()
    response = await call_next(request)
    if request.url.path != "/api/health":
        logger.info(json.dumps({
            "event": "http_request",
            "method": request.method,
            "path": request.url.path,
            "status": response.status_code,
            "duration_ms": int((time.perf_counter() - started) * 1000),
        }))
    return response


# Liveness/readiness probe for Docker and hosting platforms. Deliberately
# returns no secrets and makes no Claude or Brave calls, so it is cheap and safe
# to poll every few seconds.
@app.get("/api/health")
def health():
    return {
        "status": "ok",
        "documents_indexed": collection.count(),
        "prompt_version": PROMPT_VERSION,
    }


@app.post("/api/chat", response_model=ChatResponse)
def chat(req: ChatRequest):
    return run_agent(req.query.strip())


# Webhook: a second entry point for other systems (e.g. n8n). The shared
# secret in the X-Webhook-Secret header is the lock on the door.
@app.post("/api/webhook/ask", response_model=ChatResponse)
def webhook_ask(req: ChatRequest, x_webhook_secret: str = Header(default="")):
    # compare_digest takes the same time however many characters match,
    # so an attacker can't guess the secret one character at a time.
    if not hmac.compare_digest(x_webhook_secret, WEBHOOK_SECRET):
        raise HTTPException(status_code=401, detail="Invalid webhook secret")
    return run_agent(req.query.strip())


# Serve the generated PDFs so the frontend can link directly to sources
app.mount("/documents", StaticFiles(directory="documents"), name="documents")

# Serve the frontend
app.mount("/", StaticFiles(directory="static", html=True), name="static")
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)