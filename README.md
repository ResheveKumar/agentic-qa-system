# Agentic Codebase & Knowledge QA System
### Custom MCP + Groq (llama-3.3-70b-versatile) + LanceDB RAG + LangGraph

An end-to-end, production-ready reference system combining:

- **Custom MCP Server** (`mcp_server.py`) — exposes local file reading, directory
  listing, and system context as MCP tools using the official `mcp` Python SDK.
- **Local RAG Engine** (`rag_engine.py`) — chunks and embeds source files with
  `sentence-transformers` (`all-MiniLM-L6-v2`) and stores/searches vectors in an
  embedded **LanceDB** table (no external DB server required).
- **2-Agent LangGraph Workflow** (`agent_graph.py`) — a `StateGraph` with two
  nodes powered by `ChatGroq` (`llama-3.3-70b-versatile`):
  1. **Code Reviewer & Bug Finder** — analyzes code for bugs, edge cases,
     security/performance issues.
  2. **Unit Test Generator** — writes a full test suite informed by the review.
- **Streamlit UI** (`app.py`) — ties everything together: ingest a codebase,
  semantically search it, and run the agent workflow on pasted code.

---

## 1. Project Structure

```
.
├── requirements.txt      # All dependencies
├── mcp_server.py         # Custom MCP server (file/context tools)
├── rag_engine.py         # LanceDB + sentence-transformers RAG pipeline
├── agent_graph.py        # LangGraph 2-agent workflow (ChatGroq)
├── app.py                # Streamlit web UI (main entry point)
└── README.md             # This file
```

---

## 2. Prerequisites

- Python 3.10+
- A **free Groq API key** (Groq's LPU inference is free-tier friendly and fast)

### Getting a free Groq API key
1. Go to **https://console.groq.com/keys**
2. Sign in (Google/GitHub/email).
3. Click **"Create API Key"**, name it, and copy the value (starts with `gsk_...`).
4. You will paste this into a `.env` file in step 4 below.

---

## 3. Installation

```bash
# 1. Create and activate a virtual environment (recommended)
python -m venv venv
source venv/bin/activate        # On Windows: venv\Scripts\activate

# 2. Install dependencies
pip install -r requirements.txt
```

> Note: The first run will download the `all-MiniLM-L6-v2` sentence-transformer
> model (~90MB) from Hugging Face — this requires an internet connection once,
> after which it is cached locally.

---

## 4. Configuration

Create a `.env` file in the project root:

```bash
echo "GROQ_API_KEY=your_actual_groq_key_here" > .env
```

Or manually create `.env` with:

```
GROQ_API_KEY=gsk_xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
```

---

## 5. Running the Components

### A. Run the Custom MCP Server (standalone)

The MCP server exposes `read_file`, `list_directory`, `list_source_files`, and
`get_system_context` as MCP tools over stdio. It can be launched directly to
verify it starts correctly, or attached to any MCP-compatible client:

```bash
python mcp_server.py
```

This will block and listen on stdio — it's meant to be launched by an MCP
client (e.g. Claude Desktop's `mcpServers` config, or a custom LangChain MCP
adapter), not used interactively in a terminal. To wire it into an MCP client
config, point the client at:

```json
{
  "mcpServers": {
    "codebase-context-server": {
      "command": "python",
      "args": ["/absolute/path/to/mcp_server.py"]
    }
  }
}
```

### B. Test the RAG engine standalone

```bash
python rag_engine.py
```

This ingests the current directory into a local LanceDB store
(`./lancedb_store/`) and runs a sample search query.

### C. Test the 2-agent LangGraph workflow standalone

```bash
python agent_graph.py
```

This runs both agents (Code Reviewer → Test Generator) against a small sample
snippet and prints the review and generated tests to the console.

### D. Run the full Streamlit application (recommended entry point)

```bash
streamlit run app.py
```

This opens a browser UI with three tabs:

1. **📥 Ingest Codebase** — point it at any local directory to chunk, embed,
   and index its `.py/.md/.txt/.json/.yaml` files into LanceDB.
2. **🔍 Semantic Search** — query the indexed codebase using natural language;
   returns the most semantically relevant chunks with distance scores.
3. **🤖 Review & Test Generation** — paste any code snippet, optionally
   augment it with retrieved RAG context, and run the 2-agent LangGraph
   workflow to get a full code review followed by a generated unit test suite.

---

## 6. How It Fits Together

```
                ┌─────────────────────┐
                │   mcp_server.py      │  (MCP tools: read_file, list_directory,
                │   (stdio transport)  │   list_source_files, get_system_context)
                └──────────┬───────────┘
                           │ (attachable to any MCP client)
                           │
┌──────────────────────────┴───────────────────────────────┐
│                        app.py (Streamlit)                 │
│                                                            │
│   ┌────────────────┐        ┌────────────────────────┐    │
│   │  rag_engine.py  │──────▶│   agent_graph.py         │   │
│   │  LanceDB +      │ ctx   │   LangGraph StateGraph   │   │
│   │  sentence-      │       │   Node 1: Code Reviewer  │   │
│   │  transformers   │       │   Node 2: Test Generator │   │
│   └────────────────┘        │   (ChatGroq: llama-3.3-  │   │
│                              │    70b-versatile)        │   │
│                              └────────────────────────┘    │
└─────────────────────────────────────────────────────────┘
```

- The **RAG engine** independently indexes and retrieves codebase context.
- The **agent graph** optionally consumes that retrieved context to ground its
  code review and test generation in the actual codebase.
- The **MCP server** operates as a separate, standards-compliant tool provider
  that any MCP client (this app or external ones) can use for raw file/system
  access.

---

## 7. Troubleshooting

| Issue | Fix |
|---|---|
| `GROQ_API_KEY is not set` | Ensure `.env` exists in the same directory you run commands from, and contains `GROQ_API_KEY=...`. |
| `No supported files found` during ingestion | Confirm the directory path contains `.py/.md/.txt/.json/.yaml` files and isn't excluded (`.git`, `venv`, etc.). |
| Slow first run | The embedding model downloads on first use; subsequent runs use the local cache. |
| LanceDB table errors | Delete the `lancedb_store/` folder to reset the vector index and re-ingest. |
| Groq rate limits / API errors | Check https://console.groq.com for current free-tier rate limits; the app surfaces API errors directly in the UI. |

---

## 8. Extending This System

- Add more MCP tools (e.g. `run_tests`, `git_diff`) to `mcp_server.py`.
- Add a third LangGraph node (e.g. a "Fix Suggester" agent) between review and
  test generation.
- Swap `all-MiniLM-L6-v2` for a larger embedding model in `rag_engine.py` for
  higher retrieval accuracy at the cost of speed.
- Persist agent run history to LanceDB as a second table for auditability.
