# EmailGen

AI-powered email generator using the **reflection pattern**: a drafter agent writes the initial email and a revisor agent improves it for clarity, tone, and correctness.

Built with [AgenticBlocks](https://github.com/agenticblocks/agenticblocks) and supports multiple LLM providers (Gemini, OpenAI, Anthropic, Groq, Cohere) as well as local models via Ollama.

## Scripts

| File | Description |
|------|-------------|
| `reflection_emailcontgen_ui.py` | Gradio web UI — pick provider/model and fill email fields |
| `reflection_emailcontgen.py` | CLI version with the reflection workflow (drafter + revisor) |
| `oneway_emailcontgen.py` | CLI version — single drafter agent (no revision step) |

## Setup

```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Usage

### Web UI

```bash
python reflection_emailcontgen_ui.py
```

Opens a Gradio interface at `http://localhost:7860`. Choose between an API provider (Gemini, OpenAI, Anthropic, Groq, Cohere) or a local Ollama model, fill in the email details, and click **Gerar e-mail**.

### CLI (reflection)

```bash
python reflection_emailcontgen.py
```

Uses `ollama/gemma3:4b` by default. Edit `model_name` in the script to switch providers.

### CLI (one-way)

```bash
GEMINI_API_KEY=your_key python oneway_emailcontgen.py
```

## Environment Variables

Set the API key for your chosen provider:

| Provider  | Variable          |
|-----------|-------------------|
| Gemini    | `GEMINI_API_KEY`  |
| OpenAI    | `OPENAI_API_KEY`  |
| Anthropic | `ANTHROPIC_API_KEY` |
| Groq      | `GROQ_API_KEY`    |
| Cohere    | `COHERE_API_KEY`  |

Ollama requires no API key — just have it running locally on `localhost:11434`.
