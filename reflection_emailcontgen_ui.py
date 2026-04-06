import asyncio
import os
import requests
import gradio as gr
from pydantic import BaseModel

from agenticblocks.core.block import Block
from agenticblocks.core.graph import WorkflowGraph
from agenticblocks.runtime.executor import WorkflowExecutor
from agenticblocks.blocks.llm.agent import LLMAgentBlock, AgentInput


# ---------------------------------------------------------------------------
# Configuração de provedores e modelos sugeridos
# ---------------------------------------------------------------------------

PROVIDER_ENV_VARS: dict[str, str] = {
    "Gemini":    "GEMINI_API_KEY",
    "OpenAI":    "OPENAI_API_KEY",
    "Anthropic": "ANTHROPIC_API_KEY",
    "Groq":      "GROQ_API_KEY",
    "Cohere":    "COHERE_API_KEY",
}

PROVIDER_DEFAULT_MODELS: dict[str, list[str]] = {
    "Gemini":    ["gemini/gemini-2.0-flash", "gemini/gemini-2.5-pro-preview-03-25", "gemini/gemini-1.5-pro"],
    "OpenAI":    ["gpt-4o", "gpt-4o-mini", "gpt-4-turbo"],
    "Anthropic": ["claude-opus-4-5-20251101", "claude-sonnet-4-5-20251022", "claude-haiku-4-5-20251001"],
    "Groq":      ["groq/llama-3.3-70b-versatile", "groq/llama-3.1-8b-instant"],
    "Cohere":    ["cohere/command-r-plus", "cohere/command-r"],
}


def get_ollama_models() -> list[str]:
    """Consulta o Ollama local (localhost:11434) e retorna os modelos disponíveis."""
    try:
        resp = requests.get("http://localhost:11434/api/tags", timeout=3)
        if resp.ok:
            data = resp.json()
            return [f"ollama/{m['name']}" for m in data.get("models", [])]
    except Exception:
        pass
    return []


# ---------------------------------------------------------------------------
# Bloco bridge (mesmo do script original)
# ---------------------------------------------------------------------------

class AgentOutputToInput(BaseModel):
    response: str


class AgentBridgeBlock(Block[AgentOutputToInput, AgentInput]):
    def __init__(self):
        super().__init__(name="AgentBridge", description="Maps agent response to next agent prompt.")

    async def run(self, input: AgentOutputToInput) -> AgentInput:
        return AgentInput(prompt=input.response)


# ---------------------------------------------------------------------------
# Lógica principal
# ---------------------------------------------------------------------------

async def _run_workflow(model: str, subject: str, body: str, recipient: str, sender: str) -> str:
    graph = WorkflowGraph()

    draft_agent = LLMAgentBlock(
        name="DraftEmailAgent",
        model=model,
        description="An agent that drafts an email based on the provided input.",
        system_prompt=(
            "You are an assistant that drafts emails based on the provided "
            "subject, body, recipient name, and sender name."
        ),
    )

    revisor_agent = LLMAgentBlock(
        name="RevisorAgent",
        model=model,
        description="An agent that revises the drafted email",
        system_prompt=(
            "You are an assistant that revises emails for clarity, tone, and correctness, "
            "removing repeated words and fixing typos. Also, complete the email if it is not complete. "
            "Return only the email content without any additional text."
        ),
    )

    bridge = AgentBridgeBlock()

    drafter_node = graph.add_block(draft_agent)
    bridge_node  = graph.add_block(bridge)
    revisor_node = graph.add_block(revisor_agent)
    graph.connect(drafter_node, bridge_node)
    graph.connect(bridge_node,  revisor_node)

    executor = WorkflowExecutor(graph)

    prompt = (
        f"Draft an email with the subject '{subject}', "
        f"body '{body}', "
        f"recipient name '{recipient}', "
        f"and sender name '{sender}'."
    )

    ctx = await executor.run(initial_input={"prompt": prompt})
    output = ctx.get_output("RevisorAgent")
    return output.response


def generate_email(
    model_source: str,
    # API fields
    api_provider: str,
    api_key: str,
    api_model_choice: str,
    api_model_custom: str,
    # Local fields
    local_model_choice: str,
    local_model_custom: str,
    # Email fields
    subject: str,
    body: str,
    recipient: str,
    sender: str,
) -> str:
    if not subject.strip() or not recipient.strip() or not sender.strip():
        return "Preencha pelo menos Assunto, Destinatário e Remetente."

    # Resolve o modelo final
    if model_source == "API":
        final_model = api_model_custom.strip() if api_model_custom.strip() else api_model_choice
        if not final_model:
            return "Selecione ou informe um modelo de API."

        # Define a variável de ambiente com a chave fornecida
        if api_key.strip():
            env_var = PROVIDER_ENV_VARS.get(api_provider, "OPENAI_API_KEY")
            os.environ[env_var] = api_key.strip()
        else:
            env_var = PROVIDER_ENV_VARS.get(api_provider, "")
            if env_var and not os.getenv(env_var):
                return f"Informe a API Key (variável {env_var} não encontrada no ambiente)."
    else:  # Local / Ollama
        final_model = local_model_custom.strip() if local_model_custom.strip() else local_model_choice
        if not final_model:
            return "Selecione ou informe um modelo local."

    try:
        result = asyncio.run(_run_workflow(final_model, subject, body, recipient, sender))
        return result
    except Exception as e:
        return f"Erro ao executar o workflow:\n{e}"


# ---------------------------------------------------------------------------
# Interface Gradio
# ---------------------------------------------------------------------------

def update_model_choices(provider: str):
    models = PROVIDER_DEFAULT_MODELS.get(provider, [])
    return gr.update(choices=models, value=models[0] if models else None)


def update_local_model_choices():
    models = get_ollama_models()
    if models:
        return gr.update(choices=models, value=models[0], info="Modelos detectados no Ollama local.")
    return gr.update(choices=[], value=None, info="Nenhum modelo Ollama encontrado. Use o campo abaixo.")


def toggle_model_panels(choice: str):
    return (
        gr.update(visible=choice == "API"),
        gr.update(visible=choice == "Local (Ollama)"),
    )


with gr.Blocks(title="EmailGen — Reflection") as demo:
    gr.Markdown("# EmailGen com Reflection\nGerador de e-mails com agente rascunhador + agente revisor.")

    # ---- Seleção de fonte do modelo ----
    model_source = gr.Radio(
        choices=["API", "Local (Ollama)"],
        value="API",
        label="Fonte do modelo",
    )

    # ---- Painel API ----
    with gr.Group(visible=True) as api_panel:
        gr.Markdown("### Configuração — Modelo via API")
        with gr.Row():
            api_provider = gr.Dropdown(
                choices=list(PROVIDER_ENV_VARS.keys()),
                value="Gemini",
                label="Provedor",
            )
            api_key_input = gr.Textbox(
                label="API Key",
                placeholder="Deixe em branco para usar a variável de ambiente",
                type="password",
            )
        api_model_choice = gr.Dropdown(
            choices=PROVIDER_DEFAULT_MODELS["Gemini"],
            value=PROVIDER_DEFAULT_MODELS["Gemini"][0],
            label="Modelo sugerido",
        )
        api_model_custom = gr.Textbox(
            label="Modelo personalizado (substitui a seleção acima)",
            placeholder="ex: gemini/gemini-2.0-flash-lite",
        )

    # ---- Painel Local ----
    with gr.Group(visible=False) as local_panel:
        gr.Markdown("### Configuração — Modelo Local (Ollama)")
        with gr.Row():
            refresh_btn = gr.Button("Atualizar lista de modelos", scale=0)
        local_model_choice = gr.Dropdown(
            choices=[],
            label="Modelos disponíveis no Ollama",
            info="Clique em 'Atualizar' para detectar modelos locais.",
        )
        local_model_custom = gr.Textbox(
            label="Modelo personalizado (substitui a seleção acima)",
            placeholder="ex: ollama/llama3.2",
        )

    # ---- Campos do e-mail ----
    gr.Markdown("### Conteúdo do e-mail")
    with gr.Row():
        subject_input   = gr.Textbox(label="Assunto",     placeholder="Meeting Reminder")
        recipient_input = gr.Textbox(label="Destinatário", placeholder="John Doe")
        sender_input    = gr.Textbox(label="Remetente",    placeholder="Jane Smith")
    body_input = gr.Textbox(
        label="Corpo (esboço)",
        placeholder="Don't forget about our meeting tomorrow at 10 AM...",
        lines=4,
    )

    generate_btn = gr.Button("Gerar e-mail", variant="primary")

    output_box = gr.Textbox(label="E-mail gerado", lines=12, interactive=False)

    # ---- Eventos ----
    model_source.change(toggle_model_panels, inputs=model_source, outputs=[api_panel, local_panel])
    api_provider.change(update_model_choices, inputs=api_provider, outputs=api_model_choice)
    refresh_btn.click(update_local_model_choices, outputs=local_model_choice)

    generate_btn.click(
        fn=generate_email,
        inputs=[
            model_source,
            api_provider, api_key_input, api_model_choice, api_model_custom,
            local_model_choice, local_model_custom,
            subject_input, body_input, recipient_input, sender_input,
        ],
        outputs=output_box,
    )

if __name__ == "__main__":
    demo.launch()
