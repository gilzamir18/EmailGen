"""
EmailGen — Reflexão Interna Embutida no Bloco Principal
========================================================
Estratégia: um único LLMAgentBlock com o EmailRubricValidatorBlock
injetado como tool. O while-True nativo do LLMAgentBlock faz o ciclo
Ação → Crítica → Refinamento sem retroceder no Grafo global.

Rúbrica binária (email):
  1. O email possui saudação e despedida?
     → Inclua abertura e fechamento adequados ao contexto.
  2. O tom é compatível com o destinatário?
     → Ajuste o nível de formalidade conforme o perfil do receptor.
  3. A solicitação principal está explícita?
     → Reformule o parágrafo central com chamada à ação clara.
"""

import asyncio
import json
import re

import litellm
from pydantic import BaseModel, Field

from agenticblocks.core.block import Block
from agenticblocks.core.graph import WorkflowGraph
from agenticblocks.runtime.executor import WorkflowExecutor
from agenticblocks.blocks.llm.agent import LLMAgentBlock, AgentInput


# ---------------------------------------------------------------------------
# Rúbrica
# ---------------------------------------------------------------------------

RUBRIC: list[dict] = [
    {
        "criterion": "O email possui saudação e despedida?",
        "instruction": "Inclua abertura e fechamento adequados ao contexto.",
    },
    {
        "criterion": "O tom é compatível com o destinatário?",
        "instruction": "Ajuste o nível de formalidade conforme o perfil do receptor.",
    },
    {
        "criterion": "A solicitação principal está explícita?",
        "instruction": "Reformule o parágrafo central com chamada à ação clara.",
    },
]


# ---------------------------------------------------------------------------
# Validator Block — Ferramenta Crítica e Injetadora de Restrições
# ---------------------------------------------------------------------------

class EmailValidationInput(BaseModel):
    email_text: str = Field(
        description="The full email text to be validated against the rubric."
    )
    recipient_context: str = Field(
        description=(
            "Brief context about the recipient and purpose, e.g. "
            "'formal business meeting reminder for a client'."
        )
    )


class EmailValidationOutput(BaseModel):
    passed: bool = Field(
        description="True only when every rubric criterion is satisfied."
    )
    feedback: str = Field(
        description=(
            "If passed=True: 'All rubric criteria passed.' "
            "Otherwise: one refinement instruction per failed criterion."
        )
    )


class EmailRubricValidatorBlock(Block[EmailValidationInput, EmailValidationOutput]):
    """Static critic block — evaluates the email against the binary rubric
    and returns structured feedback so the parent LLMAgentBlock can self-correct."""

    llm_model: str = "gemini/gemini-3.1-flash-lite-preview"
    async def run(self, input: EmailValidationInput) -> EmailValidationOutput:
        rubric_lines = "\n".join(
            f"{i + 1}. Criterion: \"{r['criterion']}\""
            f"   If NO → {r['instruction']}"
            for i, r in enumerate(RUBRIC)
        )

        prompt = (
            "You are a strict email quality evaluator. "
            "Evaluate the email below against each rubric criterion. "
            "Answer YES or NO for each and return ONLY a JSON object — no prose.\n\n"
            f"Recipient context: {input.recipient_context}\n\n"
            f"Email:\n---\n{input.email_text}\n---\n\n"
            f"Rubric:\n{rubric_lines}\n\n"
            "Return this exact JSON structure:\n"
            '{\n'
            '  "results": [\n'
            '    {"criterion": "<text>", "passed": true, "instruction": ""},\n'
            '    {"criterion": "<text>", "passed": false, "instruction": "<refinement>"}\n'
            '  ]\n'
            '}'
        )

        response = await litellm.acompletion(
            model=self.llm_model,
            messages=[{"role": "user", "content": prompt}],
        )

        raw = response.choices[0].message.content or ""

        # Extrai o JSON mesmo que o modelo adicione texto extra ao redor
        json_match = re.search(r"\{[\s\S]*\}", raw)
        if not json_match:
            return EmailValidationOutput(
                passed=False,
                feedback=f"Validator could not parse LLM response. Raw output:\n{raw}",
            )

        data = json.loads(json_match.group())
        results: list[dict] = data.get("results", [])

        failed = [r for r in results if not r.get("passed", True)]

        if not failed:
            return EmailValidationOutput(passed=True, feedback="All rubric criteria passed.")

        lines = ["The email failed the following rubric criteria — please fix them:"]
        for r in failed:
            lines.append(f"  • {r['criterion']} → {r['instruction']}")

        return EmailValidationOutput(passed=False, feedback="\n".join(lines))


# ---------------------------------------------------------------------------
# Workflow: um único agente com reflexão interna via tool loop
# ---------------------------------------------------------------------------

async def main() -> None:
    model_name = "gemini/gemini-3.1-flash-lite-preview" #"ollama/gemma3:4b"

    # Ferramenta crítica: avalia o email e devolve feedback ao mesmo agente
    validator = EmailRubricValidatorBlock(
        name="EmailRubricValidator",
        description=(
            "Validates a drafted email against a binary rubric with three criteria: "
            "(1) has greeting and farewell, "
            "(2) tone matches recipient, "
            "(3) main request is explicit. "
            "Returns passed=True when all criteria are met, otherwise returns "
            "specific refinement instructions."
        ),
        llm_model=model_name,
    )

    # Agente único: o while-True interno é o loop de reflexão
    email_agent = LLMAgentBlock(
        name="EmailAgent",
        model=model_name,
        description="Drafts and self-refines emails using internal reflection.",
        system_prompt=(
            "You are an expert professional email writer.\n"
            "Workflow you MUST follow:\n"
            "  1. Draft a complete email based on the user request.\n"
            "  2. Call the EmailRubricValidator tool, passing:\n"
            "       - email_text: the full draft you just wrote\n"
            "       - recipient_context: a short description of who the recipient is and the email purpose\n"
            "  3. If passed=False, revise the email following every instruction in the feedback, "
            "then call the validator again.\n"
            "  4. Repeat steps 2-3 until passed=True.\n"
            "  5. Return ONLY the final approved email text — no commentary."
        ),
        tools=[validator],
        max_iterations=5
    )

    graph = WorkflowGraph()
    graph.add_block(email_agent)
    executor = WorkflowExecutor(graph)

    ctx = await executor.run(
        initial_input={
            "prompt": (
                "Draft a professional email with:\n"
                "  Subject: Meeting Reminder\n"
                "  Body hint: Don't forget about our meeting tomorrow at 10 AM. "
                "Please bring your project updates.\n"
                "  Recipient: John Doe (business client)\n"
                "  Sender: Jane Smith"
            )
        }
    )

    output = ctx.get_output("EmailAgent")

    print("=" * 60)
    print("Final Email (after internal reflection):")
    print("=" * 60)
    print(output.response)
    print(f"\n[Validation tool calls made: {output.tool_calls_made}]")


if __name__ == "__main__":
    asyncio.run(main())
