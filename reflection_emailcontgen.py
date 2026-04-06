import asyncio
from pydantic import BaseModel, Field
from agenticblocks.core.block import Block
from agenticblocks.core.graph import WorkflowGraph
from agenticblocks.runtime.executor import WorkflowExecutor
from agenticblocks.blocks.llm.agent import LLMAgentBlock, AgentInput
import os

class AgentOutputToInput(BaseModel):
    response: str

class AgentBridgeBlock(Block[AgentOutputToInput, AgentInput]):
    def __init__(self):
        super().__init__(name="AgentBridge", description="Maps agent response to next agent prompt.")

    async def run(self, input: AgentOutputToInput) -> AgentInput:
        return AgentInput(prompt=input.response)

async def main():
    model_name = "gemini/gemini-3.1-flash-preview" #"ollama/gemma3:4b"

    if model_name.startswith("gemini/"):
        # Check if the API key is set in the environment
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise ValueError("API key not found in environment variables. "/ 
                             "Please set GEMINI_API_KEY.")

    # Create the workflow graph
    graph = WorkflowGraph()

    draft_email_agent_block = LLMAgentBlock(
        name="DraftEmailAgent",
        model=f"{model_name}",
        description="An agent that drafts an email based on the provided input.", 
        system_prompt="You are an assistant that drafts emails based on the provided" \
        "subject, body, recipient name, and sender name.",
    )

    revisor_agent_block = LLMAgentBlock(
        name="RevisorAgent",
        model=f"{model_name}",
        description="An agent that revises the drafted email",
        system_prompt="You are an assistant that revises emails for clarity, tone,"\
            " and correctness, removing repeated"\
              " words and fixing typos. Also, complete the email if it is not complete. " \
              "Return only the email content without any additional text.",
        max_iterations=5
    )

    bridge_block = AgentBridgeBlock()

    # Add the blocks to the graph
    drafter = graph.add_block(draft_email_agent_block)
    bridge = graph.add_block(bridge_block)
    revisor = graph.add_block(revisor_agent_block)
    graph.connect(drafter, bridge)
    graph.connect(bridge, revisor)

    # Create the workflow executor
    executor = WorkflowExecutor(graph)

    try:
        # Run the workflow and get the output
        ctx = await executor.run(initial_input={"prompt": """Draft an email with the subject
                                                   'Meeting Reminder', body 'Dot forget about
                                                   our meeting tomorrow at 10 AM. Please, ....', recipient name
                                                    'John Doe', and sender name 'Jane Smith'."""})
        output = ctx.get_output("RevisorAgent")
        print("Generated Email Content:")
        print(output.response)
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    asyncio.run(main())