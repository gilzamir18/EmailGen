import asyncio
from pydantic import BaseModel, Field

from agenticblocks.core.block import Block
from agenticblocks.core.graph import WorkflowGraph
from agenticblocks.runtime.executor import WorkflowExecutor
from agenticblocks.blocks.llm.agent import LLMAgentBlock
import os

async def main():
    # Check if the API key is set in the environment
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise ValueError("API key not found in environment variables. "/ 
                         "Please set GEMINI_API_KEY.")

    # Create the workflow graph
    graph = WorkflowGraph()

    draft_email_agent_block = LLMAgentBlock(
        name="DraftEmailAgent",
        model="gemini/gemini-3.1-flash-lite-preview",
        description="An agent that drafts an email based on the provided input.",
        system_prompt="You are an assistant that drafts emails based on the provided subject, body, recipient name, and sender name.",
    )

    # Add the blocks to the graph
    drafter = graph.add_block(draft_email_agent_block)

    # Create the workflow executor
    executor = WorkflowExecutor(graph)

    try:
        # Run the workflow and get the output
        ctx = await executor.run(initial_input={"prompt": """Draft an email with the subject
                                                   'Meeting Reminder', body 'Don't forget about
                                                   our meeting tomorrow at 10 AM.', recipient name
                                                    'John Doe', and sender name 'Jane Smith'."""})
        output = ctx.get_output("DraftEmailAgent")
        print("Generated Email Content:")
        print(output.response)
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    asyncio.run(main())
