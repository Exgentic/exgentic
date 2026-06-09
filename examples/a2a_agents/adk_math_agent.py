"""Minimal ADK + A2A proof-of-life agent for Exgentic benchmark testing."""
import os
import asyncio
import logging

os.environ.setdefault("GOOGLE_GENAI_USE_VERTEXAI", "1")
os.environ.setdefault("GOOGLE_CLOUD_PROJECT", "alanblount-demo")
os.environ.setdefault("GOOGLE_CLOUD_LOCATION", "global")

from google.adk.agents import LlmAgent
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.genai import types

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

agent = LlmAgent(
    name="math_agent",
    model="gemini-3.1-flash-lite",
    instruction="""You are a math problem solver. When given a math problem:
1. Think through the solution step by step
2. Calculate the final numerical answer
3. Call the submit tool with your integer answer

If a calculate_expression tool is available, use it for arithmetic.
Always provide your final answer as an integer via the submit tool.""",
)

session_service = InMemorySessionService()
runner = Runner(agent=agent, app_name="math_bench", session_service=session_service)


async def handle_message(message: str, session_id: str = None) -> str:
    """Handle a single message and return the agent's response."""
    if session_id is None:
        session = await session_service.create_session(
            app_name="math_bench", user_id="bench"
        )
        session_id = session.id

    response_text = ""
    async for event in runner.run_async(
        user_id="bench",
        session_id=session_id,
        new_message=types.Content(
            parts=[types.Part(text=message)], role="user"
        ),
    ):
        if event.is_final_response() and event.content and event.content.parts:
            response_text = event.content.parts[0].text
            break

    return response_text


if __name__ == "__main__":
    # Quick test
    result = asyncio.run(handle_message("What is 15 * 23? Give me just the number."))
    print(f"Agent response: {result}")
