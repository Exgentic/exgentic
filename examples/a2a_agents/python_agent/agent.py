#!/usr/bin/env python3
"""Minimal A2A agent for proof-of-life testing with Exgentic.

Responds to questions (especially math) using Gemini Flash Lite via Vertex AI,
or falls back to a simple echo/mock if no API is configured.

Usage:
    python agent.py [--port 9100]
"""

from __future__ import annotations

import argparse
import logging
import os
import re
import subprocess

import uvicorn
from a2a.server.agent_execution import AgentExecutor
from a2a.server.agent_execution.context import RequestContext
from a2a.server.apps.jsonrpc.starlette_app import A2AStarletteApplication
from a2a.server.events import EventQueue, InMemoryQueueManager
from a2a.server.request_handlers import DefaultRequestHandler
from a2a.server.tasks import InMemoryTaskStore, TaskUpdater
from a2a.types import (
    AgentCapabilities,
    AgentCard,
    AgentSkill,
    Part,
    TaskState,
    TextPart,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def _get_gemini_response(prompt: str) -> str | None:
    """Call Gemini Flash Lite via google-genai SDK (supports Vertex AI)."""
    try:
        from google import genai

        # Try to get API key from env, otherwise use Vertex AI via gcloud
        api_key = os.environ.get("GEMINI_API_KEY") or os.environ.get(
            "GOOGLE_API_KEY"
        )
        if api_key:
            client = genai.Client(api_key=api_key)
        else:
            # Use Vertex AI with gcloud credentials
            project = os.environ.get("GOOGLE_CLOUD_PROJECT") or os.environ.get(
                "GCLOUD_PROJECT"
            )
            if not project:
                # Try to get project from gcloud config
                try:
                    result = subprocess.run(
                        ["gcloud", "config", "get-value", "project"],
                        capture_output=True,
                        text=True,
                        timeout=5,
                    )
                    project = result.stdout.strip()
                except Exception:
                    pass
            if project:
                client = genai.Client(
                    vertexai=True,
                    project=project,
                    location="us-central1",
                )
            else:
                logger.warning("No Gemini API key or GCP project found")
                return None

        # Try multiple model names; Vertex AI needs versioned names
        for model in [
            "gemini-2.5-flash-preview-05-20",
            "gemini-2.0-flash-001",
            "gemini-2.0-flash-lite-001",
            "gemini-1.5-flash-002",
        ]:
            try:
                response = client.models.generate_content(
                    model=model, contents=prompt
                )
                return response.text
            except Exception as e:
                logger.warning("Model %s failed: %s", model, e)
                continue
        return None
    except ImportError:
        logger.warning("google-genai not installed")
        return None
    except Exception as e:
        logger.error("Gemini call failed: %s", e)
        return None


def _mock_math_response(text: str) -> str:
    """Simple mock for basic math when no LLM is available."""
    # Try to evaluate simple arithmetic expressions
    # Extract numbers and operations from text
    numbers = re.findall(r"[-+]?\d*\.?\d+", text)
    if len(numbers) >= 2:
        try:
            # Simple heuristic: if question asks for sum/add/total/plus
            text_lower = text.lower()
            nums = [float(n) for n in numbers]
            if any(w in text_lower for w in ["sum", "add", "plus", "total", "+"]):
                return str(sum(nums))
            if any(
                w in text_lower for w in ["subtract", "minus", "difference", "-"]
            ):
                return str(nums[0] - sum(nums[1:]))
            if any(w in text_lower for w in ["multiply", "product", "times", "*"]):
                result = 1
                for n in nums:
                    result *= n
                return str(result)
            if any(w in text_lower for w in ["divide", "quotient", "/"]):
                if nums[1] != 0:
                    return str(nums[0] / nums[1])
            # Default: try to interpret as a math question, just sum
            return str(sum(nums))
        except Exception:
            pass
    return f"I received your question: {text[:200]}"


class MathAgentExecutor(AgentExecutor):
    """Simple agent that answers math questions."""

    async def execute(
        self,
        context: RequestContext,
        event_queue: EventQueue,
    ) -> None:
        updater = TaskUpdater(
            event_queue,
            context.task_id,
            context.context_id,
        )

        # Signal that we're working
        await updater.start_work()

        # Get user message text
        user_text = ""
        if context.message and context.message.parts:
            for part in context.message.parts:
                if isinstance(part.root, TextPart):
                    user_text += part.root.text

        if not user_text:
            await updater.failed(
                updater.new_agent_message(
                    [Part(root=TextPart(text="No input text received"))]
                )
            )
            return

        logger.info("Processing: %s", user_text[:200])

        # Try Gemini first, fall back to mock
        response = _get_gemini_response(user_text)
        if response is None:
            logger.info("Gemini unavailable, using mock math response")
            response = _mock_math_response(user_text)

        logger.info("Response: %s", response[:200])

        # Complete with response
        await updater.complete(
            updater.new_agent_message(
                [Part(root=TextPart(text=response))]
            )
        )

    async def cancel(
        self,
        context: RequestContext,
        event_queue: EventQueue,
    ) -> None:
        updater = TaskUpdater(
            event_queue,
            context.task_id,
            context.context_id,
        )
        await updater.cancel()


def build_agent_card(host: str, port: int) -> AgentCard:
    """Build the agent card advertising this agent's capabilities."""
    return AgentCard(
        name="Math Agent",
        description="A simple agent that answers math questions",
        url=f"http://{host}:{port}",
        version="0.1.0",
        capabilities=AgentCapabilities(streaming=True),
        skills=[
            AgentSkill(
                id="math",
                name="Math Problem Solver",
                description="Solves math problems and answers questions",
                tags=["math", "calculation", "reasoning"],
            )
        ],
        default_input_modes=["text/plain"],
        default_output_modes=["text/plain"],
    )


def main():
    parser = argparse.ArgumentParser(description="A2A Math Agent")
    parser.add_argument("--host", default="0.0.0.0", help="Host to bind to")
    parser.add_argument(
        "--port", type=int, default=9100, help="Port to listen on"
    )
    args = parser.parse_args()

    agent_card = build_agent_card(args.host, args.port)
    executor = MathAgentExecutor()
    task_store = InMemoryTaskStore()
    queue_manager = InMemoryQueueManager()

    handler = DefaultRequestHandler(
        agent_executor=executor,
        task_store=task_store,
        queue_manager=queue_manager,
    )

    a2a_app = A2AStarletteApplication(
        agent_card=agent_card,
        http_handler=handler,
    )

    app = a2a_app.build()
    logger.info("Starting A2A Math Agent on %s:%d", args.host, args.port)
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
