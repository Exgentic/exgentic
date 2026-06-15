import os, logging
os.environ.setdefault("GOOGLE_GENAI_USE_VERTEXAI", "1")
os.environ.setdefault("GOOGLE_CLOUD_PROJECT", "alanblount-demo")
os.environ.setdefault("GOOGLE_CLOUD_LOCATION", "global")

from google import genai
from a2a.server.agent_execution.agent_executor import AgentExecutor
from a2a.server.request_handlers.default_request_handler_v2 import DefaultRequestHandlerV2
from a2a.server.tasks.inmemory_task_store import InMemoryTaskStore
from a2a.server.events.in_memory_queue_manager import InMemoryQueueManager
from a2a.server.routes.agent_card_routes import create_agent_card_routes
from a2a.server.routes.jsonrpc_routes import create_jsonrpc_routes
from a2a.server.routes.fastapi_routes import add_a2a_routes_to_fastapi
from a2a.types import AgentCard, AgentCapabilities, AgentSkill, AgentInterface, Message, Part, Role
from fastapi import FastAPI
import uvicorn

logging.basicConfig(level=logging.INFO)
gclient = genai.Client(vertexai=True, project="alanblount-demo", location="global")

class MathExecutor(AgentExecutor):
    async def execute(self, context, event_queue):
        user_msg = ""
        if context.message and context.message.parts:
            for part in context.message.parts:
                if part.text:
                    user_msg += part.text
        response = gclient.models.generate_content(
            model="gemini-3.1-flash-lite",
            contents=[{"role": "user", "parts": [{"text": user_msg}]}]
        )
        reply = ""
        for p in response.candidates[0].content.parts:
            if hasattr(p, 'text') and p.text:
                reply += p.text
        msg = Message(role=Role.ROLE_AGENT, parts=[Part(text=reply)], message_id=str(hash(reply)))
        await event_queue.enqueue_event(msg)
    
    async def cancel(self, context, event_queue):
        pass

agent_card = AgentCard(
    name="Math Agent",
    description="Math problem solver using Gemini 3.1 Flash Lite",
    version="1.0.0",
    capabilities=AgentCapabilities(streaming=False, push_notifications=False),
    skills=[AgentSkill(id="math", name="Math", description="Solve math problems")],
    default_input_modes=["text/plain"],
    default_output_modes=["text/plain"],
    supported_interfaces=[
        AgentInterface(url="http://localhost:8765/jsonrpc", protocol_binding="jsonrpc/http"),
    ],
)

task_store = InMemoryTaskStore()
queue_manager = InMemoryQueueManager()
handler = DefaultRequestHandlerV2(
    agent_executor=MathExecutor(),
    task_store=task_store,
    queue_manager=queue_manager,
    agent_card=agent_card,
)

app = FastAPI()
card_routes = create_agent_card_routes(agent_card)
jsonrpc = create_jsonrpc_routes(handler, rpc_url="/jsonrpc", enable_v0_3_compat=True)
add_a2a_routes_to_fastapi(app, agent_card_routes=card_routes, jsonrpc_routes=jsonrpc)

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8765, log_level="info")
