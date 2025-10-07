import os
import json
import requests
import asyncio
from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from typing import AsyncGenerator, Any

from langgraph.checkpoint.memory import MemorySaver
from langchain_openai import ChatOpenAI
from langgraph_codeact import create_codeact


# Initialize FastAPI app
app = FastAPI(title="Agent SSE Server")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allow all methods
    allow_headers=["*"],  # Allow all headers
)

# Initialize the agent
llm_config = {
    "model": "doubao-seed-1-6-250615",
    "base_url": os.environ["ARK_API_BASE"],
    "api_key": os.environ["ARK_API_KEY"],
    "temperature": 0.2,
    "extra_body": {"thinking": {"type":"disabled"}}
}
llm = ChatOpenAI(**llm_config)

def remote_eval(code: str, language: str = "python") -> str:
    if language == "python":
        url = os.environ["SANDBOX_URL"] + "/run_code"
        body = {
            "code": code,
            "language": language,
        }
        response = requests.post(url, json=body, headers={"Authorization": f"Bearer {os.environ['AUTH_KEY']}"})
        response.raise_for_status()
        json = response.json()
        print("code result -->", json["run_result"]["stdout"])
        return json["run_result"]["stdout"]
    else:
        raise ValueError(f"Unsupported language: {language}")

code_act = create_codeact(llm, remote_eval)
    
@app.get("/sse")
async def chat_sse_endpoint(request: Request) -> StreamingResponse:
    """
    Endpoint for streaming agent responses using Server-Sent Events (SSE) via GET request.
    
    The client connects using EventSource API with a query parameter 'message',
    and the server streams back the agent's response as SSE events.
    """
    # Parse the query parameter
    user_message = request.query_params.get("message", "")
    
    # Create the messages list for the agent
    messages = [
        {
            "role": "user",
            "content": user_message
        }
    ]

    agent = code_act.compile(checkpointer=MemorySaver())
    
    return StreamingResponse(
        stream_response(messages, agent),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",  # Disable Nginx buffering if you're using Nginx
        }
    )


# Mount static files directory
app.mount("/", StaticFiles(directory="static", html=True), name="static")

async def stream_response(messages: list, agent: Any) -> AsyncGenerator[str, None]:
    """
    Stream the agent's response as SSE events.
    
    Args:
        messages: List of messages to send to the agent
        
    Yields:
        SSE formatted events with the agent's response
    """
    # Send initial event to establish connection
    # yield "data: {\"status\": \"started\"}\n\n"
    
    # Stream the agent's response
    async for typ, chunk in agent.astream(
        {"messages": messages},
        stream_mode=["values", "messages"],
        config={"configurable": {"thread_id": 1}},
    ):
        if typ == "messages":
            # Stream the message content
            data = {
                "type": "event",
                "content": chunk[0].content,
            }
            yield f"data: {json.dumps(data)}\n\n"
            await asyncio.sleep(0)

    yield "data: {\"status\": \"completed\"}\n\n"
    await asyncio.sleep(0)
        
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)