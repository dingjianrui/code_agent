# Code Agent - AI Assistant with SSE Streaming

This project implements an AI code assistant with real-time streaming responses using Server-Sent Events (SSE). It consists of a FastAPI backend server and a simple HTML/JavaScript frontend.

## Project Structure

- `sse_server.py` - Backend FastAPI server that handles agent communication
- `static/index.html` - Frontend interface for interacting with the code agent

## Backend (sse_server.py)

The backend is built with FastAPI and provides:

- SSE streaming endpoint (`/sse`) for real-time agent responses
- Integration with LangGraph and CodeAct for code execution capabilities
- Remote code evaluation through a sandbox environment
- CORS support for cross-origin requests
- Static file serving for the frontend

Key components:
- LangChain OpenAI integration for LLM capabilities
- LangGraph for agent state management
- Async streaming response handling
- Server-Sent Events (SSE) for real-time communication

## Frontend (index.html)

The frontend provides a simple chat interface with:

- Real-time message streaming using the EventSource API
- Markdown rendering for code blocks and formatted text
- Clean, responsive UI for chat interactions
- Error handling for connection issues

## Setup and Usage

### Prerequisites

- Python 3.8+
- Required environment variables:
  - `ARK_API_BASE` - Base URL for the LLM API
  - `ARK_API_KEY` - API key for authentication
  - `SANDBOX_URL` - URL for the code execution sandbox
  - `AUTH_KEY` - Authentication key for the sandbox

### Installation

1. Clone the repository
2. Install dependencies:
   ```
   pip install fastapi uvicorn langchain-openai langgraph-codeact
   ```

### Running the Application

1. Start the server:
   ```
   python sse_server.py
   ```
2. Open a browser and navigate to `http://localhost:8000`
3. Start chatting with the code agent!

## How It Works

1. User sends a message through the frontend
2. The message is sent to the backend via SSE connection
3. The backend processes the message using the LangGraph agent
4. Responses are streamed back to the frontend in real-time
5. The frontend renders the responses with Markdown formatting

## Features

- Real-time streaming responses
- Code execution capabilities
- Markdown rendering for code blocks
- Simple and intuitive UI