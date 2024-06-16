import os
import asyncio
from concurrent.futures import ThreadPoolExecutor
from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from contextlib import asynccontextmanager
import uvicorn
from ai_functions.open_ai_functions import invoke_rag_chain
from db import insert_message, get_recent_messages, insert_history, update_history_name, get_history_list, get_chat_history, check_and_insert_session

# Configuration
DEBUG_MODE = True  # Change this to False for production

# Set the USER_AGENT environment variable
os.environ['USER_AGENT'] = 'MyApp/1.0.0'  # Replace with your user agent

# Thread pool executor with a limited number of workers
MAX_WORKERS = 3
executor = ThreadPoolExecutor(max_workers=MAX_WORKERS)

# Queue to handle incoming requests
request_queue = asyncio.Queue()

from datetime import datetime

def log_error(error_id, function, reason):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with open("errorfile.txt", "a") as f:
        f.write(f"Timestamp: {timestamp}, Error ID: {error_id}, Function: {function}, Reason: {reason}\n")

# async def handle_request(session_id: str, user_input: str):
#     try:
#         print("Handling request for session:", session_id)
#         response = await invoke_rag_chain(user_input, session_id)
#         print("AI response received")
        
#         return response
#     except Exception as e:
#         error_id = "ERR001"
#         log_error(error_id, "handle_request", str(e))
#         print(f"Error {error_id} occurred in handle_request: {str(e)}")
#         return None

# Function to handle a request and generate the streaming response
async def handle_request(session_id, query, history_id):
    try:
        print("Handling request for session:", session_id)
        loop = asyncio.get_event_loop()
        response = await loop.run_in_executor(None, invoke_rag_chain, query, history_id)
        print("AI response received")
        return response
    except Exception as e:
        error_id = "ERR001"
        log_error(error_id, "handle_request", str(e))
        print(f"Error {error_id} occurred in handle_request: {str(e)}")
        return None

# Define the lifespan context manager
@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        print("Starting up the application")
        asyncio.create_task(process_queue())
        yield
        print("Shutting down the application")
        executor.shutdown(wait=False)
    except Exception as e:
        error_id = "ERR002"
        log_error(error_id, "lifespan", str(e))
        print(f"Error {error_id} occurred in lifespan: {str(e)}")

app = FastAPI(lifespan=lifespan)

# Request model for validation
class QueryRequest(BaseModel):
    query: str
    session_id: str
    history_id: str

class ChangeHistoryNameRequest(BaseModel):
    session_id: str
    history_id: str
    new_name: str

class ViewHistoryRequest(BaseModel):
    session_id: str

class ViewChatHistoryRequest(BaseModel):
    history_id: str

@app.post("/stream")
async def stream_response(request_body: QueryRequest):
    try:
        print(f"Received query: {request_body.query} with session_id: {request_body.session_id} and history_id: {request_body.history_id}")

        if not request_body.query or not request_body.session_id or not request_body.history_id:
            error_message = "All 'query', 'session_id', and 'history_id' must be provided"
            print(error_message)
            return JSONResponse(content={"error": error_message}, status_code=400)

        check_and_insert_session(request_body.session_id, request_body.history_id)

        # insert_message(request_body.history_id, 'User', request_body.query)
        response = await handle_request(request_body.session_id, request_body.query, request_body.history_id)
        if response is None:
            error_message = "Error processing the request"
            print(error_message)
            return JSONResponse(content={"error": error_message}, status_code=500)

        # insert_message(request_body.history_id, 'AI', response)
        return JSONResponse(content={"response": response}, status_code=200)
    except HTTPException as http_err:
        error_id = "ERR010"
        error_message = f"HTTPException {error_id}: {str(http_err)}"
        print(error_message)
        return JSONResponse(content={"error": error_message}, status_code=http_err.status_code)
    except Exception as e:
        error_id = "ERR010"
        error_message = f"Error {error_id}: {str(e)}"
        print(error_message)
        return JSONResponse(content={"error": error_message}, status_code=500)

@app.post("/change_history_name")
async def change_history_name(request_body: ChangeHistoryNameRequest):
    try:
        update_history_name(request_body.session_id, request_body.history_id, request_body.new_name)
        return JSONResponse(content={"message": f"History name updated successfully for history_id {request_body.history_id}"}, status_code=200)
    except Exception as e:
        error_id = "ERR011"
        error_message = f"Error {error_id}: {str(e)}"
        print(error_message)
        return JSONResponse(content={"error": error_message}, status_code=500)

@app.post("/view_history_id_list")
async def view_history_id_list(request_body: ViewHistoryRequest):
    try:
        history_list = get_history_list(request_body.session_id)
        return JSONResponse(content={"history_list": history_list}, status_code=200)
    except Exception as e:
        error_id = "ERR012"
        error_message = f"Error {error_id}: {str(e)}"
        print(error_message)
        return JSONResponse(content={"error": error_message}, status_code=500)

@app.post("/view_chat_history")
async def view_chat_history(request_body: ViewChatHistoryRequest):
    try:
        chat_history = get_chat_history(request_body.history_id)
        return JSONResponse(content={"chat_history": chat_history}, status_code=200)
    except Exception as e:
        error_id = "ERR013"
        error_message = f"Error {error_id}: {str(e)}"
        print(error_message)
        return JSONResponse(content={"error": error_message}, status_code=500)

# Background task to process the queue
async def process_queue():
    try:
        while True:
            if not request_queue.empty():
                request = await request_queue.get()
                # Process the request here if needed
                await request_queue.put(request)
                print("Processed request from queue")
            await asyncio.sleep(0.1)
    except Exception as e:
        error_id = "ERR005"
        log_error(error_id, "process_queue", str(e))
        print(f"Error {error_id} occurred in process_queue: {str(e)}")

# Run the application
if __name__ == "__main__":
    try:
        if DEBUG_MODE:
            # For development
            print("Running in development mode")
            uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)
        else:
            # For production
            print("Running in production mode")
            uvicorn.run(app, host="0.0.0.0", port=8000)
    except Exception as e:
        error_id = "ERR006"
        log_error(error_id, "__main__", str(e))
        print(f"Error {error_id} occurred in __main__: {str(e)}")