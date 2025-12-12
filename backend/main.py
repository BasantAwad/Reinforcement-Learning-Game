from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
import asyncio
import json
from .training import TrainingManager

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

training_manager = TrainingManager()

@app.get("/")
def read_root():
    return {"message": "RL Platform Backend Ready - FIXED VERSION"}

@app.on_event("startup")
async def startup_event():
    print(">>> BACKEND STARTED: FIXED VERSION - READY FOR CONNECTIONS <<<")

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    print("WebSocket accepted")
    try:
        while True:
            data = await websocket.receive_text()
            print(f"Received: {data}")
            message = json.loads(data)
            
            if message["type"] == "start":
                game_id = message["gameId"]
                algo_id = message["algoId"]
                print(f"Starting training: {game_id} with {algo_id}")
                asyncio.create_task(training_manager.start_training(game_id, algo_id, websocket))
                
            elif message["type"] == "stop":
                print("Stopping training")
                training_manager.stop()
                await websocket.send_json({"type": "info", "message": "Training stopped"})
                
    except WebSocketDisconnect:
        training_manager.stop()
        print("Client disconnected")
    except Exception as e:
        training_manager.stop()
        print(f"Error in Websocket: {e}")
        import traceback
        traceback.print_exc()
