from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
import asyncio
import json
import os
from .training import TrainingManager
from .metrics import metrics_tracker
from .report_generator import generate_markdown_report

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

@app.get("/api/metrics")
def get_metrics():
    """Get all tracked metrics."""
    return metrics_tracker.get_comparison()

@app.get("/api/metrics/export")
def export_metrics():
    """Export metrics to JSON file."""
    filepath = "metrics_export.json"
    metrics_tracker.export_to_json(filepath)
    return {"message": f"Metrics exported to {filepath}"}

@app.get("/api/report/generate")
def generate_report():
    """Generate markdown report for analysis."""
    comparison = metrics_tracker.get_comparison()
    filepath = generate_markdown_report(comparison)
    return {"message": f"Report generated: {filepath}", "filepath": filepath}

@app.get("/api/report/download")
def download_report():
    """Download the generated markdown report."""
    filepath = "RL_ANALYSIS_REPORT.md"
    if os.path.exists(filepath):
        return FileResponse(filepath, filename=filepath, media_type='text/markdown')
    return {"error": "Report not found. Generate it first using /api/report/generate"}

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
                env_mode = message.get("envMode", "static")  # Default to static if not specified
                print(f"Starting training: {game_id} with {algo_id} (mode: {env_mode})")
                asyncio.create_task(training_manager.start_training(game_id, algo_id, websocket, env_mode))
                
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
