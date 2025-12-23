"""
Simple script to run the backend server
"""
import uvicorn

if __name__ == "__main__":
    print("=" * 60)
    print("Starting RL Platform Backend Server")
    print("=" * 60)
    print("Server will run on http://localhost:8000")
    print("WebSocket endpoint: ws://localhost:8000/ws")
    print("Press Ctrl+C to stop")
    print("=" * 60)
    
    uvicorn.run(
        "backend.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )
