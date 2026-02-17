# server.py
from flask import Flask, request
from flask_socketio import SocketIO, emit

app = Flask(__name__)
socketio = SocketIO(app, cors_allowed_origins="*", async_mode="eventlet")

# Optional: simple health check
@app.get("/")
def ok():
    return {"ok": True}

# Producer (your Python estimator) sends "state" -> server re-broadcasts to everyone
@socketio.on("state")
def on_state(data):
    socketio.emit("state", data)  # <-- no broadcast kwarg

@socketio.on("connect")
def on_connect():
    print("Client connected:", request.sid)

@socketio.on("disconnect")
def on_disconnect():
    print("Client disconnected:", request.sid)

if __name__ == "__main__":
    print("✅ Socket.IO bridge running on http://127.0.0.1:5000")
    socketio.run(app, host="127.0.0.1", port=5000)
