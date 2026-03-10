import eventlet
eventlet.monkey_patch()

from flask import Flask, request
from flask_socketio import SocketIO

app = Flask(__name__)

socketio = SocketIO(
    app,
    cors_allowed_origins="*",
    async_mode="eventlet",
    logger=True,
    engineio_logger=True,
)

@socketio.on("connect")
def on_connect():
    # This prints for BOTH Unity and Python producer
    print(f"✅ CONNECT sid={request.sid} ip={request.remote_addr} ua={request.headers.get('User-Agent')}")

@socketio.on("disconnect")
def on_disconnect():
    print(f"⚠️ DISCONNECT sid={request.sid}")

@socketio.on("state")
def on_state(data):
    socketio.emit("state", data)

if __name__ == "__main__":
    print("✅ Socket.IO bridge running on http://0.0.0.0:5000")
    socketio.run(app, host="0.0.0.0", port=5000)
