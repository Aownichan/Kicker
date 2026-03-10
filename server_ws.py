import asyncio
import json
import websockets

CLIENTS = set()

async def handler(ws):
    CLIENTS.add(ws)
    print("✅ Unity WS connected")
    try:
        async for msg in ws:
            # Unity doesn't need to send anything; ignore messages
            pass
    except Exception:
        pass
    finally:
        CLIENTS.discard(ws)
        print("⚠️ Unity WS disconnected")

async def broadcast_state(queue: asyncio.Queue):
    while True:
        state = await queue.get()
        if not CLIENTS:
            continue
        payload = json.dumps(state)
        dead = []
        for ws in CLIENTS:
            try:
                await ws.send(payload)
            except Exception:
                dead.append(ws)
        for ws in dead:
            CLIENTS.discard(ws)

async def producer_listener(queue: asyncio.Queue):
    """
    Producer sends JSON state via a websocket connection to ws://127.0.0.1:8765/producer
    """
    async def prod_handler(ws):
        print("✅ Producer connected")
        try:
            async for msg in ws:
                try:
                    state = json.loads(msg)
                    await queue.put(state)
                except Exception:
                    continue
        finally:
            print("⚠️ Producer disconnected")

    return await websockets.serve(prod_handler, "127.0.0.1", 8765, ping_interval=None)

async def unity_listener(queue: asyncio.Queue):
    return await websockets.serve(handler, "0.0.0.0", 8766, ping_interval=None)

async def main():
    queue = asyncio.Queue()
    prod_server = await producer_listener(queue)
    unity_server = await unity_listener(queue)
    print("✅ WS producer: ws://127.0.0.1:8765")
    print("✅ WS unity:    ws://127.0.0.1:8766")
    await broadcast_state(queue)

if __name__ == "__main__":
    asyncio.run(main())
