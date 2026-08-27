"""
Endpoint /ws - WebSocket untuk broadcasting status progress.
"""
from __future__ import annotations

import asyncio
import json
import logging
from typing import Dict, Set

from fastapi import APIRouter, WebSocket, WebSocketDisconnect

logger = logging.getLogger("asmeranda.api.ws")
router = APIRouter()

class ConnectionManager:
    """Mendukung multi-channel WebSocket: dataset_id, job_id, atau state_id."""
    def __init__(self):
        # Menyimpan active connections: channel_id -> set of WebSockets
        self.active_connections: Dict[str, Set[WebSocket]] = {}

    async def connect(self, websocket: WebSocket, channel_id: str):
        await websocket.accept()
        if channel_id not in self.active_connections:
            self.active_connections[channel_id] = set()
        self.active_connections[channel_id].add(websocket)
        logger.info(f"WebSocket connected for channel {channel_id}")

    def disconnect(self, websocket: WebSocket, channel_id: str):
        if channel_id in self.active_connections:
            self.active_connections[channel_id].discard(websocket)
            if not self.active_connections[channel_id]:
                del self.active_connections[channel_id]
        logger.info(f"WebSocket disconnected for channel {channel_id}")

    async def broadcast(self, channel_id: str, message: dict):
        if channel_id in self.active_connections:
            websockets = list(self.active_connections[channel_id])
            for connection in websockets:
                try:
                    await connection.send_text(json.dumps(message))
                except Exception as exc:
                    logger.error(f"Error sending message to websocket: {exc}")
                    self.disconnect(connection, channel_id)

manager = ConnectionManager()

@router.websocket("/{channel_id}")
async def websocket_endpoint(websocket: WebSocket, channel_id: str):
    await manager.connect(websocket, channel_id)
    try:
        while True:
            # Tetap terbuka dan mendengarkan ping/pesan dari client (jika ada)
            data = await websocket.receive_text()
            if data == "ping":
                await websocket.send_text(json.dumps({"type": "pong"}))
    except WebSocketDisconnect:
        manager.disconnect(websocket, channel_id)
    except Exception as exc:
        logger.error(f"WebSocket error for channel {channel_id}: {exc}")
        manager.disconnect(websocket, channel_id)

