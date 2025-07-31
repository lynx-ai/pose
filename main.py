import io, time, base64, json
import torch
import asyncio
import uvicorn
import requests
import argparse
from PIL import Image
from pydantic import BaseModel
from typing import Dict, Optional, AsyncGenerator
from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.responses import StreamingResponse, Response
from fastapi.middleware.cors import CORSMiddleware
from sp_logging import setup_logging, RequestIdMiddleware

from controlnet_aux import DWposeDetector

logger = setup_logging()
PORT = 8293

app = FastAPI()
app.add_middleware(RequestIdMiddleware)

dwpose = DWposeDetector("cuda:0" if torch.cuda.is_available() else "cpu")

class PoseParams(BaseModel):
    image: str

class PoseResponse(BaseModel):
    image: str

@app.websocket("/pose")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    logger.info("WebSocket connection established")

    try:
        while True:
            data = await websocket.receive_text()
            message = json.loads(data)

            image_data = message["image"].split(",")[-1]
            image_bytes = base64.b64decode(image_data)
            image = Image.open(io.BytesIO(image_bytes))
            image = image.transpose(Image.FLIP_LEFT_RIGHT)

            start_time = time.time()
            result = dwpose(image, output_type="pil", include_hands=True, include_face=True)
            end_time = time.time()

            execution_time = end_time - start_time
            print(f"dwpose took {execution_time:.4f} seconds")

            buffered = io.BytesIO()
            result.save(buffered, format="JPEG", quality=85)
            result_image = base64.b64encode(buffered.getvalue()).decode()

            response = {"image": result_image}
            await websocket.send_text(json.dumps(response))

    except WebSocketDisconnect:
        logger.info("WebSocket client disconnected")
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
        await websocket.close(code=1011, reason="Internal server error")

@app.get("/health", status_code=200)
async def index():
    return "ok"

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--origin", default="http://localhost:5173", help="Allowed CORS origin")
    args = parser.parse_args()

    app.add_middleware(
        CORSMiddleware,
        allow_origins=[args.origin],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    logger.info(f"Starting server on port {PORT} with allowed origin: {args.origin}")
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=PORT,
        log_level="error",
        log_config=None
    )
