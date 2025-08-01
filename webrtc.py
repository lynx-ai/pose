import asyncio
import json
import logging
import uuid
import time
import base64
import io
import re
import os
from typing import Dict, Set, Optional
from dataclasses import dataclass
from PIL import Image

import torch
from aiohttp import web, WSMsgType
from aiortc import RTCPeerConnection, RTCSessionDescription, RTCDataChannel

from controlnet_aux import DWposeDetector
from sp_logging import setup_logging

logger = setup_logging()

# Global pose detector
dwpose = DWposeDetector("cuda:0" if torch.cuda.is_available() else "cpu")

@dataclass
class PeerInfo:
    id: str
    handle: str
    peer_connection: RTCPeerConnection
    data_channel: Optional[RTCDataChannel] = None

class WebRTCPoseServer:
    def __init__(self, password: str, cors_origin: str = "*"):
        self.room_password = password
        self.cors_origin = cors_origin.rstrip('/') if cors_origin != "*" else cors_origin
        self.signaling_connections = {}  # peer_id -> websocket for status updates
        self.pcs = set()  # Store peer connections
        self.peers = {}  # peer_id -> PeerInfo mapping
        
    def _validate_handle(self, handle: str) -> bool:
        """Validate handle contains only letters, numbers, and underscores"""
        return bool(re.match(r'^[a-zA-Z0-9_]+$', handle))
        
    async def handle_websocket(self, request):
        """Handle WebSocket connections for status updates only"""
        ws = web.WebSocketResponse()
        await ws.prepare(request)
        
        peer_id = str(uuid.uuid4())
        self.signaling_connections[peer_id] = ws
        logger.info(f"Status WebSocket connected: {peer_id}")
        
        # Send current peer count
        await ws.send_str(json.dumps({
            'type': 'peer_count',
            'count': len(self.peers),
            'peers': [{'id': pid, 'handle': info.handle} for pid, info in self.peers.items()]
        }))
        
        try:
            async for msg in ws:
                if msg.type == WSMsgType.TEXT:
                    # Handle any status messages if needed
                    pass
                elif msg.type == WSMsgType.ERROR:
                    logger.error(f'Status WebSocket error from {peer_id}: {ws.exception()}')
        except Exception as e:
            logger.error(f"Error handling status WebSocket {peer_id}: {e}")
        finally:
            self.signaling_connections.pop(peer_id, None)
            logger.info(f"Status WebSocket disconnected: {peer_id}")
            
        return ws
    
    async def handle_webrtc_offer(self, request):
        """Handle WebRTC offer (like aiortc examples)"""
        logger.info(f"Received WebRTC offer from {request.remote}")
        params = await request.json()
        
        # Validate password
        password = params.get('password')
        if password != self.room_password:
            return web.Response(
                status=403,
                content_type="application/json",
                text=json.dumps({'error': 'Invalid password'})
            )
        
        handle = params.get('handle', f'User{str(uuid.uuid4())[:6]}')
        
        # Validate handle
        if not self._validate_handle(handle):
            return web.Response(
                status=400,
                content_type="application/json",
                text=json.dumps({'error': 'Handle must contain only letters, numbers, and underscores'})
            )
        
        offer = RTCSessionDescription(sdp=params["sdp"], type=params["type"])

        pc = RTCPeerConnection()
        peer_id = str(uuid.uuid4())
        pc_id = f"PeerConnection({peer_id[:8]})"
        self.pcs.add(pc)
        
        # Store peer info
        peer_info = PeerInfo(id=peer_id, handle=handle, peer_connection=pc)
        self.peers[peer_id] = peer_info

        def log_info(msg, *args):
            logger.info(pc_id + " " + msg, *args)

        log_info("Created for %s with handle %s", request.remote, handle)

        @pc.on("datachannel")
        def on_datachannel(channel):
            log_info("Data channel %s received", channel.label)
            
            # Store the data channel reference
            if peer_id in self.peers:
                self.peers[peer_id].data_channel = channel
            
            @channel.on("message")
            def on_message(message):
                asyncio.create_task(self.handle_data_channel_message(peer_id, message))

        @pc.on("connectionstatechange")
        async def on_connectionstatechange():
            log_info("Connection state is %s", pc.connectionState)
            if pc.connectionState in ["failed", "closed", "disconnected"]:
                await pc.close()
                self.pcs.discard(pc)
                # Remove peer and broadcast updated count
                if self.peers.pop(peer_id, None):
                    log_info("Peer %s removed, remaining peers: %d", handle, len(self.peers))
                    await self.broadcast_peer_count()
            elif pc.connectionState == "connected":
                await self.broadcast_peer_count()

        # handle offer
        await pc.setRemoteDescription(offer)

        # send answer
        answer = await pc.createAnswer()
        await pc.setLocalDescription(answer)
        
        # Broadcast updated peer count
        await self.broadcast_peer_count()

        return web.Response(
            content_type="application/json",
            text=json.dumps({
                "sdp": pc.localDescription.sdp, 
                "type": pc.localDescription.type,
                "peer_id": peer_id
            })
        )
    
    async def handle_data_channel_message(self, peer_id: str, message: str):
        """Handle messages received through data channel"""
        try:
            data = json.loads(message)
            message_type = data.get('type')
            
            if message_type == 'pose_request':
                await self.process_pose_request(peer_id, data)
            else:
                # Broadcast to other peers
                await self.broadcast_to_others(peer_id, data)
                
        except json.JSONDecodeError:
            logger.error(f"Invalid JSON in data channel message from {peer_id}")
        except Exception as e:
            logger.error(f"Error handling data channel message from {peer_id}: {e}")
    
    async def process_pose_request(self, peer_id: str, data: dict):
        """Process pose detection request"""
        try:
            # Extract image data
            image_data = data.get('image', '').split(',')[-1]
            if not image_data:
                return

            # Decode and process image with proper error handling
            try:
                image_bytes = base64.b64decode(image_data, validate=True)
            except Exception as e:
                logger.error(f"Invalid base64 image data from {peer_id}: {e}")
                return
                
            try:
                image = Image.open(io.BytesIO(image_bytes))
                image = image.transpose(Image.FLIP_LEFT_RIGHT)
            except Exception as e:
                logger.error(f"Invalid image format from {peer_id}: {e}")
                return

            start_time = time.time()
            result = dwpose(image, output_type="pil", include_hands=True, include_face=True)
            end_time = time.time()

            execution_time = end_time - start_time
            logger.debug(f"Pose detection took {execution_time:.4f} seconds")

            buffered = io.BytesIO()
            result.save(buffered, format="JPEG", quality=85)
            result_image = base64.b64encode(buffered.getvalue()).decode()

            # Send result back to requesting peer
            peer_info = self.peers.get(peer_id)
            if peer_info and peer_info.data_channel and peer_info.data_channel.readyState == "open":
                response_data = {
                    'type': 'pose_response',
                    'image': f"data:image/jpeg;base64,{result_image}",
                    'execution_time': execution_time
                }
                try:
                    peer_info.data_channel.send(json.dumps(response_data))
                    logger.debug(f"Sent pose response to peer {peer_id} (took {execution_time:.4f}s)")
                except Exception as e:
                    logger.error(f"Error sending pose response to {peer_id}: {e}")
            else:
                logger.warning(f"No data channel available for peer {peer_id}")

        except Exception as e:
            logger.error(f"Error processing pose request from {peer_id}: {e}")
    
    async def broadcast_to_others(self, sender_id: str, message: dict):
        """Broadcast message to all peers except sender"""
        # For data channel broadcasting, we'd need to track data channels
        # This is more complex with the aiortc model
        pass
    
    async def broadcast_peer_count(self):
        """Broadcast updated peer count to all status connections"""
        peer_list = [{'id': pid, 'handle': info.handle} for pid, info in self.peers.items()]
        message = json.dumps({
            'type': 'peer_count',
            'count': len(self.peers),
            'peers': peer_list
        })
        
        # Send to all status WebSocket connections
        dead_connections = []
        for peer_id, ws in self.signaling_connections.items():
            try:
                if not ws.closed:
                    await ws.send_str(message)
                else:
                    dead_connections.append(peer_id)
            except Exception as e:
                logger.error(f"Error broadcasting to {peer_id}: {e}")
                dead_connections.append(peer_id)
        
        # Clean up dead connections
        for peer_id in dead_connections:
            self.signaling_connections.pop(peer_id, None)

def create_app(password: str, cors_origin: str = "*"):
    server = WebRTCPoseServer(password, cors_origin)
    app = web.Application()
    app._webrtc_server = server  # Store reference for cleanup
    
    # Add CORS middleware
    @web.middleware
    async def cors_middleware(request, handler):
        if request.method == "OPTIONS":
            response = web.Response()
        else:
            response = await handler(request)

        response.headers['Access-Control-Allow-Origin'] = server.cors_origin
        response.headers['Access-Control-Allow-Methods'] = 'GET, POST, OPTIONS'
        response.headers['Access-Control-Allow-Headers'] = 'Content-Type'
        return response

    app.middlewares.append(cors_middleware)
    
    # Routes
    app.router.add_get('/ws', server.handle_websocket)  # Status updates only
    app.router.add_post('/offer', server.handle_webrtc_offer)  # WebRTC negotiation
    app.router.add_get('/health', lambda req: web.Response(text='ok'))
    
    return app

async def on_shutdown(app):
    # Get server instance and close peer connections
    server = getattr(app, '_webrtc_server', None)
    if server:
        coros = [pc.close() for pc in server.pcs]
        await asyncio.gather(*coros)
        server.pcs.clear()
        server.peers.clear()
        
        # Close any remaining WebSocket connections
        for ws in server.signaling_connections.values():
            if not ws.closed:
                await ws.close()

if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description="WebRTC Pose Server")
    parser.add_argument("--host", default="0.0.0.0", help="Host to bind to")
    parser.add_argument("--port", type=int, default=8020, help="Port to bind to")
    parser.add_argument("--password", required=True, help="Room password for WebRTC connections")
    parser.add_argument("--cors-origin", default="*", help="CORS origin (default: *)")
    parser.add_argument("--verbose", "-v", action="count", default=0)
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.basicConfig(level=logging.DEBUG)
    else:
        logging.basicConfig(level=logging.INFO)
    
    app = create_app(args.password, args.cors_origin)
    app.on_shutdown.append(on_shutdown)
    logger.info(f"Starting WebRTC Pose Server on {args.host}:{args.port}")
    logger.info(f"CORS origin: {args.cors_origin}")
    
    web.run_app(app, host=args.host, port=args.port)
