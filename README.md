# WebRTC Pose Server

A WebRTC-based pose detection server that handles real-time pose processing through WebRTC data channels.

## Requirements

- Python 3.8+
- Dependencies listed in requirements.txt (if available)

## Installation

```bash
pip install aiohttp aiortc pillow torch
```

## Usage

Start the server with required parameters:

```bash
python webrtc.py --password YOUR_SECURE_PASSWORD
```

### Command Line Options

- `--password` (required): Room password for WebRTC connections
- `--host`: Host to bind to (default: 0.0.0.0)
- `--port`: Port to bind to (default: 8020)
- `--cors-origin`: CORS origin header (default: *)
- `--verbose`, `-v`: Enable verbose logging

### Examples

Basic server:
```bash
python webrtc.py --password mySecretPassword123
```

Custom host/port with restricted CORS:
```bash
python webrtc.py --password mySecretPassword123 --host 127.0.0.1 --port 9000 --cors-origin "https://myapp.com"
```

Debug mode:
```bash
python webrtc.py --password mySecretPassword123 --verbose
```

## API Endpoints

- `GET /ws` - WebSocket for status updates and peer count
- `POST /offer` - WebRTC offer endpoint for establishing connections
- `GET /health` - Health check endpoint

## Security Notes

- Always use a strong password for the `--password` parameter
- Restrict CORS origins in production using `--cors-origin`
- Consider running behind a reverse proxy for additional security