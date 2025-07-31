import os, time, uuid, json, logging

from typing import Optional
from contextvars import ContextVar
from fastapi import Request, Response
from datetime import datetime, timezone
from logging.handlers import RotatingFileHandler
from starlette.datastructures import Headers
from starlette.middleware.base import BaseHTTPMiddleware

request_id_context_var = ContextVar("request_id", default=None)

class RequestIdMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        request_id = generate_request_id()
        request_id_context_var.set(request_id)
        request.state.request_id = request_id
        
        start_time = time.time()
        response: Optional[Response] = None
        try:
            response = await call_next(request)
            return response
        finally:
            process_time = (time.time() - start_time) * 1000  # ms
            
            logger = logging.getLogger("API_LOG")
            status_code = response.status_code if response else 500
            
            headers: Headers = request.headers
            headers_dict = dict(headers.items())
            
            x_forwarded_for = headers.get("x-forwarded-for")
            x_real_ip = headers.get("x-real-ip")
            client_host = request.client.host if request.client else ""
            
            log_data = {
                "method": request.method,
                "path": request.url.path,
                "query_params": str(request.query_params) if request.query_params else "",
                "status_code": status_code,
                "duration_ms": f"{process_time:.2f}",
                "client_host": client_host,
                "x_forwarded_for": x_forwarded_for or "",
                "x_real_ip": x_real_ip or "",
                "host": f"{request.url.hostname}:{request.url.port}",
                "content_length": response.headers.get("content-length", "") if response else "",
                "protocol": request.scope.get("type", ""),
                # Wrap in quotes using JSON encoding
                "user_agent": json.dumps(headers.get("user-agent", "")),
                "referer": json.dumps(headers.get("referer", "")),
                # "headers": json.dumps(headers_dict)  # Log all headers
            }
            
            log_message = " ".join([f"{k}={v}" for k, v in log_data.items()])
            logger.info(log_message, extra={"request_id": request_id})


def generate_request_id():
    timestamp = int(time.time() * 1000)  # milliseconds since epoch
    unique_id = uuid.uuid4().hex[:8]  # first 4 characters of UUID
    return f"req_{timestamp:x}_{unique_id}"

class RequestIdFilter(logging.Filter):
    def filter(self, record):
        try:
            record.request_id = request_id_context_var.get()
        except LookupError:
            record.request_id = "uh oh"
        return True

class ISO8601UTCFormatter(logging.Formatter):
    def formatTime(self, record, datefmt=None):
        dt = datetime.fromtimestamp(record.created, timezone.utc)
        return dt.strftime('%Y-%m-%dT%H:%M:%S.%f')[:-3] + 'Z'

def setup_logging(
    log_file=None,
    log_level=logging.INFO,
    max_bytes=1024 * 1024,  # 1 MB
    backup_count=5
):
    log_dir = 'logs'
    os.makedirs(log_dir, exist_ok=True)

    if log_file is None:
        import inspect
        calling_file = inspect.stack()[-1].filename
        log_file = os.path.join(log_dir, f"{os.path.splitext(os.path.basename(calling_file))[0]}.log")
    else:
        log_file = os.path.join(log_dir, log_file)

    root_logger = logging.getLogger()
    root_logger.setLevel(log_level)
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)

    log_format = '%(asctime)s [%(request_id)s] %(name)s %(message)s'
    formatter = ISO8601UTCFormatter(log_format)
    request_id_filter = RequestIdFilter()

    file_handler = RotatingFileHandler(
        log_file,
        maxBytes=max_bytes,
        backupCount=backup_count
    )
    file_handler.setFormatter(formatter)
    file_handler.setLevel(log_level)
    file_handler.addFilter(request_id_filter)

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    console_handler.setLevel(log_level)
    console_handler.addFilter(request_id_filter)

    root_logger.addHandler(file_handler)
    root_logger.addHandler(console_handler)

    return root_logger


