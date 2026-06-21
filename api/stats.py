import json
from http.server import BaseHTTPRequestHandler

# Must match the values used in ingest.py
CHUNK_SIZE = 600
OVERLAP = 60
TOP_K = 8


class handler(BaseHTTPRequestHandler):

    def do_GET(self):
        result = {
            "chunk_size": CHUNK_SIZE,
            "overlap_ratio": round(OVERLAP / CHUNK_SIZE, 2),
            "top_k": TOP_K,
        }

        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(json.dumps(result).encode())
