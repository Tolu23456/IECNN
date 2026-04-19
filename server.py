import http.server
import socketserver
import os

PORT = 5000
HOST = "0.0.0.0"

os.chdir(os.path.dirname(os.path.abspath(__file__)))

with socketserver.TCPServer((HOST, PORT), http.server.SimpleHTTPRequestHandler) as httpd:
    print(f"Serving on http://{HOST}:{PORT}")
    httpd.serve_forever()
