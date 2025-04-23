import http.server
import socketserver

PORT = 8000

class ThreadSafeHTTPHandler(http.server.SimpleHTTPRequestHandler):
    def end_headers(self):
        self.send_header("Cross-Origin-Opener-Policy", "same-origin")
        self.send_header("Cross-Origin-Embedder-Policy", "require-corp")
        super().end_headers()

with socketserver.TCPServer(("", PORT), ThreadSafeHTTPHandler) as httpd:
    print(f"Serving at http://localhost:{PORT}")
    httpd.serve_forever()
