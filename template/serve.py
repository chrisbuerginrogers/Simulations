#!/usr/bin/env python3
"""Serves index.html on localhost. Run: python3 serve.py"""
import http.server
import os
import socketserver
import webbrowser

PORT = 8766

os.chdir(os.path.dirname(os.path.abspath(__file__)))


class Handler(http.server.SimpleHTTPRequestHandler):
    def log_message(self, format, *args):
        pass  # keep the terminal quiet; errors still show via exceptions


if __name__ == "__main__":
    with socketserver.TCPServer(("127.0.0.1", PORT), Handler) as httpd:
        url = f"http://127.0.0.1:{PORT}/index.html"
        print(f"Generic mjlab task config builder running at {url}  (Ctrl+C to stop)")
        webbrowser.open(url)
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            print("\nStopped.")
