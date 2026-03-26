"""
Reverse proxy — routes path-prefixed requests to the correct agent's internal Flask server.

/lead-enrichment/webhook      → localhost:5010/webhook
/competitor-intel/webhook/sync → localhost:5011/webhook/sync
...etc

Runs on PORT (default 10000, Render's default exposed port).
"""

import os
import http.server
import urllib.request
import urllib.error

AGENT_ROUTES: dict[str, int] = {
    "lead-enrichment": 5010,
    "competitor-intel": 5011,
    "content-pipeline": 5012,
    "reputation-monitor": 5013,
    "orchestrator": 5014,
    "pricehawk": 5015,
    "talentradar": 5016,
    "adrecon": 5017,
    "alphascout": 5018,
    "dealflow": 5019,
}


class ProxyHandler(http.server.BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path == "/" or self.path == "/health":
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(b'{"status":"ok","agents":10}')
            return
        self._proxy()

    def do_POST(self):
        self._proxy()

    def do_PATCH(self):
        self._proxy()

    def _proxy(self):
        parts = self.path.strip("/").split("/", 1)
        agent_slug = parts[0]
        rest = "/" + parts[1] if len(parts) > 1 else "/"

        port = AGENT_ROUTES.get(agent_slug)
        if not port:
            self.send_response(404)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(f'{{"error":"unknown agent: {agent_slug}"}}'.encode())
            return

        target_url = f"http://127.0.0.1:{port}{rest}"

        content_length = int(self.headers.get("Content-Length", 0))
        body = self.rfile.read(content_length) if content_length > 0 else None

        req = urllib.request.Request(
            target_url,
            data=body,
            method=self.command,
        )
        for key, val in self.headers.items():
            if key.lower() not in ("host", "transfer-encoding"):
                req.add_header(key, val)

        try:
            with urllib.request.urlopen(req, timeout=300) as resp:
                self.send_response(resp.status)
                for key, val in resp.getheaders():
                    if key.lower() not in ("transfer-encoding",):
                        self.send_header(key, val)
                self.end_headers()
                self.wfile.write(resp.read())
        except urllib.error.HTTPError as e:
            self.send_response(e.code)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(e.read())
        except (urllib.error.URLError, ConnectionRefusedError):
            self.send_response(502)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(f'{{"error":"agent {agent_slug} not ready"}}'.encode())

    def log_message(self, format, *args):
        print(f"[proxy] {args[0]}")


def run_proxy():
    port = int(os.environ.get("PORT", 10000))
    server = http.server.HTTPServer(("0.0.0.0", port), ProxyHandler)
    print(f"[proxy] listening on :{port}")
    server.serve_forever()


if __name__ == "__main__":
    run_proxy()
