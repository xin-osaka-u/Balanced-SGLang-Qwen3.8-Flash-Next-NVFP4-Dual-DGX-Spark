#!/usr/bin/env python3
"""Tiny OpenAI-API proxy: forwards to the SGLang server and injects
chat_template_kwargs.enable_thinking=false into every chat request, so
evaluation harnesses that cannot send nested kwargs (lm_eval) run in the
same non-thinking mode our own clients use. Logs one line per request."""
import json, os, sys, time, urllib.request, urllib.error
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
UP = os.environ.get("NT_UPSTREAM", "http://127.0.0.1:8000")
PORT = int(os.environ.get("NT_PORT", "8001"))
LOG = os.environ.get("NT_LOG", os.path.expanduser("./nt_proxy.log"))
INJECT = os.environ.get("NT_INJECT", "1") != "0"
def log(s):
    with open(LOG, "a") as f: f.write(time.strftime("%H:%M:%S ") + s + "\n")
class H(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    def log_message(self, *a): pass
    def _fwd(self, method, body=None, ctype=None):
        req = urllib.request.Request(UP + self.path, data=body, method=method)
        if ctype: req.add_header("Content-Type", ctype)
        try:
            with urllib.request.urlopen(req, timeout=3600) as r:
                return r.status, r.headers.get("Content-Type", "application/json"), r.read()
        except urllib.error.HTTPError as e:
            return e.code, e.headers.get("Content-Type", "application/json"), e.read()
    def _send(self, st, ct, data):
        self.send_response(st); self.send_header("Content-Type", ct)
        self.send_header("Content-Length", str(len(data))); self.end_headers(); self.wfile.write(data)
    def do_GET(self):
        self._send(*self._fwd("GET"))
    def do_POST(self):
        n = int(self.headers.get("Content-Length", "0")); raw = self.rfile.read(n)
        t0 = time.time(); tag = ""
        if INJECT and self.path.startswith("/v1/chat/completions"):
            try:
                b = json.loads(raw); k = dict(b.get("chat_template_kwargs") or {}); k["enable_thinking"] = False
                b["chat_template_kwargs"] = k; raw = json.dumps(b).encode(); tag = " nt"
            except Exception as e: tag = " inject-failed:%s" % e
        st, ct, data = self._fwd("POST", raw, self.headers.get("Content-Type", "application/json"))
        u = {}
        try: u = json.loads(data).get("usage") or {}
        except Exception: pass
        log("%s %s%s st=%s p=%s c=%s %.1fs" % (self.path, self.command, tag, st, u.get("prompt_tokens"), u.get("completion_tokens"), time.time() - t0))
        self._send(st, ct, data)
if __name__ == "__main__":
    log("proxy start port=%d upstream=%s inject=%s" % (PORT, UP, INJECT))
    ThreadingHTTPServer(("127.0.0.1", PORT), H).serve_forever()
