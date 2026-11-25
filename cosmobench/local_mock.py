"""Mock Anthropic Messages server: dry-run the local-model mode without loading a model.

    python -m cosmobench.local_mock --port 9939 --log results/v5/local-mock/requests.jsonl

It answers `/v1/messages` (streaming or not) and `/v1/messages/count_tokens` like a
llama-server would, with a fixed script, and logs every request (model, sampling fields,
thinking/effort fields, beta headers, tool names, prompt size, concurrency). The script:

- the request offers the Skill tool and no tool has run yet: call Skill on the hub
  (`cosmoskill:observational-cosmology`), so the log shows the context after a skill load;
- the request offers Bash and no tool has run yet: run `fetch_paper.py <first arXiv id>
  --abstract` with the path given in the system prompt (exercises the gate hook);
- otherwise: one `N: cannot verify — mock` line per numbered claim (or one line for a question).

Standard library only. Binds 127.0.0.1 only.
"""
from __future__ import annotations

import argparse
import json
import re
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

LOCK = threading.Lock()
STATE = {"in_flight": 0, "max_in_flight": 0, "n": 0}
CHARS_PER_TOKEN = 3.5  # rough, for the log only


def _text(content) -> str:
    if isinstance(content, str):
        return content
    out = []
    for b in content if isinstance(content, list) else []:
        if not isinstance(b, dict):
            continue
        if b.get("type") == "text":
            out.append(b.get("text", ""))
        elif b.get("type") == "tool_result":
            out.append(_text(b.get("content")))
        elif b.get("type") == "tool_use":
            out.append(json.dumps(b.get("input") or {}))
    return "\n".join(out)


def _system_text(system) -> str:
    return system if isinstance(system, str) else _text(system)


def _tool_results(messages) -> list[str]:
    out = []
    for m in messages:
        c = m.get("content")
        for b in c if isinstance(c, list) else []:
            if isinstance(b, dict) and b.get("type") == "tool_result":
                out.append(_text(b.get("content")))
    return out


def plan_reply(body: dict) -> dict:
    """The scripted reply: {"tool": name, "input": {...}} or {"text": "..."}."""
    tools = {t.get("name"): t for t in body.get("tools") or [] if isinstance(t, dict)}
    messages = body.get("messages") or []
    first_user = _text(messages[0].get("content")) if messages else ""
    done = _tool_results(messages)
    if not done and "Skill" in tools:
        schema = (tools["Skill"].get("input_schema") or {}).get("properties") or {}
        key = "skill" if "skill" in schema else (next(iter(schema)) if schema else "skill")
        return {"tool": "Skill", "input": {key: "cosmoskill:observational-cosmology"}}
    if not done and "Bash" in tools:
        m = re.search(r"python3 (\S+fetch_paper\.py)", _system_text(body.get("system")))
        a = re.search(r"arXiv:(\d{4}\.\d{4,5})", first_user)
        if m and a:
            return {"tool": "Bash", "input": {"command": f"python3 {m.group(1)} {a.group(1)} --abstract",
                                              "description": "fetch the abstract"}}
    n = len(re.findall(r"^\d+\. ", first_user, flags=re.M))
    lines = [f"{k}: cannot verify — mock" for k in range(1, n + 1)] or ["mock answer: I cannot verify this."]
    return {"text": "\n".join(lines)}


def _usage(body: dict) -> tuple[int, int]:
    size = len(_system_text(body.get("system"))) + sum(len(_text(m.get("content"))) for m in body.get("messages") or [])
    size += len(json.dumps(body.get("tools") or []))
    return int(size / CHARS_PER_TOKEN), 20


def summarise(path: str, headers, body: dict) -> dict:
    tools = [t.get("name") for t in body.get("tools") or [] if isinstance(t, dict)]
    msgs = body.get("messages") or []
    results = _tool_results(msgs)
    return {
        "path": path,
        "model": body.get("model"),
        "stream": body.get("stream"),
        "keys": sorted(body),
        "max_tokens": body.get("max_tokens"),
        "temperature": body.get("temperature"),
        "top_p": body.get("top_p"),
        "top_k": body.get("top_k"),
        "thinking": body.get("thinking"),
        "output_config": body.get("output_config"),
        "context_management": body.get("context_management"),
        "metadata_keys": sorted((body.get("metadata") or {}).keys()),
        "beta": headers.get("anthropic-beta"),
        "auth": "x-api-key" if headers.get("x-api-key") else ("bearer" if headers.get("authorization") else "none"),
        "n_tools": len(tools),
        "tools": tools,
        "system_chars": len(_system_text(body.get("system"))),
        "tools_chars": len(json.dumps(body.get("tools") or [])),
        "messages": len(msgs),
        "message_chars": sum(len(_text(m.get("content"))) for m in msgs),
        "tool_result_chars": [len(r) for r in results],
        "est_prompt_tokens": _usage(body)[0],
    }


class Handler(BaseHTTPRequestHandler):
    log_path: Path = Path("requests.jsonl")
    protocol_version = "HTTP/1.1"

    def log_message(self, *args):  # silence the default stderr log
        pass

    def _send_json(self, code: int, obj: dict) -> None:
        data = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("content-type", "application/json")
        self.send_header("content-length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def _log(self, rec: dict) -> None:
        with LOCK:
            self.log_path.parent.mkdir(parents=True, exist_ok=True)
            with self.log_path.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(rec, ensure_ascii=False) + "\n")

    def do_HEAD(self):
        self.send_response(200)
        self.send_header("content-length", "0")
        self.end_headers()
        self._log({"path": self.path, "method": "HEAD", "t": time.time()})

    def do_GET(self):
        self._log({"path": self.path, "method": "GET", "t": time.time()})
        if self.path.startswith("/props"):
            self._send_json(200, {"default_generation_settings": {"n_ctx": 131072}, "model_alias": "mock"})
        else:
            self._send_json(200, {"status": "ok"})

    def do_POST(self):
        n = int(self.headers.get("content-length") or 0)
        try:
            body = json.loads(self.rfile.read(n) or b"{}")
        except json.JSONDecodeError:
            body = {}
        with LOCK:
            STATE["in_flight"] += 1
            STATE["n"] += 1
            STATE["max_in_flight"] = max(STATE["max_in_flight"], STATE["in_flight"])
            in_flight = STATE["in_flight"]
        t0 = time.time()
        try:
            rec = summarise(self.path.split("?")[0], self.headers, body)
            rec.update({"method": "POST", "t": t0, "in_flight": in_flight})
            if self.path.startswith("/v1/messages/count_tokens"):
                self._send_json(200, {"input_tokens": _usage(body)[0]})
            elif self.path.startswith("/v1/messages"):
                reply = plan_reply(body)
                rec["reply"] = reply.get("tool") or "text"
                time.sleep(0.2)  # long enough to see overlapping requests in the log
                if body.get("stream"):
                    self._stream(body, reply)
                else:
                    self._send_json(200, self._message(body, reply))
            else:
                self._send_json(404, {"error": "not found"})
        finally:
            with LOCK:
                STATE["in_flight"] -= 1
            rec["dt"] = round(time.time() - t0, 3)
            self._log(rec)

    @staticmethod
    def _block(reply: dict) -> dict:
        if "tool" in reply:
            return {"type": "tool_use", "id": f"toolu_mock{STATE['n']:05d}", "name": reply["tool"],
                    "input": reply["input"]}
        return {"type": "text", "text": reply["text"]}

    def _message(self, body: dict, reply: dict) -> dict:
        inp, out = _usage(body)
        return {"id": f"msg_mock{STATE['n']:05d}", "type": "message", "role": "assistant",
                "model": body.get("model") or "mock", "content": [self._block(reply)],
                "stop_reason": "tool_use" if "tool" in reply else "end_turn", "stop_sequence": None,
                "usage": {"input_tokens": inp, "output_tokens": out}}

    def _stream(self, body: dict, reply: dict) -> None:
        inp, out = _usage(body)
        msg = self._message(body, reply)
        block = msg["content"][0]
        start = dict(msg, content=[], stop_reason=None, usage={"input_tokens": inp, "output_tokens": 1})
        if block["type"] == "tool_use":
            first = dict(block, input={})
            delta = {"type": "input_json_delta", "partial_json": json.dumps(block["input"])}
        else:
            first = {"type": "text", "text": ""}
            delta = {"type": "text_delta", "text": block["text"]}
        events = [
            ("message_start", {"type": "message_start", "message": start}),
            ("content_block_start", {"type": "content_block_start", "index": 0, "content_block": first}),
            ("content_block_delta", {"type": "content_block_delta", "index": 0, "delta": delta}),
            ("content_block_stop", {"type": "content_block_stop", "index": 0}),
            ("message_delta", {"type": "message_delta",
                               "delta": {"stop_reason": msg["stop_reason"], "stop_sequence": None},
                               "usage": {"output_tokens": out}}),
            ("message_stop", {"type": "message_stop"}),
        ]
        payload = "".join(f"event: {e}\ndata: {json.dumps(d)}\n\n" for e, d in events).encode()
        self.send_response(200)
        self.send_header("content-type", "text/event-stream")
        self.send_header("cache-control", "no-cache")
        self.send_header("content-length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=9939)
    ap.add_argument("--log", type=Path, required=True)
    a = ap.parse_args(argv)
    Handler.log_path = a.log
    srv = ThreadingHTTPServer(("127.0.0.1", a.port), Handler)
    print(f"mock Anthropic server on http://127.0.0.1:{a.port}, log {a.log}", flush=True)
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
