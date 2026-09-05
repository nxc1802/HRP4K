#!/usr/bin/env python3
"""Marimo remote execution script for pair programming on running Marimo instances."""

import argparse
import json
import sys
import urllib.request
import urllib.error

def get_session_id(url: str, token: str) -> str:
    headers = {
        "User-Agent": "Mozilla/5.0",
        "Authorization": f"Bearer {token}",
    }
    req = urllib.request.Request(f"{url.rstrip('/')}/api/sessions", headers=headers)
    with urllib.request.urlopen(req) as resp:
        data = json.loads(resp.read().decode("utf-8"))
        if isinstance(data, dict):
            session_ids = list(data.keys())
        elif isinstance(data, list):
            session_ids = [s.get("session_id") or s.get("id") for s in data]
        else:
            session_ids = []
        if not session_ids:
            raise RuntimeError(f"No active sessions on Marimo server! Response: {data}")
        return session_ids[0]

def execute_code(url: str, token: str, code: str, session_id: str | None = None) -> bool:
    if session_id is None:
        session_id = get_session_id(url, token)
    
    endpoint = f"{url.rstrip('/')}/api/kernel/execute"
    payload = json.dumps({"code": code}).encode("utf-8")
    
    headers = {
        "User-Agent": "Mozilla/5.0",
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json",
        "Marimo-Session-Id": session_id,
    }
    
    req = urllib.request.Request(endpoint, data=payload, headers=headers, method="POST")
    current_event = None
    success = True
    
    try:
        with urllib.request.urlopen(req) as resp:
            for raw_line in resp:
                line = raw_line.decode("utf-8", errors="ignore").rstrip("\r\n")
                if line.startswith("event:"):
                    current_event = line[len("event:"):].strip()
                elif line.startswith("data:"):
                    data_str = line[len("data:"):].strip()
                    try:
                        payload_data = json.loads(data_str)
                        if current_event in ("stdout", "stderr"):
                            content = payload_data.get("data", "")
                            if current_event == "stdout":
                                sys.stdout.write(content)
                                sys.stdout.flush()
                            else:
                                sys.stderr.write(content)
                                sys.stderr.flush()
                        elif current_event == "done":
                            if not payload_data.get("success", True):
                                err = payload_data.get("error", {})
                                err_msg = err.get("msg") or str(err)
                                sys.stderr.write(f"\n[Execution Error] {err_msg}\n")
                                success = False
                            output_data = payload_data.get("output", {}).get("data")
                            if output_data:
                                print(output_data)
                    except json.JSONDecodeError:
                        pass
    except urllib.error.HTTPError as e:
        sys.stderr.write(f"HTTP Error {e.code}: {e.read().decode('utf-8', errors='ignore')}\n")
        return False
    except Exception as e:
        sys.stderr.write(f"Error connecting to Marimo: {e}\n")
        return False
        
    return success

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", required=True)
    parser.add_argument("--token", required=True)
    parser.add_argument("--code", default=None)
    parser.add_argument("--session-id", default=None)
    args, unknown = parser.parse_known_args()
    
    if args.code:
        code_str = args.code
    else:
        code_str = sys.stdin.read()
        
    ok = execute_code(args.url, args.token, code_str, args.session_id)
    sys.exit(0 if ok else 1)
