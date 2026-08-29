#!/usr/bin/env bash
# execute-code.sh: Execute code on a running Marimo server via API

set -e

URL=""
TOKEN=""
CODE_FILE=""
CODE_STR=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --url)
      URL="$2"
      shift 2
      ;;
    --token)
      TOKEN="$2"
      shift 2
      ;;
    --file)
      CODE_FILE="$2"
      shift 2
      ;;
    --code)
      CODE_STR="$2"
      shift 2
      ;;
    *)
      shift
      ;;
  esac
done

if [[ -z "$URL" ]]; then
  echo "Error: --url is required" >&2
  exit 1
fi

export MARIMO_TARGET_URL="${URL%/}"
export MARIMO_AUTH_TOKEN="${TOKEN}"
export MARIMO_CODE_STR="${CODE_STR}"
export MARIMO_CODE_FILE="${CODE_FILE}"

python3 - << 'PYEOF'
import sys
import os
import json
import urllib.request

url = os.environ.get("MARIMO_TARGET_URL", "")
token = os.environ.get("MARIMO_AUTH_TOKEN", "")
code_str = os.environ.get("MARIMO_CODE_STR", "")
code_file = os.environ.get("MARIMO_CODE_FILE", "")

headers = {
    "User-Agent": "Mozilla/5.0",
}
if token:
    headers["Authorization"] = f"Bearer {token}"

def get_session_id():
    req = urllib.request.Request(f"{url}/api/sessions", headers=headers)
    with urllib.request.urlopen(req) as resp:
        data = json.loads(resp.read().decode("utf-8"))
        session_ids = list(data.keys())
        if not session_ids:
            raise RuntimeError("No active sessions on Marimo server!")
        return session_ids[0]

code = code_str
if not code:
    if code_file:
        with open(code_file, "r", encoding="utf-8") as f:
            code = f.read()
    else:
        code = sys.stdin.read()

if not code.strip():
    print("No code provided to execute.")
    sys.exit(0)

sid = get_session_id()
exec_url = f"{url}/api/kernel/execute"
payload = json.dumps({"code": code}).encode("utf-8")
req_headers = dict(headers)
req_headers["Content-Type"] = "application/json"
req_headers["Marimo-Session-Id"] = sid

req = urllib.request.Request(exec_url, data=payload, headers=req_headers, method="POST")
with urllib.request.urlopen(req) as resp:
    current_event = None
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
                        err = payload_data.get("error", {}).get("msg", "Unknown error")
                        sys.stderr.write(f"\n[Execution Error] {err}\n")
                        sys.exit(1)
            except json.JSONDecodeError:
                pass
PYEOF
