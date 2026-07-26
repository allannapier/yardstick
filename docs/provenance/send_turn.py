import json
import sys
import urllib.request

MASTER_KEY = open("/home/allan/code/measure/docs/provenance/.master_key").read().strip()


def send(run_id, messages):
    body = {
        "model": "probe-claude-mock",
        "max_tokens": 50,
        "system": "You are a coding agent.",
        "messages": messages,
    }
    req = urllib.request.Request(
        "http://localhost:4000/v1/chat/completions",
        data=json.dumps(body).encode(),
        headers={
            "Authorization": f"Bearer {MASTER_KEY}",
            "Content-Type": "application/json",
            "x-ys-run": run_id,
        },
        method="POST",
    )
    with urllib.request.urlopen(req) as resp:
        return json.load(resp)


if __name__ == "__main__":
    run_id = sys.argv[1]
    messages = json.loads(sys.argv[2])
    print(send(run_id, messages))
