"""Submit a Character Swap API example to an already configured ComfyUI server."""
import argparse
import hashlib
import json
from pathlib import Path
import time
import urllib.error
import urllib.request
import uuid


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workflow", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--server", default="http://127.0.0.1:8188")
    parser.add_argument("--source", help="Source filename already present in ComfyUI/input")
    parser.add_argument("--reference", help="Character image filename already present in ComfyUI/input")
    parser.add_argument("--duration", type=float, help="Source duration in seconds; zero keeps the full source")
    args = parser.parse_args()
    graph = json.loads(args.workflow.read_text())
    if args.source:
        graph["source"]["inputs"]["file"] = args.source
    if args.reference:
        graph["reference"]["inputs"]["image"] = args.reference
    if args.duration is not None:
        graph["prepare"]["inputs"]["duration_seconds"] = args.duration
    args.output.mkdir(parents=True, exist_ok=True)
    server = args.server.rstrip("/")
    request = urllib.request.Request(
        server + "/prompt",
        data=json.dumps({"prompt": graph, "client_id": str(uuid.uuid4())}).encode(),
        headers={"Content-Type": "application/json"},
    )
    started = time.perf_counter()
    try:
        with urllib.request.urlopen(request, timeout=60) as response:
            submitted = json.load(response)
    except urllib.error.HTTPError as error:
        raise SystemExit(error.read().decode()) from error
    prompt_id = submitted["prompt_id"]
    print("Submitted", prompt_id, flush=True)
    (args.output / "submitted-api.json").write_text(json.dumps(graph, indent=2) + "\n")
    while True:
        with urllib.request.urlopen(server + "/history/" + prompt_id, timeout=60) as response:
            history = json.load(response)
        if prompt_id in history:
            result = history[prompt_id]
            record = {
                "elapsed_seconds": time.perf_counter() - started,
                "workflow_sha256": hashlib.sha256(args.workflow.read_bytes()).hexdigest(),
                "history": result,
            }
            (args.output / "run.json").write_text(json.dumps(record, indent=2) + "\n")
            print(result["status"]["status_str"], flush=True)
            if result["status"]["status_str"] != "success":
                raise SystemExit(1)
            return
        time.sleep(2)


if __name__ == "__main__":
    main()
