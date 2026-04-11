import argparse
import json
import time
from typing import Dict, List, Tuple, Optional

from openai import OpenAI

"""
submit cmd
python batched_api_call.py submit  --problems-jsonl 40_train.jsonl --system-prompt-file system_prompt.txt  --batch-id-file batch_id.txt   --batch-input-jsonl batch_input.jsonl

fetch cmd


"""

MODEL = "gpt-5.2-2025-12-11"
ENDPOINT = "/v1/responses"
COMPLETION_WINDOW = "24h"
DEFAULT_POLL_INTERVAL_S = 30

REQUIRED_KEYS = {"folder", "family_id", "text", "problem_id"}


def load_problems_jsonl_strict(path: str) -> List[Dict[str, str]]:
    """
    Strictly reads JSONL where each line is a JSON object containing exactly
    the required keys at minimum:
      folder, family_id, text, problem_id

    - Errors if a line isn't valid JSON object.
    - Errors if required keys are missing.
    - Errors if problem_id duplicates.
    """
    problems: List[Dict[str, str]] = []
    seen_ids = set()

    with open(path, "r", encoding="utf-8") as f:
        for lineno, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue  # allow blank lines
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as e:
                raise ValueError(f"{path}: line {lineno}: invalid JSON: {e}") from e

            if not isinstance(obj, dict):
                raise ValueError(f"{path}: line {lineno}: expected JSON object, got {type(obj)}")

            missing = REQUIRED_KEYS - set(obj.keys())
            if missing:
                raise ValueError(f"{path}: line {lineno}: missing keys: {sorted(missing)}")

            # Force required fields to be strings
            folder = str(obj["folder"])
            family_id = str(obj["family_id"])
            text = str(obj["text"])
            problem_id = str(obj["problem_id"])

            if problem_id in seen_ids:
                raise ValueError(f"{path}: duplicate problem_id '{problem_id}' (line {lineno})")
            seen_ids.add(problem_id)

            problems.append(
                {
                    "folder": folder,
                    "family_id": family_id,
                    "text": text,
                    "problem_id": problem_id,
                }
            )

    if not problems:
        raise ValueError(f"{path}: no problems found (empty file or only blank lines)")

    return problems


def read_text_file(path: str) -> str:
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


# ----------------------------
# Batch input generation
# ----------------------------
def write_batch_input_jsonl(
    problems: List[Dict[str, str]],
    system_prompt: str,
    out_path: str,
    model: str = MODEL,
    endpoint: str = ENDPOINT,
) -> None:
    """
    Writes a batch input JSONL file. Each request uses custom_id=problem_id, so join is trivial.

    Batch input JSONL format: each line has custom_id, method, url, body. :contentReference[oaicite:5]{index=5}
    """
    with open(out_path, "w", encoding="utf-8") as f:
        for p in problems:
            problem_id = p["problem_id"]
            question_text = p["text"]

            req = {
                "custom_id": problem_id,   # join key
                "method": "POST",
                "url": endpoint,           # must match batch endpoint
                "body": {
                    "model": model,
                    "input": [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": question_text},
                    ],
                    "reasoning": {"effort": "high"},
                    # Optional: add if you want stricter length control
                    # "max_output_tokens": 400,
                    # "temperature": 0,
                },
            }
            f.write(json.dumps(req, ensure_ascii=False) + "\n")


# ----------------------------
# Batch submit / fetch
# ----------------------------
def submit_batch(
    client: OpenAI,
    batch_input_jsonl_path: str,
    completion_window: str = COMPLETION_WINDOW,
    endpoint: str = ENDPOINT,
) -> str:
    """
    Uploads input JSONL with purpose=batch and creates a batch job. :contentReference[oaicite:6]{index=6}
    """
    batch_file = client.files.create(
        file=open(batch_input_jsonl_path, "rb"),
        purpose="batch",
    )

    batch = client.batches.create(
        input_file_id=batch_file.id,
        endpoint=endpoint,
        completion_window=completion_window,
    )

    return batch.id


def poll_batch_to_terminal(
    client: OpenAI,
    batch_id: str,
    poll_interval_s: int = DEFAULT_POLL_INTERVAL_S,
) -> Dict:
    """
    Polls until the batch is in a terminal status.
    Terminal statuses are defined in the Batch API reference. :contentReference[oaicite:7]{index=7}
    """
    terminal = {"completed", "failed", "expired", "cancelled"}
    while True:
        batch = client.batches.retrieve(batch_id)
        status = getattr(batch, "status", None) or batch.get("status")
        if status in terminal:
            # Return as a plain dict-like structure
            return batch
        time.sleep(poll_interval_s)


def download_file_text(client: OpenAI, file_id: str) -> str:
    """
    Downloads file content using files.content(file_id). :contentReference[oaicite:8]{index=8}
    """
    resp = client.files.content(file_id)
    # SDK returns an object with .text in examples
    if hasattr(resp, "text") and isinstance(resp.text, str):
        return resp.text
    # fallback if it is bytes-like
    content = getattr(resp, "content", None)
    if isinstance(content, (bytes, bytearray)):
        return content.decode("utf-8", errors="replace")
    return str(resp)


def parse_jsonl(text: str) -> List[Dict]:
    out: List[Dict] = []
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        out.append(json.loads(line))
    return out


def extract_response_text_from_responses_api_body(body: Dict) -> str:
    """
    Responses API returns an 'output' array with content items; SDKs may also provide 'output_text'.
    Do not assume output[0].content[0].text; scan for output_text items. :contentReference[oaicite:9]{index=9}
    """
    if isinstance(body, dict) and isinstance(body.get("output_text"), str):
        return body["output_text"]

    texts: List[str] = []

    output = body.get("output", [])
    if not isinstance(output, list):
        return ""

    for item in output:
        if not isinstance(item, dict):
            continue

        # Common: item.type == "message" with item.content list containing {"type":"output_text","text":...}
        content = item.get("content")
        if isinstance(content, list):
            for c in content:
                if not isinstance(c, dict):
                    continue
                if c.get("type") in ("output_text", "text") and isinstance(c.get("text"), str):
                    texts.append(c["text"])

        # Less common: text directly on item
        if item.get("type") in ("output_text", "text") and isinstance(item.get("text"), str):
            texts.append(item["text"])

    return "\n".join(t.strip() for t in texts if t and t.strip()).strip()


def merge_to_results_jsonl(
    problems: List[Dict[str, str]],
    output_rows: List[Dict],
    error_rows: List[Dict],
    out_jsonl_path: str,
) -> Tuple[int, int]:
    """
    Writes ONE jsonl file with fields:
      problem_id, text, response

    Uses custom_id==problem_id to join. Output ordering may differ from input. :contentReference[oaicite:10]{index=10}
    """
    # Map problem_id -> response text (or None if failed)
    resp_map: Dict[str, Optional[str]] = {}

    for row in output_rows:
        cid = row.get("custom_id")
        resp = row.get("response") or {}
        body = resp.get("body") or {}
        if cid is not None:
            resp_map[str(cid)] = extract_response_text_from_responses_api_body(body)

    for row in error_rows:
        cid = row.get("custom_id")
        if cid is not None and str(cid) not in resp_map:
            resp_map[str(cid)] = None

    n_ok = 0
    n_fail = 0

    with open(out_jsonl_path, "w", encoding="utf-8") as f:
        for p in problems:
            pid = p["problem_id"]
            text = p["text"]
            response_text = resp_map.get(pid)
            if response_text is None:
                n_fail += 1
            else:
                n_ok += 1

            rec = {
                "problem_id": pid,
                "text": text,
                "response": response_text,
            }
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    return n_ok, n_fail


def fetch_and_write_results(
    client: OpenAI,
    batch_id: str,
    problems: List[Dict[str, str]],
    out_jsonl_path: str,
    poll_interval_s: int = DEFAULT_POLL_INTERVAL_S,
) -> None:
    batch = poll_batch_to_terminal(client, batch_id, poll_interval_s=poll_interval_s)
    status = getattr(batch, "status", None) or batch.get("status")

    output_file_id = getattr(batch, "output_file_id", None) or batch.get("output_file_id")
    error_file_id = getattr(batch, "error_file_id", None)

    output_rows: List[Dict] = []
    error_rows: List[Dict] = []

    if output_file_id:
        out_text = download_file_text(client, output_file_id)
        output_rows = parse_jsonl(out_text)

    if error_file_id:
        err_text = download_file_text(client, error_file_id)
        error_rows = parse_jsonl(err_text)

    n_ok, n_fail = merge_to_results_jsonl(problems, output_rows, error_rows, out_jsonl_path)

    print(f"batch_id={batch_id} status={status} ok={n_ok} failed_or_missing={n_fail}")
    print(f"Wrote: {out_jsonl_path}")


# ----------------------------
# CLI
# ----------------------------
def cmd_submit(args) -> None:
    client = OpenAI()

    problems = load_problems_jsonl_strict(args.problems_jsonl)
    system_prompt = read_text_file(args.system_prompt_file)

    write_batch_input_jsonl(
        problems=problems,
        system_prompt=system_prompt,
        out_path=args.batch_input_jsonl,
        model=args.model,
        endpoint=ENDPOINT,
    )
    import ipdb; ipdb.set_trace()
    batch_id = submit_batch(
        client=client,
        batch_input_jsonl_path=args.batch_input_jsonl,
        completion_window=COMPLETION_WINDOW,
        endpoint=ENDPOINT,
    )

    with open(args.batch_id_file, "w", encoding="utf-8") as f:
        f.write(batch_id + "\n")

    print(f"Submitted batch_id={batch_id}")
    print(f"Saved batch_id to: {args.batch_id_file}")


def cmd_fetch(args) -> None:
    client = OpenAI()

    problems = load_problems_jsonl_strict(args.problems_jsonl)

    with open(args.batch_id_file, "r", encoding="utf-8") as f:
        batch_id = f.read().strip()
    if not batch_id:
        raise ValueError(f"{args.batch_id_file}: empty batch id")

    fetch_and_write_results(
        client=client,
        batch_id=batch_id,
        problems=problems,
        out_jsonl_path=args.out_jsonl,
        poll_interval_s=args.poll_interval_s,
    )


def main():
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)

    p_submit = sub.add_parser("submit")
    p_submit.add_argument("--problems-jsonl", required=True, help="JSONL input; each line has folder,family_id,text,problem_id")
    p_submit.add_argument("--system-prompt-file", required=True, help="Text file containing SYSTEM_PROMPT")
    p_submit.add_argument("--batch-id-file", required=True, help="Where to write the created batch_id (text file)")
    p_submit.add_argument("--batch-input-jsonl", required=True, help="Where to write the batch input JSONL file")
    p_submit.add_argument("--model", default=MODEL) 
    p_submit.set_defaults(func=cmd_submit)

    p_fetch = sub.add_parser("fetch")
    p_fetch.add_argument("--problems-jsonl", required=True, help="Same JSONL input used for submit")
    p_fetch.add_argument("--batch-id-file", required=True, help="Text file containing batch_id")
    p_fetch.add_argument("--out-jsonl", required=True, help="Final merged JSONL with problem_id,text,response")
    p_fetch.add_argument("--poll-interval-s", type=int, default=DEFAULT_POLL_INTERVAL_S)
    p_fetch.set_defaults(func=cmd_fetch)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()