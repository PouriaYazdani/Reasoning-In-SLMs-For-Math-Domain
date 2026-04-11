import json
import random
from pathlib import Path

SYSTEM_PROMPT_PATH = Path("system_prompt.txt")
INPUT_JSONL_PATH = Path("40_train_responses.jsonl")  # each line: {"problem_id":"...", "text":"...", "response":"..."}
TRAIN_JSONL_PATH = Path("../finetune/train.jsonl")               # each line: {"messages":[...]}
VALID_JSONL_PATH = Path("../finetune/validation.jsonl")          # each line: {"messages":[...]}
SEED = 47

# Validation families (5 families => 50 samples if each family has 10 problems)
VALID_FAMILIES = ["f10", "f20", "f39", "f34", "f48"]

system_prompt = SYSTEM_PROMPT_PATH.read_text(encoding="utf-8")
if not system_prompt.strip():
    raise RuntimeError("system_prompt.txt is empty or whitespace.")

def family_from_problem_id(problem_id: str) -> str:
    # "f01_01" -> "f01", "f10_03" -> "f10"
    if "_" not in problem_id:
        raise RuntimeError(f"Invalid problem_id format (expected like 'f01_01'), got: {problem_id!r}")
    return problem_id.split("_", 1)[0]

train_rows = []
valid_rows = []

# Read input jsonl
with INPUT_JSONL_PATH.open("r", encoding="utf-8") as f:
    for line_no, line in enumerate(f, start=1):
        line = line.strip()
        if not line:
            raise RuntimeError(f"Empty line at {INPUT_JSONL_PATH}:{line_no}")

        obj = json.loads(line)

        required = {"problem_id", "text", "response"}
        if not required.issubset(set(obj.keys())):
            raise RuntimeError(
                f"Expected keys {required} at {INPUT_JSONL_PATH}:{line_no}, got {set(obj.keys())}"
            )

        if not isinstance(obj["problem_id"], str):
            raise RuntimeError(f"'problem_id' must be a string at {INPUT_JSONL_PATH}:{line_no}")
        if not isinstance(obj["text"], str) or not isinstance(obj["response"], str):
            raise RuntimeError(f"'text' and 'response' must be strings at {INPUT_JSONL_PATH}:{line_no}")
        if not obj["problem_id"].strip():
            raise RuntimeError(f"'problem_id' is empty/whitespace at {INPUT_JSONL_PATH}:{line_no}")
        if not obj["text"].strip():
            raise RuntimeError(f"'text' is empty/whitespace at {INPUT_JSONL_PATH}:{line_no}")
        if not obj["response"].strip():
            raise RuntimeError(f"'response' is empty/whitespace at {INPUT_JSONL_PATH}:{line_no}")

        sample = {
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": obj["text"]},
                {"role": "assistant", "content": obj["response"]},
            ]
        }

        fam = family_from_problem_id(obj["problem_id"])
        if fam in VALID_FAMILIES:
            valid_rows.append(sample)
        else:
            train_rows.append(sample)

if not train_rows and not valid_rows:
    raise RuntimeError("No samples loaded from input.jsonl")

# Deterministic shuffle (separately per split)
rng = random.Random(SEED)
rng.shuffle(train_rows)
rng.shuffle(valid_rows)

# Write train jsonl
with TRAIN_JSONL_PATH.open("w", encoding="utf-8") as f:
    for r in train_rows:
        f.write(json.dumps(r, ensure_ascii=False) + "\n")

# Write validation jsonl
with VALID_JSONL_PATH.open("w", encoding="utf-8") as f:
    for r in valid_rows:
        f.write(json.dumps(r, ensure_ascii=False) + "\n")

print(f"Validation families: {VALID_FAMILIES}")
print(f"Wrote {len(train_rows)} samples to {TRAIN_JSONL_PATH}")
print(f"Wrote {len(valid_rows)} samples to {VALID_JSONL_PATH}")