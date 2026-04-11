from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
import torch

SYSTEM_PROMPT_STRONG = """
You are a human-like math problem PLANNER ONLY. Do not solve. Do not compute/derive/simplify. No numeric results.

Task: extract ONLY what is necessary to reach the asked-for quantity <ANS>.
Omit anything not used downstream.

Output exactly one plain-text block:
<<<PLAN>>>
Goal: <ANS> = (what the problem asks, in words)
Unknowns:
- <U1>=...
- <U2>=...   
(only true unknowns / decision outcomes)
Core constraints:
- <C1>: ...
- <C2>: ...
Dependency skeleton:
- <R1> from <C?> and <U?>
- <R2> from <C?> and <R1>
Plan:
- Identify the minimum set of intermediates needed for <ANS>.
- Compute ... -> <R1>.
- Compute ... -> <R2>.
- Combine intermediates to express <ANS>. (still no execution)

<<<END>>>

Rules:
- Do NOT create placeholders for fixed givens; refer to them as “given in the statement”.
- Do NOT name standard sub-parameters unless they are required intermediates; use <Rk> instead.
- No equations or operator symbols; describe relations in words.
- Nothing outside the block.
"""

QUESTION_TEXT = """
In a class of 127 students, 40 take Math, 78 take Physics, 60 take CS. Exactly 33 take Math and Physics, 12 take Math and CS, 21 take Physics and CS, and 9 take all three. Additionally, 6 take none. Determine how many take exactly one subject and verify consistency of the data.
"""

MODEL_ID = "Qwen/Qwen3-1.7B"
ADAPTER_DIR = "qwen3_planner_recipe2_lora/checkpoint-33"  # your saved adapter dir

tok = AutoTokenizer.from_pretrained(MODEL_ID, use_fast=True)
base = AutoModelForCausalLM.from_pretrained(
    MODEL_ID, torch_dtype=torch.float16, device_map="auto"
)
model = PeftModel.from_pretrained(base, ADAPTER_DIR)
model.eval()

messages = [
    {"role": "system", "content": SYSTEM_PROMPT_STRONG},
    {"role": "user", "content": QUESTION_TEXT},
]
prompt = tok.apply_chat_template(
    messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
)
inp = tok(prompt, return_tensors="pt").to(model.device)

with torch.no_grad():
    out = model.generate(**inp, max_new_tokens=800, do_sample=False)
print(tok.decode(out[0][inp["input_ids"].shape[1] :], skip_special_tokens=False))
