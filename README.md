 Bachelor's thesis project on
improving multi-step reasoning in a small language model (Qwen3-1.7B) under
personal-hardware constraints (single RTX 3060, 6GB VRAM), using a two-stage
**plan-and-solve** architecture and targeted, error-driven fine-tuning.
 
Full methodology, error taxonomy, and results are in the accompanying thesis report,
[*"Reasoning in Small Language Models for Mathematical Problem Solving Domain."*](Report.pdf)


## Approach
 
1. Split inference into two stages: the model first produces a **plan** (goal,
   unknowns, core constraints, dependency skeleton — no computation), then solves the
   problem conditioned on that plan.
2. Ran the base model on math word problems and analyzed the planning-stage outputs for
   recurring failure patterns.
3. Clustered problems into 4 groups by dominant error type, and selected the most
   critical cluster (long, multi-constraint numerical word problems) for intervention.
4. Generated a targeted dataset for that cluster: 50 parametrized problem families,
   500 total instances, rejection-sampled to guarantee validity, active constraints,
   and unique solutions.
5. Used a high-reasoning-effort teacher model (GPT-5.2, via batched API calls) to
   generate structured "gold" plans for every instance.
6. Fine-tuned the planning behavior of Qwen3-1.7B on this data using QLoRA
  , and evaluated the
   before/after planning and solving quality.
## Repository structure
 
| Path | Contents |
|---|---|
| `dataset_gen_codes/` | Generators for the 50 problem families (parametrized templates + rejection sampling) |
| `all50_families_manifest.json` | Manifest describing all 50 problem families |
| `api_call/` | Batched calls to the teacher model (GPT-5.2) to produce solution plans |
| `investigate_problem_types/` | Baseline error analysis / problem clustering |
| `example_tasks/` | Sample problems used to probe/demonstrate model behavior |
| `out_algebra/`, `out_calc/`, `out_dmopt/`, `out_geometry/`, `out_nt_fin/`, `out_stat/` | Generated problem instances, split by mathematical domain |
| `finetune/` | QLoRA/PEFT fine-tuning scripts for Qwen3-1.7B |
| `evaluate_finetuned.ipynb` | Before/after evaluation of the fine-tuned model |
| `base_model_responses.jsonl` | Sample outputs from the base (untuned) model |
| `recipe1_sample_responses.jsonl`, `recipe2_sample_responses.jsonl` | Sample outputs from the two QLoRA fine-tuning configurations tried |
| `responses.txt` | Additional raw response samples |
 

## Tools & environment
 
- **Local inference / error analysis:** Qwen3-1.7B, GGUF (Q8_0 quantization), served
  with `llama.cpp` (CUDA backend) on Windows.
- **Teacher data generation:** OpenAI API (GPT-5.2, high reasoning effort), batched
  requests.
- **Fine-tuning:** Hugging Face `transformers`, `datasets`, `peft` (LoRA adapters),
  `bitsandbytes` (  QLoRA on Qwen3-1.7B with 4-bit NF4 base weights and **FP16**
  compute/mixed-precision training ), adapters on
  `q_proj` / `k_proj` / `v_proj` / `o_proj`. 
- **Hardware:** single RTX 3060, 6GB VRAM.


## Example: Base Model vs. Fine-tuned Model Plans

<table>
  <tr>
    <td><img width="1298" height="834" alt="image2_english" src="https://github.com/user-attachments/assets/ab0b0d55-6893-4b34-a0ae-ee09c3dc1202" />
    <tf><img width="1370" height="792" alt="image3_clean" src="https://github.com/user-attachments/assets/b9a3012c-54ce-4e91-bc8b-163529b796b0" />
  </tr>

</table>

## License
 
MIT — see [LICENSE](LICENSE).
 
