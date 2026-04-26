"""Patch nb13 v3 — push to TRUE SOTA per tier. T4=7B, A100=14B, H100=32B."""
import json
from pathlib import Path

PATH = Path('notebooks/13_MASTER_HACKATHON_FINAL.ipynb')
nb = json.loads(PATH.read_text(encoding='utf-8'))


def lines(s):
    return [l + '\n' for l in s.splitlines()]


# Cell 0 — title + UPDATED instructions for judges
nb['cells'][0]['source'] = lines("""# 🚀 SupplyMind Master Hackathon Notebook

## *The single canonical OpenEnv India 2026 submission run*

**Theme 3 Professional Tasks** (with Theme 1 Multi-Agent + Theme 2 Long-Horizon hat-trick) · **License**: MIT · **Live**: [shaurya-noodle-supplymind.hf.space](https://shaurya-noodle-supplymind.hf.space)

---

### 📋 Judges: how to run this

**Step 1**: pick your runtime mode in cell 1 (`MODE = ...`).

| MODE | Runtime to select in Colab | LLM trained in §6 GRPO | Total wallclock | Cost |
|---|---|---|---|---|
| `cpu_quick` | Runtime → CPU | (skipped, REINFORCE only) | **~10 min** | $0 (free) |
| `t4_full` | Runtime → T4 GPU | **Qwen2.5-7B-Instruct** (4-bit Unsloth) | **~75 min** | $0 (free Colab T4) |
| `a100_max` | Runtime → A100 GPU | **Qwen2.5-14B-Instruct** (4-bit Unsloth) | **~55 min** | Pro Colab credits |
| `h100_beast` | Runtime → H100 GPU | **Qwen2.5-32B-Instruct** (4-bit Unsloth) | **~50 min** | Pro Colab credits |

**Step 2**: Runtime → Run all. The notebook auto-detects your GPU, picks the SOTA model that fits, configures bf16/fp16, and runs all 13 sections end-to-end.

**Step 3**: at the bottom you get a sha256-stamped HTML submission certificate with every metric.

---

### 📋 What this notebook does

13 sections, each emitting a sha256-stamped JSON receipt + inline plot + assertion check.

| § | Section | Emits |
|---|---|---|
| **0** | Setup + repo clone + GPU detect + helpers | env vars, GPU info, retry helper |
| **1** | OpenEnv compliance + 269-attack adversarial defense gauntlet | 100% blocked verification |
| **2** | Live HF Space rollout (PROVES env is live RIGHT NOW) | sha256-stamped step trace |
| **3** | REINFORCE Wordle 1500-ep · 8% → 100% · p=2.71e⁻¹⁸ · d=4.28 | reward curve PNG + raw arrays |
| **4** | Conformal action filter LIVE viz (Vovk 2005) | 9 of 102 actions accepted at α=0.10 |
| **5** | Process supervision step-credit trajectory (Lightman 2023) | variance amplification chart |
| **6** | GRPO 200-step on Qwen2.5 (1.5B/7B/14B/32B per MODE) + Unsloth | merged_16bit checkpoint |
| **7** | Baseline grid: DQN + QRDQN + TRPO + A2C real episodic | leaderboard JSON |
| **8** | Cross-env transfer Wordle → Reasoning Gym (live measurement) | entropy ratio + lift |
| **9** | 4-method causal counterfactual replay on Tōhoku 2011 | pooled $268B vs anchor $235B |
| **10** | Live data ingest: FRED 8/8 + NewsAPI 5/5 + NOAA 3/3 + WandB | sha256 of every API response |
| **11** | Brent ensemble refit on FRED-real → median <2.5% rel err | 8-event refit chart |
| **12** | 250-feature usage manifest + mosaic plot + HTML certificate | submission certificate.html |

---

### 🔑 Required (for sections that use real APIs)

Set these in Colab `userdata` (left sidebar 🔐) OR paste when prompted in §10:
- `WANDB_API_KEY` (W&B experimental tracking active throughout §3 + §6)
- `FRED_API_KEY` (closes L9 — real Brent for 8 historical events)
- `NEWS_API_KEY` (closes G4 — live Hormuz/Suez/Iran news ingest)
- `NOAA_TOKEN` (real tropical cyclone data)
- `HF_TOKEN` (only needed if you switch to gated LLaMA models — Qwen is ungated)

---

### 🎯 Why Qwen2.5 family (not LLaMA-3)

- **Ungated** — downloads instantly without HF Hub license accept blocker
- **State-of-art** — Qwen2.5-7B beats LLaMA-3.1-8B on most reasoning benchmarks (Oct 2024 release)
- **Unsloth optimized** — full 4-bit + LoRA + safe `merged_16bit` save path supported
- **Three sizes auto-pick** — 7B fits free T4, 14B fits A100, 32B fits H100

The notebook will auto-detect your GPU compute capability and pick the largest Qwen2.5 model that fits. T4 (sm_75) → fp16 + 7B. A100/L4 (sm_80+) → bf16 + 14B/7B. H100 (sm_90) → bf16 + 32B.""")

# Cell 1 — MODE setting (4 modes)
nb['cells'][1]['source'] = lines("""# MODE selection — pick based on your runtime
# 'cpu_quick'  : free Colab CPU, skip GRPO+baseline grid, ~10 min
# 't4_full'    : free Colab T4 (15GB), Qwen2.5-7B GRPO + baseline grid, ~75 min
# 'a100_max'   : Pro Colab A100/L4, Qwen2.5-14B GRPO, ~55 min
# 'h100_beast' : Pro Colab H100, Qwen2.5-32B GRPO, ~50 min (max scale, true SOTA)
MODE = 't4_full'  # change as needed
PUSH_RECEIPTS_TO_REPO = False
WANDB_PROJECT = 'supplymind-master-nb'

assert MODE in ('cpu_quick', 't4_full', 'a100_max', 'h100_beast')
import time as _t
_NOTEBOOK_T0 = _t.time()
print(f'╔═══════════════════════════════════════════════════════════════════╗')
print(f'║  SUPPLYMIND MASTER NOTEBOOK · MODE = {MODE:<22s}        ║')
print(f'║  Started: {_t.strftime("%Y-%m-%dT%H:%M:%SZ", _t.gmtime(_NOTEBOOK_T0)):<27s}                          ║')
print(f'╚═══════════════════════════════════════════════════════════════════╝')""")

# Cell 5 — install + auto-detect bf16 + SOTA model picker (TRUE STATE-OF-ART per tier)
nb['cells'][5]['source'] = lines("""# Pinned dependency install
!pip install -q --upgrade pip 2>&1 | tail -1
!pip install -q torch transformers==4.46.0 accelerate==1.0.1 peft==0.13.2 trl==0.11.4 bitsandbytes==0.44.1 datasets 2>&1 | tail -1
!pip install -q stable-baselines3 sb3-contrib gymnasium scipy matplotlib seaborn httpx pydantic 2>&1 | tail -1
!pip install -q reasoning-gym wandb tqdm 2>&1 | tail -1
if MODE in ('t4_full', 'a100_max', 'h100_beast'):
    !pip install -q 'unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git' 2>&1 | tail -3 || echo 'Unsloth optional'
import torch, numpy as np
GPU = torch.cuda.is_available()
GPU_NAME = torch.cuda.get_device_name(0) if GPU else 'cpu'
GPU_MEM_GB = torch.cuda.get_device_properties(0).total_memory / 1e9 if GPU else 0
# Auto-detect bf16: Ampere+ (sm_80+) supports bf16, Turing T4 (sm_75) does NOT
if GPU:
    cap = torch.cuda.get_device_capability(0)
    SUPPORTS_BF16 = cap[0] >= 8
else:
    SUPPORTS_BF16 = False
USE_BF16 = SUPPORTS_BF16
USE_FP16 = GPU and not SUPPORTS_BF16
DTYPE_LABEL = 'bf16' if USE_BF16 else ('fp16' if USE_FP16 else 'fp32')

# TRUE SOTA model picker — push largest Qwen2.5 that fits the GPU
def pick_model(mode, vram_gb):
    \"\"\"Pick largest Qwen2.5 that fits in 4-bit + GRPO activations.
    Memory budget: model_4bit + 1.3x for GRPO activations + LoRA opt state.
    Qwen2.5 ungated (no HF_TOKEN gating).\"\"\"
    if mode == 'cpu_quick':
        return None
    elif mode == 't4_full':
        # T4 15GB: Qwen2.5-7B 4-bit ~5GB + GRPO act ~6GB + LoRA opt ~1GB = ~12GB SAFE
        return 'Qwen/Qwen2.5-7B-Instruct'
    elif mode == 'a100_max':
        # A100 40GB / L4 24GB: Qwen2.5-14B 4-bit ~10GB + act ~12GB + opt ~2GB = ~24GB
        if vram_gb >= 30: return 'Qwen/Qwen2.5-14B-Instruct'
        else: return 'Qwen/Qwen2.5-7B-Instruct'  # L4 24GB falls back
    else:  # h100_beast
        # H100 80GB: Qwen2.5-32B 4-bit ~20GB + act ~25GB + opt ~5GB = ~50GB SOTA MAX
        if vram_gb >= 70: return 'Qwen/Qwen2.5-32B-Instruct'
        elif vram_gb >= 30: return 'Qwen/Qwen2.5-14B-Instruct'
        else: return 'Qwen/Qwen2.5-7B-Instruct'
LLM_MODEL = pick_model(MODE, GPU_MEM_GB)
print(f'CUDA: {GPU} | GPU: {GPU_NAME} ({GPU_MEM_GB:.1f} GB) | DTYPE: {DTYPE_LABEL}' if GPU else 'CPU mode (DTYPE: fp32)')
print(f'LLM for §6 GRPO: {LLM_MODEL or "(skipped in cpu_quick)"}')
if MODE != 'cpu_quick' and not GPU:
    print('Warning: GPU mode requires CUDA. Falling back to cpu_quick.')
    MODE = 'cpu_quick'; LLM_MODEL = None""")

# Cell 23 — LLaMA load with auto-tuned per model size + ungated Qwen
nb['cells'][23]['source'] = lines("""banner('S6', f'GRPO 200-step on {LLM_MODEL or "skipped"}', expected_metrics='final_loss<10, merged_16bit save OK', features=['B1','D8','AA1'])
if MODE == 'cpu_quick' or LLM_MODEL is None:
    print('SKIPPED on cpu_quick mode'); section_done('S6', 'skipped')
else:
    USE_UNSLOTH = False
    try: from unsloth import FastLanguageModel; USE_UNSLOTH = True
    except ImportError: from transformers import AutoModelForCausalLM, AutoTokenizer
    # Tune max_seq_length per model size to fit memory
    if '7B' in LLM_MODEL: max_sl = 1024
    elif '14B' in LLM_MODEL: max_sl = 768
    else: max_sl = 512  # 32B
    if USE_UNSLOTH:
        model, tokenizer = FastLanguageModel.from_pretrained(LLM_MODEL, max_seq_length=max_sl, dtype=None, load_in_4bit=True)
        # LoRA rank scales inversely with model size (smaller r for bigger model = lower mem)
        if '7B' in LLM_MODEL: lora_r = 16
        elif '14B' in LLM_MODEL: lora_r = 8
        else: lora_r = 4  # 32B
        model = FastLanguageModel.get_peft_model(model, r=lora_r, target_modules=['q_proj','k_proj','v_proj','o_proj'],
            lora_alpha=lora_r*2, lora_dropout=0.05, bias='none', use_gradient_checkpointing='unsloth')
    else:
        from transformers import AutoModelForCausalLM, AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(LLM_MODEL)
        model_dtype = torch.bfloat16 if USE_BF16 else (torch.float16 if USE_FP16 else torch.float32)
        model = AutoModelForCausalLM.from_pretrained(LLM_MODEL, torch_dtype=model_dtype, device_map='auto')
    if tokenizer.pad_token is None: tokenizer.pad_token = tokenizer.eos_token
    print(f'Model: {LLM_MODEL} (Unsloth={USE_UNSLOTH}, dtype={DTYPE_LABEL}, max_seq={max_sl})')""")

# Cell 24 — GRPO config tuned aggressive per size
nb['cells'][24]['source'] = lines(r"""if MODE != 'cpu_quick' and LLM_MODEL is not None:
    from trl import GRPOConfig, GRPOTrainer
    from datasets import Dataset
    def grpo_reward(prompts, completions, **kw):
        rs = []
        for p, c in zip(prompts, completions):
            m = re.search(r'\b([a-zA-Z]{5})\b', c)
            if not m: rs.append(-0.20); continue
            g = m.group(1).lower()
            if g not in WORD_SET: rs.append(-1.00); continue
            tm = re.search(r'TARGET:\s*(\w{5})', p)
            if not tm: rs.append(0.0); continue
            t = tm.group(1).lower(); fb = score_guess(g, t)
            ng = sum(1 for f in fb if f=='green'); ny = sum(1 for f in fb if f=='yellow')
            r = 0.05*ng + 0.02*ny
            if g==t: r += 1.0
            rs.append(float(r))
        return rs
    train_ds = Dataset.from_list([{'prompt':f'You are playing Wordle. Output a single 5-letter word guess.\nTARGET: {random.choice(WORD_LIST[:50])}\nGUESS: '} for _ in range(200)])
    # Aggressive batch+grad-accum auto-tune per model size for max throughput
    if '7B' in LLM_MODEL:   bs, ga, ng, comp_len = 1, 4, 4, 32   # T4: tight but max throughput
    elif '14B' in LLM_MODEL: bs, ga, ng, comp_len = 1, 8, 4, 32  # A100: more headroom
    else:                    bs, ga, ng, comp_len = 1, 16, 2, 24 # 32B H100: aggressive grad accum
    cfg = GRPOConfig(output_dir='/content/wordle-grpo', max_steps=200,
        per_device_train_batch_size=bs, num_generations=ng, gradient_accumulation_steps=ga,
        learning_rate=1e-5, bf16=USE_BF16, fp16=USE_FP16,
        max_prompt_length=128, max_completion_length=comp_len, logging_steps=10, save_steps=200,
        report_to='wandb' if os.environ.get('WANDB_API_KEY') else 'none',
        remove_unused_columns=False)
    trainer = GRPOTrainer(model=model, args=cfg, train_dataset=train_ds, processing_class=tokenizer, reward_funcs=[grpo_reward])
    t0 = time.time(); res = trainer.train(); n6_t = time.time()-t0
    print(f'\n[OK] GRPO complete: {n6_t:.0f}s, final_loss={res.metrics.get("train_loss", "?")}')
    out, _ = write_receipt('nb13_S6_grpo.json', {
        'name':'grpo_master_nb', 'model':LLM_MODEL, 'unsloth':USE_UNSLOTH,
        'mode':MODE, 'config':{'max_steps':200,'num_gen':ng,'batch':bs,'grad_accum':ga,'lr':1e-5,'dtype':DTYPE_LABEL,'max_completion_len':comp_len},
        'wall_clock_s':round(n6_t,1), 'final_loss':float(res.metrics.get('train_loss', 0)),
        'gpu':GPU_NAME, 'gpu_mem_gb':round(GPU_MEM_GB,1)}, features=['B1','D8'])
    feat('B1','D8')
    if USE_UNSLOTH:
        model.save_pretrained_merged('/content/wordle-grpo-merged-16bit', tokenizer, save_method='merged_16bit')
        from transformers import AutoModelForCausalLM
        merged_dtype = torch.bfloat16 if USE_BF16 else (torch.float16 if USE_FP16 else torch.float32)
        rm = AutoModelForCausalLM.from_pretrained('/content/wordle-grpo-merged-16bit', torch_dtype=merged_dtype, device_map='auto')
        ti = tokenizer('Hello world', return_tensors='pt').to(rm.device)
        to = rm.generate(**ti, max_new_tokens=20, do_sample=False, pad_token_id=tokenizer.eos_token_id)
        print('Post-merge inference:', tokenizer.decode(to[0], skip_special_tokens=True)[:120])
        feat('AA1')
    section_done('S6', f'{n6_t:.0f}s GRPO on {LLM_MODEL.split("/")[-1]}')""")

PATH.write_text(json.dumps(nb, indent=1, ensure_ascii=False), encoding='utf-8')
print(f'Patched {PATH}')
print('SOTA push:')
print('  cpu_quick   -> no LLM (Wordle REINFORCE only)')
print('  t4_full     -> Qwen2.5-7B-Instruct (4-bit Unsloth, max_sl=1024, lora_r=16)')
print('  a100_max    -> Qwen2.5-14B-Instruct (4-bit Unsloth, max_sl=768, lora_r=8)')
print('  h100_beast  -> Qwen2.5-32B-Instruct (4-bit Unsloth, max_sl=512, lora_r=4)')
print('Auto-pick + auto-tune. W&B reports activated when WANDB_API_KEY present.')
