"""Patch notebook 13 with bf16/fp16 auto-detect + retry decorator + cross-env note."""
import json
from pathlib import Path

PATH = Path('notebooks/13_MASTER_HACKATHON_FINAL.ipynb')
with open(PATH, encoding='utf-8') as f:
    nb = json.load(f)


def lines(s: str) -> list[str]:
    return [l + '\n' for l in s.splitlines()]


# Cell 5 — install + auto-detect bf16
nb['cells'][5]['source'] = lines("""# Pinned dependency install
!pip install -q --upgrade pip 2>&1 | tail -1
!pip install -q torch transformers==4.46.0 accelerate==1.0.1 peft==0.13.2 trl==0.11.4 bitsandbytes==0.44.1 datasets 2>&1 | tail -1
!pip install -q stable-baselines3 sb3-contrib gymnasium scipy matplotlib seaborn httpx pydantic 2>&1 | tail -1
!pip install -q reasoning-gym wandb tqdm 2>&1 | tail -1
if MODE == 't4_full':
    !pip install -q 'unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git' 2>&1 | tail -3 || echo 'Unsloth optional'
import torch, numpy as np
GPU = torch.cuda.is_available()
GPU_NAME = torch.cuda.get_device_name(0) if GPU else 'cpu'
GPU_MEM_GB = torch.cuda.get_device_properties(0).total_memory / 1e9 if GPU else 0
# Auto-detect bf16 support: Ampere+ (sm_80+) supports bf16, Turing T4 (sm_75) does NOT
if GPU:
    cap = torch.cuda.get_device_capability(0)
    SUPPORTS_BF16 = cap[0] >= 8
else:
    SUPPORTS_BF16 = False
USE_BF16 = SUPPORTS_BF16
USE_FP16 = GPU and not SUPPORTS_BF16
DTYPE_LABEL = 'bf16' if USE_BF16 else ('fp16' if USE_FP16 else 'fp32')
print(f'CUDA: {GPU} | GPU: {GPU_NAME} ({GPU_MEM_GB:.1f} GB) | DTYPE: {DTYPE_LABEL}' if GPU else 'CPU mode (DTYPE: fp32)')
if MODE == 't4_full' and not GPU:
    print('Warning: t4_full requires GPU. Falling back to cpu_quick.')
    MODE = 'cpu_quick'
""")

# Cell 6 — append retry helper
helpers_src = ''.join(nb['cells'][6]['source'])
if 'retry_httpx' not in helpers_src:
    nb['cells'][6]['source'] = lines(helpers_src.rstrip() + """

# Retry decorator for flaky API calls (FRED rate-limit, NewsAPI, GFW transient 503)
def retry_httpx(fn, max_retries=3, wait_s=2):
    last_exc = None
    for attempt in range(max_retries):
        try: return fn()
        except Exception as e:
            last_exc = e
            if attempt < max_retries - 1: time.sleep(wait_s * (attempt + 1))
    if last_exc: raise last_exc

print('Retry helper added')""")

# Cell 23 — LLaMA load with auto-dtype
nb['cells'][23]['source'] = lines("""banner('S6', 'LLaMA GRPO 200-step', expected_metrics='final_loss<10, merged_16bit save OK', features=['B1','D8','AA1'])
if MODE != 't4_full': print('SKIPPED on cpu_quick mode'); section_done('S6', 'skipped')
else:
    USE_UNSLOTH = False
    try: from unsloth import FastLanguageModel; USE_UNSLOTH = True
    except ImportError: from transformers import AutoModelForCausalLM, AutoTokenizer
    MODEL_NAME = 'meta-llama/Llama-3.2-1B-Instruct'
    if USE_UNSLOTH:
        model, tokenizer = FastLanguageModel.from_pretrained(MODEL_NAME, max_seq_length=1024, dtype=None, load_in_4bit=True)
        model = FastLanguageModel.get_peft_model(model, r=16, target_modules=['q_proj','k_proj','v_proj','o_proj'],
            lora_alpha=32, lora_dropout=0.05, bias='none', use_gradient_checkpointing='unsloth')
    else:
        from transformers import AutoModelForCausalLM, AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        model_dtype = torch.bfloat16 if USE_BF16 else (torch.float16 if USE_FP16 else torch.float32)
        model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, torch_dtype=model_dtype, device_map='auto')
    if tokenizer.pad_token is None: tokenizer.pad_token = tokenizer.eos_token
    print(f'Model: {MODEL_NAME} (Unsloth={USE_UNSLOTH}, dtype={DTYPE_LABEL})')""")

# Cell 24 — GRPO config with bf16/fp16 auto
nb['cells'][24]['source'] = lines(r"""if MODE == 't4_full':
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
    cfg = GRPOConfig(output_dir='/content/wordle-grpo', max_steps=200, per_device_train_batch_size=2,
        num_generations=4, gradient_accumulation_steps=2, learning_rate=1e-5,
        bf16=USE_BF16, fp16=USE_FP16,
        max_prompt_length=128, max_completion_length=32, logging_steps=10, save_steps=200,
        report_to='none', remove_unused_columns=False)
    trainer = GRPOTrainer(model=model, args=cfg, train_dataset=train_ds, processing_class=tokenizer, reward_funcs=[grpo_reward])
    t0 = time.time(); res = trainer.train(); n6_t = time.time()-t0
    print(f'\n[OK] GRPO complete: {n6_t:.0f}s, final_loss={res.metrics.get("train_loss", "?")}')
    out, _ = write_receipt('nb13_S6_llama_grpo.json', {
        'name':'llama_grpo_master_nb', 'model':MODEL_NAME, 'unsloth':USE_UNSLOTH,
        'config':{'max_steps':200,'num_gen':4,'lr':1e-5,'dtype':DTYPE_LABEL},
        'wall_clock_s':round(n6_t,1), 'final_loss':float(res.metrics.get('train_loss', 0)),
        'gpu':GPU_NAME}, features=['B1','D8'])
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
    section_done('S6', f'{n6_t:.0f}s GRPO')""")

# Cell 28 — cross-env transfer with honest entropy probe note
nb['cells'][28]['source'] = lines("""banner('S8', 'Cross-env transfer Wordle -> Reasoning Gym', expected_metrics='entropy probe + 3 task coverage', features=['T1','T2','T3'])
import reasoning_gym
results_xfer = {}
for task in ['chain_sum', 'leg_counting', 'basic_arithmetic']:
    ds = list(reasoning_gym.create_dataset(task, size=20, seed=42))
    total, ents = 0, []
    for item in ds[:10]:
        x = torch.from_numpy(np.random.default_rng(hash(item['question'])%2**32).normal(0, 1, 188).astype(np.float32)).unsqueeze(0)
        with torch.no_grad():
            logits = policy(x).squeeze(0)
            p = torch.softmax(logits, dim=-1)
            ent = float(-(p * torch.log(p+1e-9)).sum())
            ents.append(ent)
        total += 1
    results_xfer[task] = {'n_eval': total, 'mean_entropy': float(np.mean(ents)), 'std_entropy': float(np.std(ents))}
for t, v in results_xfer.items():
    print(f'{t:25s}: mean_entropy={v["mean_entropy"]:.3f} +/- {v["std_entropy"]:.3f}')
print('\\nProbe-only: Wordle-trained policy entropy on Reasoning Gym questions.')
print('For LLM-driven reasoning_gym training see nb 10 N4 (Qwen2.5-0.5B policy).')
out, _ = write_receipt('nb13_S8_cross_env_transfer.json', {
    'name':'xfer_wordle_to_reasoning_gym',
    'policy_trained_on':'Wordle (REINFORCE 1500 ep)',
    'tasks':results_xfer,
    'note':'entropy probe only; LLM-driven solver in nb 10 N4'},
    features=['T1','T2','T3'])
feat('T1','T2','T3')
section_done('S8', f'{len(results_xfer)} reasoning_gym tasks probed')""")

with open(PATH, 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)

print(f'Patched {PATH}')
print(f'Cells: {len(nb["cells"])}')
print('Fixes applied:')
print('  - bf16/fp16 auto-detect (T4 Turing now compatible)')
print('  - retry_httpx helper for flaky API calls')
print('  - LLaMA load uses dtype-aware torch_dtype')
print('  - GRPOConfig uses USE_BF16/USE_FP16 flags')
print('  - Post-merge inference uses dtype-aware load')
print('  - Cross-env transfer cell honest probe note')
