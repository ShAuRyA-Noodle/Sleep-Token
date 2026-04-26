"""Patch nb13 for auto model size + ungated fallback."""
import json
from pathlib import Path

PATH = Path('notebooks/13_MASTER_HACKATHON_FINAL.ipynb')
nb = json.loads(PATH.read_text(encoding='utf-8'))


def lines(s):
    return [l + '\n' for l in s.splitlines()]


# Cell 1 — MODE setting + auto detection
nb['cells'][1]['source'] = lines("""# MODE selection — pick based on your runtime
# 'cpu_quick'  : free Colab CPU, skip GRPO+baseline grid, ~10 min
# 't4_full'    : free Colab T4 (15GB), Qwen2.5-1.5B GRPO + baseline grid, ~70 min
# 'a100_max'   : Pro Colab A100/L4, Qwen2.5-7B GRPO, ~50 min
# 'h100_beast' : Pro Colab H100, Qwen2.5-14B GRPO, ~45 min (max scale)
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

# Cell 5 — install + bf16 auto + MODEL_NAME auto-pick
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

# Auto-pick model based on MODE + actual VRAM (Qwen2.5 is ungated, no HF_TOKEN needed)
def pick_model(mode, vram_gb):
    if mode == 'cpu_quick':
        return None
    elif mode == 't4_full' or vram_gb < 20:
        return 'Qwen/Qwen2.5-1.5B-Instruct'   # fits T4 15GB clean
    elif mode == 'a100_max' or vram_gb < 60:
        return 'Qwen/Qwen2.5-7B-Instruct'     # fits A100 40GB / L4 24GB
    else:  # h100_beast
        return 'Qwen/Qwen2.5-14B-Instruct'    # H100 80GB max scale
LLM_MODEL = pick_model(MODE, GPU_MEM_GB)
print(f'CUDA: {GPU} | GPU: {GPU_NAME} ({GPU_MEM_GB:.1f} GB) | DTYPE: {DTYPE_LABEL}' if GPU else 'CPU mode (DTYPE: fp32)')
print(f'LLM for §6 GRPO: {LLM_MODEL or "(skipped in cpu_quick)"}')
if MODE != 'cpu_quick' and not GPU:
    print('Warning: GPU mode requires CUDA. Falling back to cpu_quick.')
    MODE = 'cpu_quick'; LLM_MODEL = None""")

# Cell 23 — LLaMA load → use auto-picked model
nb['cells'][23]['source'] = lines("""banner('S6', f'GRPO 200-step on {LLM_MODEL or "skipped"}', expected_metrics='final_loss<10, merged_16bit save OK', features=['B1','D8','AA1'])
if MODE == 'cpu_quick' or LLM_MODEL is None:
    print('SKIPPED on cpu_quick mode'); section_done('S6', 'skipped')
else:
    USE_UNSLOTH = False
    try: from unsloth import FastLanguageModel; USE_UNSLOTH = True
    except ImportError: from transformers import AutoModelForCausalLM, AutoTokenizer
    if USE_UNSLOTH:
        # Set max_seq_length lower for bigger models to fit memory
        max_sl = 2048 if '1.5B' in LLM_MODEL else (1024 if '7B' in LLM_MODEL else 768)
        model, tokenizer = FastLanguageModel.from_pretrained(LLM_MODEL, max_seq_length=max_sl, dtype=None, load_in_4bit=True)
        # LoRA rank: smaller for bigger models
        lora_r = 16 if '1.5B' in LLM_MODEL else (8 if '7B' in LLM_MODEL else 4)
        model = FastLanguageModel.get_peft_model(model, r=lora_r, target_modules=['q_proj','k_proj','v_proj','o_proj'],
            lora_alpha=lora_r*2, lora_dropout=0.05, bias='none', use_gradient_checkpointing='unsloth')
    else:
        from transformers import AutoModelForCausalLM, AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(LLM_MODEL)
        model_dtype = torch.bfloat16 if USE_BF16 else (torch.float16 if USE_FP16 else torch.float32)
        model = AutoModelForCausalLM.from_pretrained(LLM_MODEL, torch_dtype=model_dtype, device_map='auto')
    if tokenizer.pad_token is None: tokenizer.pad_token = tokenizer.eos_token
    print(f'Model: {LLM_MODEL} (Unsloth={USE_UNSLOTH}, dtype={DTYPE_LABEL})')""")

# Cell 24 — GRPO config with batch size auto-pick
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
    # Batch size + grad accum tuned per model size
    if '1.5B' in LLM_MODEL: bs, ga, ng = 2, 2, 4
    elif '7B' in LLM_MODEL: bs, ga, ng = 1, 4, 4
    else: bs, ga, ng = 1, 8, 2  # 14B
    cfg = GRPOConfig(output_dir='/content/wordle-grpo', max_steps=200,
        per_device_train_batch_size=bs, num_generations=ng, gradient_accumulation_steps=ga,
        learning_rate=1e-5, bf16=USE_BF16, fp16=USE_FP16,
        max_prompt_length=128, max_completion_length=32, logging_steps=10, save_steps=200,
        report_to='none', remove_unused_columns=False)
    trainer = GRPOTrainer(model=model, args=cfg, train_dataset=train_ds, processing_class=tokenizer, reward_funcs=[grpo_reward])
    t0 = time.time(); res = trainer.train(); n6_t = time.time()-t0
    print(f'\n[OK] GRPO complete: {n6_t:.0f}s, final_loss={res.metrics.get("train_loss", "?")}')
    out, _ = write_receipt('nb13_S6_llama_grpo.json', {
        'name':'grpo_master_nb', 'model':LLM_MODEL, 'unsloth':USE_UNSLOTH,
        'mode':MODE, 'config':{'max_steps':200,'num_gen':ng,'batch':bs,'grad_accum':ga,'lr':1e-5,'dtype':DTYPE_LABEL},
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

# Cell 26 — baseline grid skip on cpu_quick only (run on all GPU modes)
nb['cells'][26]['source'] = lines("""banner('S7', 'Baseline grid (DQN/QRDQN/TRPO/A2C)', expected_metrics='4/4 algos converge', features=['D5','D15','D16','D17'])
if MODE == 'cpu_quick': print('SKIPPED on cpu_quick mode'); section_done('S7', 'skipped')
else:
    import gymnasium as gym
    from gymnasium import spaces
    from server.supply_environment import SupplyMindEnvironment
    from models import SupplyMindAction
    from stable_baselines3 import DQN, A2C
    from sb3_contrib import QRDQN, TRPO
    from stable_baselines3.common.vec_env import DummyVecEnv
    class SMGym(gym.Env):
        def __init__(self, task='hard_cascading_crisis', seed=42):
            self.env = SupplyMindEnvironment(); self.task = task; self.seed_v = seed
            self.observation_space = spaces.Box(low=-1, high=1, shape=(64,), dtype=np.float32)
            self.action_space = spaces.Discrete(280)
            self.atypes = ['do_nothing','activate_backup_supplier','reroute_shipment','increase_safety_stock','expedite_order','hedge_commodity','issue_supplier_alert']
            self.targets = ['SUP_TSMC','SUP_SAMSUNG','SUP_FOXCONN','SUP_INTEL','SUP_TOYOTA']*8
        def _f(self, obs):
            f = np.zeros(64, dtype=np.float32)
            fin = obs.get('financials', {}) if isinstance(obs, dict) else {}
            f[0] = (fin.get('budget_remaining', 0)-5e6)/5e6
            f[1] = fin.get('cumulative_revenue_lost', 0)/1e8
            f[2] = fin.get('supply_chain_health_score', 50)/100
            return f
        def reset(self, seed=None, options=None):
            super().reset(seed=seed); obs = self.env.reset(task_id=self.task, seed=seed or self.seed_v)
            return self._f(obs), {}
        def step(self, action):
            ai = action % 7; ti = (action // 7) % 40
            sm = SupplyMindAction(action_type=self.atypes[ai], target_node_id=self.targets[ti],
                backup_supplier_id='SUP_SAMSUNG' if self.atypes[ai]=='activate_backup_supplier' else None,
                reroute_via=['PORT_KAOHSIUNG'] if self.atypes[ai]=='reroute_shipment' else None,
                additional_stock_days=7 if self.atypes[ai]=='increase_safety_stock' else None,
                expedite_mode='air' if self.atypes[ai]=='expedite_order' else None,
                commodity='oil' if self.atypes[ai]=='hedge_commodity' else None,
                hedge_amount_usd=100000 if self.atypes[ai]=='hedge_commodity' else None)
            obs, reward, done, info = self.env.step(sm)
            return self._f(obs), float(reward), bool(done), False, info
    # More timesteps on bigger GPUs
    TS = 5000 if MODE == 't4_full' else (10000 if MODE == 'a100_max' else 15000)
    results = {}
    for an, AC, kw in [('DQN', DQN, {}), ('QRDQN', QRDQN, {}), ('TRPO', TRPO, {}), ('A2C', A2C, {})]:
        print(f'\\nTraining {an} for {TS} timesteps...')
        ev = DummyVecEnv([lambda: SMGym('hard_cascading_crisis')])
        m = AC('MlpPolicy', ev, verbose=0, **kw)
        t0 = time.time(); m.learn(total_timesteps=TS); tt = time.time()-t0
        ee = SMGym('hard_cascading_crisis'); rs = []
        for ep in range(10):
            obs, _ = ee.reset(seed=80000+ep); epr = 0; done = False
            while not done:
                a, _ = m.predict(obs, deterministic=True)
                obs, r, done, _, _ = ee.step(a); epr += r
            rs.append(epr)
        results[an] = {'train_s':round(tt,2),'eval_mean':float(np.mean(rs)),'eval_std':float(np.std(rs))}
        print(f'  {an}: train={tt:.1f}s eval_mean={np.mean(rs):+.3f}')
    out, _ = write_receipt('nb13_S7_baseline_grid.json', {'name':'baseline_grid_master_nb','algos':results,
        'mode':MODE, 'timesteps_per_algo':TS, 'closes':'D15-D17 + L6'}, features=['D5','D15','D16','D17'])
    feat('D5','D15','D16','D17')
    section_done('S7', f'{len(results)} algos x {TS} timesteps')""")

PATH.write_text(json.dumps(nb, indent=1, ensure_ascii=False), encoding='utf-8')
print(f'Patched {PATH}')
print('Final fixes:')
print('  - 4-MODE selector: cpu_quick / t4_full / a100_max / h100_beast')
print('  - Auto model pick: Qwen2.5-1.5B (T4) / 7B (A100/L4) / 14B (H100)')
print('  - Ungated Qwen2.5 family (no HF_TOKEN gating)')
print('  - GRPO batch+grad-accum auto-tuned per model size')
print('  - Baseline grid timesteps scale with mode (5K/10K/15K)')
