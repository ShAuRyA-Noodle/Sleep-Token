# Hackathon Submission — Theme & Criteria Mapping

## Theme selection: **#3 Professional Tasks** (primary) + **#2 Long-Horizon Planning** (secondary)

### Why Theme #3 fits perfectly

The hackathon brief defines Theme #3 as: *"environment that simulates a REAL professional workflow — where the AI must use actual tools, call APIs, interact with real-ish systems, and do genuine work. The key rule here is: no shortcuts allowed."*

SupplyMind is exactly this:

| Brief requirement | SupplyMind delivery |
|---|---|
| "Real professional workflow" | Supply-chain risk manager — a $184B/year actual job category |
| "Use actual tools" | 7 discrete actions = real industry tools (backup activation, oil hedging, vessel rerouting, expediting via air/rail/sea, safety stock, supplier alerts) |
| "Call APIs" | 5 live data sources (NewsAPI, GDELT, USGS, FRED, MarineTraffic) wired into env state |
| "Interact with real-ish systems" | Live HF Space deployment, FastAPI endpoints, MCP JSON-RPC, WebSocket |
| "Do genuine work" | Counterfactual computes real ₹/$ savings against historical analogs |
| "No shortcuts" | Programmatic verifier (env.grade()) — cannot fake. Anti-reward-hacking suite tests 6 attacks, all rejected |

### Why Theme #2 also fits (mention as secondary)

| Brief requirement | SupplyMind delivery |
|---|---|
| "Many steps over a LONG period" | Hard task = 60 days × 200 max steps |
| "Reward/feedback comes very late" | Cumulative_revenue_lost only stabilizes at episode end |
| "Early mistake can ruin everything" | Spending budget too early on wrong action → no resources for actual disaster |
| "Beyond context memory limits" | 60-step episode × 1500-token observation each = exceeds typical LLM context |
| "300 scattered instructions" | Action recommender produces 5-action plan; agent must execute over time |

**Strategy**: Lead with Theme #3, mention Theme #2 in 1 sentence. Don't dilute focus.

---

## Stack alignment — every required tool used

| Required tool | Our usage | File |
|---|---|---|
| **OpenEnv (latest)** | `Environment` base class, valid `openenv.yaml`, 19 formal compliance tests | [`../server/`](../server/), [`../openenv.yaml`](../openenv.yaml) |
| **TRL** | `GRPOTrainer` for LLM agent training | [`training/train_grpo_supplymind.py`](training/train_grpo_supplymind.py) |
| **GRPO** | The training algorithm — group-relative policy optimization, no critic, multi-completion sampling | [`training/train_grpo_supplymind.py`](training/train_grpo_supplymind.py) |
| **Unsloth** | `FastLanguageModel` 4-bit NF4 quantization, 2× speed on T4 | [`training/train_grpo_supplymind.py`](training/train_grpo_supplymind.py) |
| **HuggingFace Spaces** | Live deployment at `huggingface.co/spaces/Shaurya-Noodle/Supplymind` | [`deploy/HF_SPACE_DEPLOY.md`](deploy/HF_SPACE_DEPLOY.md) |

---

## Criteria mapping (40/30/20/10)

### Criterion 1 — Environment Innovation (40%)

> *"Does this environment exist to teach an AI something it currently CANNOT do well? Is this domain underexplored? Could a researcher write a paper about training on this?"*

**Innovation evidence**:
1. **Real-data calibration depth** — 261,175 real points across 8 sources. No prior OpenEnv submission has this.
2. **Live geopolitical ingestion** — NewsAPI + GDELT + USGS + FRED + MarineTraffic streaming into env state. Novel for OpenEnv.
3. **Anti-reward-hacking adversarial suite** — 6 attack vectors, layered defenses, receipt at `tests/receipts/adversarial_reward_audit.json`. Rare even in research.
4. **12-frontier LLM judge panel** — 12 OpenRouter models from 5 labs grading scenarios. Krippendorff α published. Total cost ₹3.
5. **Counterfactual digital twin** — 100-rollout MC with paired bootstrap CI95 [$177.74M, $179.52M].
6. **Custom 50-line GCN** — pure PyTorch, beats MLP by −64% MAE on hard graph.
7. **Per-horizon split-conformal RL** — textbook Foygel Barber 2022 for safe abstention.
8. **Karpathy autoresearch loop** — autonomous ML research overnight, discovered curriculum learning.
9. **3 calibrated difficulty levels** — 12/25/40 nodes, real cost constants, real disruption curves.
10. **Domain depth** — supply-chain risk has zero prior coverage in OpenEnv ecosystem.

**Paper-worthiness**: Yes. The combination of real-data calibration + LLM agent training + counterfactual evaluation has 4 paper-quality contributions: (a) the env design, (b) the conformal RL safety layer, (c) the autoresearch curriculum discovery, (d) the 15-judge panel methodology.

### Criterion 2 — Storytelling & Presentation (30%)

> *"Can a non-technical person understand what you built, why it matters, and what the AI learned? 3-5 minute README. WANT TO TRY your environment."*

**Storytelling assets**:
- **README.md** — designed for 3-5 minute read, 30-second pitch up top, headline numbers table, 4-minute judge path
- **2-minute video** — `demo/VIDEO_SCRIPT.md` with 8-scene storyboard
- **HF blog post** — `demo/BLOG_POST.md`, formatted for HuggingFace blog
- **Hormuz showstopper** — real news → real 0.99 match → real ₹2,160 crore savings, on stage in 90 seconds
- **One-line summary** — *"Reads today's news, finds historical analogs, computes savings — every number verifiable in 30 seconds"*

**The "want to try" hook**: 4 reproducibility paths — (1) HF Space `/live/hormuz-closure`, (2) Colab notebook one-click train, (3) any of 35 receipts re-run, (4) full repo `pytest` (250 tests in 2m38s).

### Criterion 3 — Showing Improvement in Rewards (20%)

> *"Reward plots over training time. Loss plots. Baseline vs trained agent. Numbers showing improvement."*

**Evidence**:
- [`results/reward_curve.png`](results/reward_curve.png) — GRPO training reward over 200 steps, axes labeled, x=steps y=mean_reward
- [`results/loss_curve.png`](results/loss_curve.png) — GRPO loss over training
- [`results/baseline_vs_trained.png`](results/baseline_vs_trained.png) — Untrained Qwen-2.5-0.5B vs GRPO-trained, **on same axes**
- [`results/training_log.csv`](results/training_log.csv) — raw numbers, judge-replayable
- **Plus the prior 10,800-episode bootstrap benchmark** ([`v3_arcadia/results/R6_EUCLIDIAN.json`](../v3_arcadia/results/R6_EUCLIDIAN.json)) — non-overlapping CI95 across 4 policies × 3 tasks
- **Plus per-action-type ablation** — masking +26.8%, GNN +48-64%, conformal dev=0.024

**Plot rules followed**:
✅ Both axes labeled (x = step / episode, y = reward / loss)
✅ PNG committed to repo (not just in notebook)
✅ Embedded in README with caption
✅ Multi-run on same axes (baseline_vs_trained.png)

### Criterion 4 — Reward & Training Pipeline (10%)

> *"Is your reward logic sensible? Does your pipeline actually produce improvement?"*

**Reward design**:
- **7-component dense reward** in `[-1, 1]`:
  1. Revenue preservation (35%) — main signal
  2. Stockout prevention (25%) — event-driven
  3. Proactive bonus (15%) — time-discounted, encourages early action
  4. Cost penalty (10%) — discourages waste
  5. Health maintenance (5%) — supply chain health score delta
  6. SLA compliance (5%) — customer delay reduction
  7. Unnecessary action (5%) — penalizes no-effect actions
- **Programmatic verifier first** — `env.grade()` returns deterministic 0-1 score (5× same-seed test passes)
- **Anti-hacking** — 6 adversarial attacks tested in `tests/test_reward_hacking_adversarial.py`, all rejected
- **Curriculum learning** — easy → medium → hard, autoresearch-validated +0.0967 CI95 lift
- **Lagrangian budget constraint** — mathematically guaranteed budget adherence in `rl/constrained_ppo.py`

**Training pipeline**:
- GRPO via TRL `GRPOTrainer`
- Multi-completion sampling (num_generations=4)
- Reward via env.step() — programmatic, not LLM-judged
- Single-step formulation for fast iteration
- LoRA r=16, 4-bit NF4 via Unsloth
- T4-friendly (Colab free tier compatible)

---

## Engineering rules — all followed

✅ Uses `Environment` (or `MCPEnvironment`) base class
✅ Client/server separation (client/ never imports server/)
✅ Gym-style API: `reset()`, `step()`, `state()` all work
✅ Valid `openenv.yaml` manifest
✅ NEVER uses reserved tool names (no user actions named `reset`, `step`, `state`, `close`)
✅ Pydantic v2 typed contracts
✅ FastAPI server with MCP + WebSocket
✅ Deterministic grader (5× verified)
✅ Action validation (graceful failure on bad input)
✅ Episode termination guaranteed within max_steps

---

## What makes this submission unwinnable to copy

A competing team would need to replicate, in 36 hours:

1. 261,175 data points from 8 public sources, cleaned and calibrated
2. 13 SOTA foundation models running locally with custom Modelfiles
3. 5 versions of fine-tuned Ollama analyst models
4. Custom GCN, conformal prediction, MaskablePPO with action mask wrapper
5. 5 live data ingestion adapters with SQLite event store and SHA-256 dedup
6. 8-event crisis library with ≥3 citations each
7. 10,800-episode bootstrap benchmark with statistical machinery
8. 6 anti-reward-hacking adversarial tests
9. 15-judge consensus panel (12 frontier + 3 local) with Krippendorff α
10. Counterfactual digital twin with paired bootstrap CI95
11. Karpathy autoresearch loop with 5 experiments executed
12. 250 tests passing in 2m38s
13. 35 reproducibility receipts with SHA-256 stdout tracking

Most teams will have one or two of these. We have all 13.

---

## Submission deliverables (final list)

1. ✅ HF Space URL — `huggingface.co/spaces/Shaurya-Noodle/Supplymind`
2. ✅ Colab notebook — `Final_Submit/training/colab_train_grpo.ipynb`
3. ✅ Reward + loss + baseline-vs-trained PNGs — `Final_Submit/results/`
4. ✅ 2-minute video URL — `Final_Submit/demo/VIDEO_URL.txt`
5. ✅ HF blog post — `Final_Submit/demo/BLOG_POST.md`
6. ✅ README — this folder
7. ✅ 35 receipts — `ShAuRyA_Supplymind/receipts/` + `ShAuRyA_Phoenix/receipts_v2/`
8. ✅ openenv.yaml — `../openenv.yaml`
9. ✅ Two upstream PRs ready — `ShAuRyA_Phoenix/upstream_prs/{meta_openenv,alibaba_roll}/`

Ready to submit.
