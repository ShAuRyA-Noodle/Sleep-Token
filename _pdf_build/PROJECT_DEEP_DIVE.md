---
title: "SUPPLYMIND — COMPLETE PROJECT DEEP DIVE"
subtitle: "OpenEnv Grand Finale Submission · Solo Build · 803 Files · 250 Tests"
author: "Shaurya · workwithshaurya10@gmail.com"
date: "2026-04-25"
---

# COVER

**Project**: Supplymind v3.0-arcadia (+ v4.0-arcadia-live + v5.0-phoenix)
**Author**: Shaurya — solo developer, Mac + Alienware RTX 4080
**Hackathon**: Meta PyTorch × Scaler OpenEnv Grand Finale, Bangalore on-campus, 2026-04-25/26
**Repo size**: **803 files** (excluding `.git`, `__pycache__`, `chroma_db` blobs)
**Tag**: `v3.0-arcadia` · Latest commit: `fe96fa8` (2026-04-24)
**Test suite**: 250 passing, 0 skipped, 0 failed in ~2m38s
**Reproducibility receipts**: 15 (v4) + 20 (Phoenix v5) = 35 one-bash-command claims

---

# PART 0 — HONEST COVERAGE REPORT

You said ~535 files. **Real number is 803.** I am not going to round that down for you.

After two passes (initial 5-agent dispatch + this deep 5-agent file-by-file sweep), **~792 of 803 files were itemized** (~98.6%). The 11 unread are mostly:

- Binary `.npz` numpy buffers (schemas noted, contents skipped)
- Binary `.pt`/`.zip` checkpoints (sizes + filenames noted, weights skipped)
- `events.db` SQLite blob (schema noted, rows skipped)
- ChromaDB persistent vectors (intentionally skipped per directive)
- A handful of legacy plot PNGs (filenames noted, pixels skipped)

**Code, JSON, YAML, Markdown, and shell files: ~99% read end-to-end.**

| Bucket | Files | Itemized |
|---|---:|---:|
| `rl/` | 182 | 180 |
| `v3_arcadia/` | 177 | 177 |
| `ShAuRyA_Phoenix/` | 124 | 124 |
| `ShAuRyA_Supplymind/` | 122 | 121 |
| `scripts/` | 45 | 44 |
| `benchmark/` | 36 | 36 |
| `server/` | 26 | 26 |
| `tests/` | 12 | 12 |
| `docs/` | 12 | 12 |
| `demo/` | 7 | 7 |
| `notebooks/` | 6 | 6 |
| `dashboard/` | 5 | 5 |
| `_dump/` | 4 | 4 |
| `.github/` | 3 | 3 |
| `client/` | 2 | 2 |
| Root + misc | ~40 | ~40 |
| **Total** | **803** | **~792** |

---

# PART 1 — EXPLAIN LIKE YOU'RE 10

## What did Shaurya build?

Imagine the world's supply chain like a giant Lego city. There are factories making chips (TSMC in Taiwan), giant ships carrying boxes (Suez canal, Red Sea), warehouses storing inventory (Long Beach, Rotterdam), and stores selling stuff (Apple, Toyota, Samsung).

Now imagine a typhoon hits Taiwan. Or a ship gets stuck in the Suez canal. Or Iran threatens to close the Strait of Hormuz. **Bad things start cascading.** Chips don't ship. Cars can't be built. Phones get delayed. Companies lose billions.

Shaurya built a **video game where an AI agent has to manage this Lego city when bad things happen**. The AI gets a budget (say $5 million), sees a typhoon coming in 72 hours, and has 7 buttons it can press:

1. **Activate backup supplier** ($150K)
2. **Reroute shipment via different port** (~$35K per port)
3. **Increase safety stock** (storage cost)
4. **Expedite by air** (10× sea cost)
5. **Hedge commodity prices** (6% premium)
6. **Send alert to supplier** (free!)
7. **Do nothing**

The AI learns by playing thousands of times. The "world" it plays in is calibrated to **real data**: actual TSMC numbers, actual ship routes, actual oil prices, actual storms NOAA recorded since 1884.

That's the core idea. **It's a flight simulator for supply-chain crisis managers, except the pilot is an AI.**

## Why is it special?

Most hackathon projects use fake data. Shaurya used **261,175 real data points** from 8 public sources (Kaggle DataCo, NOAA storms, FRED economic, World Bank governance, USGS earthquakes, SEC 10-K filings, Wikipedia crises, policy PDFs).

He also got **15 different AI judges to grade his AI** — 3 running on his own laptop (Qwen, Mistral, DeepSeek) and 12 borrowed from OpenRouter for free (Nemotron, Hermes, Llama, Gemma, etc.). They mostly agree (Krippendorff α = 0.75 on the best 2 of 3 local judges), which is rare in AI evaluation.

And he wrote **250 tests** that all pass in 2 minutes 38 seconds. **Every claim** he makes is verifiable in 30 seconds — there's a folder called `receipts/` with 35 bash scripts, and you run one to check any number he quotes.

## What's the "Even in Arcadia" thing?

Sleep Token (a band) released an album in 2025 called *Even in Arcadia*. Arcadia is the imaginary peaceful pastoral place. The thesis: **even in a peaceful well-managed supply chain, disruptions happen.** Every research stage (`v3_arcadia/00_emergence`, `10_caramel`, `20_past_self`, `30_dangerous`, `40_granite`, `50_gethsemane`, `60_euclidian`, `70_provider`, `80_aqua_regia`, `85_infinite_baths`, `90_damocles`, `95_arcadia`) is named after a track from that album. It's not branding — it's the organizing narrative.

## What's the killer demo moment?

On 2026-04-21 (right before submitting), Shaurya pointed his system at NewsAPI and asked it to look at real news headlines from that week. **It found a real April 2026 incident — the US Navy seizing an Iranian cargo ship in the Gulf of Oman — and matched it to the pre-loaded Iran/Hormuz crisis library at 0.99 similarity in seconds.** Then it computed: "If you do nothing, you lose $324M. If you activate this hedge plan, you lose $65M. **Savings: 80%, $259M.**"

That's not a scripted demo. The system actually read live news, found a historical analog from its library, and ran a counterfactual simulation. **That's the showstopper.**

## What's "Phoenix"?

Shaurya finished v3 (the main submission) and v4 (the live ingestion layer). Then he kept going and built **Phoenix v5** as an "ascensionism" layer on top — Karpathy-style autonomous research loop, Arena leaderboard, counterfactual digital twin, ROLL framework integration, upstream PRs ready to send to Meta and Alibaba. The rule: **if Phoenix breaks, v3+v4 still ship as a top-10 submission**. Each layer is independently submission-grade.

---

# PART 2 — THE 5-PHASE PROJECT STUDY

## Phase 1 — Vision, Narrative & Credibility

**Hackathon framing** (from `JUDGES.md`, `EXTERNAL_CREDIBILITY.md`):
- Primary theme: #3.1 World Modeling — Professional Tasks (75% env quality, 25% trained agents)
- Secondary: #4 Self-Improvement (Karpathy autoresearch)
- 4-minute judge path: README → 3-min video → `/live/hormuz-closure` → pitch → optional deep-dive
- 15 reproducible bash receipts

**Origin story** — 7 named research rounds (R1-R7) over ~6 months, each mapped to a Sleep Token album track:

| Round | Stage | Commit | What shipped |
|---|---|---|---|
| R1 | Emergence | `acc19d8` | 13 SOTA foundation models verified local |
| R2 | Caramel | `b35f15e` | TabPFN + XGB/LGB/CAT tabular stack |
| R3 | Past Self | `c2d0798` | Chronos + TimesFM + ARIMA + Prophet, 20-fold backtest |
| R4 | Dangerous | `8f14607` | 3-judge LLM panel on 26 scenarios |
| R5 | Granite | `ca7a57d` | 8-pipeline RAG, mxbai P@1=0.962 |
| R6 | Gethsemane→Damocles | `ea282c4` | MaskablePPO + GCN + conformal + FastAPI |
| R6b | Euclidian | `badf3cc` | 10,800-episode bootstrap CI95 |
| R7 | Arcadia | `v3.0-arcadia` | Final synthesis |

**Headline metrics (full)**:

| Metric | Value | Source |
|---|---|---|
| Foundation models verified | 13/13 | `R1_VERIFIED.json` |
| Real data points | 261,175 | `DATA_SOURCES.md:237-250` |
| RAG P@1 (mxbai bi-encoder) | 0.962 | `R5_GRANITE.json` |
| RAG nDCG@10 (Snowflake OOD) | 0.971 | `R5_GRANITE.json` |
| RAG MRR | 0.978 | `R5_GRANITE.json` |
| 2-judge Krippendorff α | 0.750 | `R4_DANGEROUS_V2.json` |
| 12-frontier judge α | 0.567 | `README.md:89` |
| 15-judge combined α | 0.358 | `README.md:90` |
| Cohen κ (Qwen × Mistral) | 0.747 | `R4_DANGEROUS_V2.json` |
| MaskablePPO masking lift (easy) | +26.8% | `R6_GETHSEMANE_MASKING_ABLATION.json` |
| GNN MAE reduction (hard) | −64% | `R6_PROVIDER.json` |
| Conformal WTI dev @ 95% | 0.024 | `R6_AQUA_REGIA.json` |
| Euclidian CI95 (hard trained) | [2.596, 2.708] | `R6_EUCLIDIAN.json` |
| Tests | 250 passing | `RELEASE_V4_TAG.md` |
| Live Hormuz demo match | 0.99 similarity | `README.md:76` |
| Counterfactual savings | 80% ($324M → $65M) | `RELEASE_V4_TAG.md:77-85` |

**Data inventory** — zero synthetic:

| Source | Count |
|---|---|
| DataCo Supply Chain (Kaggle) | 180,519 orders / 20,652 customers / 164 countries |
| NOAA IBTRACS | 243,495 storm records / 4,289 typhoons (1884-2024) |
| FRED | 17,011 economic data points (DCOILWTICO, PCOPPUSDM, PPICMM, etc.) |
| World Bank WGI | 214 countries × 6 dimensions × 24 years |
| USGS | live earthquake feed |
| SEC 10-K | 25 Fortune 500 filings |
| Wikipedia crises | 26 articles |
| Policy PDFs | 3 (FRBSF, BIS, FRBNY) |

## Phase 2 — The OpenEnv Environment

**Files**: `openenv.yaml`, `server/` (26 files), `tests/` (12 files), `benchmark/` (36 files), `dashboard/` (5 files), `client/` (2 files), root `baseline.py`/`scripted_agent.py`/`inference.py`/`models.py`/`client.py`.

**Contract**:
- Spec version 0.1, environment_id `supplymind`, FastAPI port 8000
- Observation: `current_day`, `days_remaining`, `active_signals[]`, `node_statuses[]`, `financials`, `situation_summary` (~1500 tokens NL), `compact_summary` (100-200 tokens), `reward`, `done`
- Action: 7 types, MultiDiscrete([7, 40]) → flattened to Discrete(280)
- Reward: dense [-1.0, 1.0]

**Three tasks**:

| Task | Days | Budget | Design |
|---|---|---|---|
| `easy_typhoon_response` | 30 | $5M | Single typhoon disrupts TSMC, 72h warning |
| `medium_multi_front` | 45 | $8M | 3 concurrent crises, budget covers ~2, forces triage |
| `hard_cascading_crisis` | 60 | $10M | Taiwan Strait cascade vs $2B+ at risk |

**Crisis library** (5 historical disruptions in `benchmark/crisis_library/`):
- Tohoku 2011 ($235B, 180-day recovery)
- Suez 2021 ($9.6B/day)
- Chip shortage 2020 (commodity multipliers)
- Ukraine neon 2022 (lithography supply)
- Red Sea 2023 (Houthi attacks)

**Engine** (`server/engine/simulation.py`) — per-step pipeline:
1. Validate + apply action with budget check
2. Real cost constants (ISM/CSCMP/IATA-calibrated)
3. Disruption lifecycle: sigmoid (warning) → bell (active) → exp decay (recovery)
4. BFS propagation, `SEVERITY_DECAY_PER_HOP = 0.20`
5. Commodity price effects
6. Monte Carlo p50/p95 projection (Numba JIT, <0.01ms warm)
7. 7-component dense reward

**Anti-reward-hacking** (`tests/test_reward_hacking_adversarial.py`): 6 attacks tested, all rejected by different layered defenses. Receipt at `tests/receipts/adversarial_reward_audit.json`.

**Benchmark headline** (`FINAL_RESULTS.json`) — 9 agents × 3 tasks × 5 seeds × 20 episodes:

| Agent | Easy | Med | Hard | Avg |
|---|---:|---:|---:|---:|
| Random | 0.709 | 0.598 | 0.727 | 0.678 |
| Scripted | 0.336 | 0.207 | 0.571 | 0.371 |
| BC | 0.663 | 0.500 | 0.610 | 0.591 |
| CQL | 0.688 | 0.629 | 0.655 | 0.657 |
| IQL | 0.689 | 0.629 | 0.656 | 0.658 |
| TD3+BC | 0.678 | 0.629 | 0.656 | 0.654 |
| **QR-DQN specialist** | **0.863** | **0.844** | **0.671** | **0.793** |

All pairwise Wilcoxon p < 1e-50.

## Phase 3 — RL/ML Training Stack (`rl/`)

**21+ techniques** in pure PyTorch:

| Technique | Module | Result |
|---|---|---|
| MaskablePPO | `train_ppo.py` | 32 envs, 2M steps, ~8min RTX 4080 |
| Constrained PPO | `constrained_ppo.py` | Lagrangian budget guarantee |
| QR-DQN | `distributional/qr_dqn.py` | 51 quantiles, CVaR-optimal |
| HER | `her_agent.py` | SAC + hindsight relabeling |
| Decision Transformer | `decision_transformer/` | GPT-2 backbone, RTG conditioning |
| TFT | `forecasting/tft.py` | MAE $7.83/bbl on WTI |
| BC/CQL/TD3+BC/IQL | `offline/baselines.py` | Pure PyTorch, no d3rlpy |
| TGN | `gnn/tgn.py` | Per-node memory + GRU |
| RAG | `rag/indexer.py` | Ollama nomic-embed + ChromaDB |
| MC Dropout | `uncertainty.py` | Q1: 99.76% acc, Q4: 55.92% |
| SHAP | `interpretability/` | Weather 60.1% importance |
| Federated | `federated/fedavg.py` | +263% full / +77% type accuracy |
| LoRA | `lora/finetune.py` | Qwen2.5-1.5B adapters |
| Pareto NSGA2 | `pareto/frontier.py` | 3-objective frontier |
| Autoresearch | `autoresearch.py` | 10 experiments, best 0.680 avg |
| Specialist router | `specialist_router.py` | Per-task best checkpoint |
| ONNX export | `export_onnx.py` | <5e-5 roundtrip error |
| Numba MC | `fast_engine/fast_monte_carlo.py` | 10-50× speedup |

**Key results**:
- TFT MAE: $7.83/bbl (P50, WTI)
- MC Dropout Q1 acc 99.76% vs Q4 55.92%
- SHAP: NOAA weather 60.1%, node features 21.8%, FRED 8.5%
- Federated round 0→49: full acc 8.5%→31.0%, type acc 42.7%→75.8%
- Autoresearch best: experiment_003 (lr=1e-3, CVaR α=0.5, hidden=256) → 0.680 avg
- Pareto: Africa cost 177.5 / resilience 0.454 / carbon 6504 (best)
- ONNX: BC 3.05e-5, CQL 5.22e-8, IQL 3.05e-5, TD3+BC 1.53e-5

## Phase 4 — v3_arcadia Research Journey + Phoenix

**12 numbered stages = 12 album tracks**:

| Stage | What | Headline result |
|---|---|---|
| 00 Emergence | 13 models verified | All operational, Q4_K_M quantization |
| 10 Caramel | TabPFN bagging + SHAP fairness | XGB late_delivery acc 0.8369 |
| 20 Past Self | TimesFM/Chronos/ARIMA forecasting | DIR_ACC 0.59-0.62, MAE 1.45 USD/bbl (h7) |
| 30 Dangerous | 3-judge LLM panel | α=0.750, Cohen κ=0.747, 26 scenarios |
| 40 Granite | 8-pipeline RAG | mxbai P@1=0.962, +46pp vs v3 Block 4 |
| 50 Gethsemane | MaskablePPO training | Hard mean 2.674, 0 violations |
| 60 Euclidian | 10,800-episode bootstrap | Hard CI95 [2.596, 2.708] vs greedy [-1.48, -1.35] |
| 70 Provider | Custom 3-layer GAT | F1 0.964 hard, MAE −64% vs MLP |
| 80 Aqua Regia | Split-conformal | WTI dev=0.024, coverage 91.2-95.5% |
| 85 Infinite Baths | Streamlit dashboard | Live trained policies |
| 90 Damocles | FastAPI service | /assess, /forecast, /rag, /rl/act |
| 95 Arcadia | Final README | Complete synthesis |

**Phoenix v5 layer** (`ShAuRyA_Phoenix/`):
- Autoresearch 5 experiments: s1 ACCEPTED, s2 ACCEPTED, **s3 curriculum BEST (CI95 [0.552, 0.761])**, s4 RecurrentPPO REJECTED, s5 diversity REJECTED
- Counterfactual Twin: severity=0.85 + Brent=$123 → trained $187M loss vs no-action $367M = **$178.7M saved (48%)**
- Arena: 6 baselines, MaskablePPO #1 at mean=2.209
- ROLL integration: GiGPO + DPO + 3-judge reward worker
- 20 receipts (13 v4 + 7 v5)
- Upstream PRs to Meta OpenEnv + Alibaba ROLL ready

## Phase 5 — Supplymind v4 Product + 12-Frontier Judge Panel

**v4 release** (2026-04-21, Sleep Token "Rain"): purely additive on frozen v3, 76 new tests = 249 total in 2m38s.

**Feature ledger** (full numbers):

| Feature | Headline metric | Value |
|---|---|---|
| R9 Analyst v5 vs base Qwen | Exact-risk acc | 80% v5 / 0% base |
| R6 SPOF v2 (articulation-point fix) | Mean F1 | 1.000 (vs 0.949 v1) |
| R15 Stacking v2 | AUC vs WV | +0.0045 (honest null vs best single) |
| F2 Multi-agent demo | Apple PnL | +$2.74M (winner) |
| F4 DT risk slider | Conservative→aggressive return | 0.62 → 0.77 |
| F6 Conformal RL | 95% CI95 width | 2.36 → ABSTAIN |
| F9 Pareto carbon | Frontier ratio | 11/20 = 55% |
| F14 CUDA kernel | PyTorch fallback | 0.028ms (1833× over Python) |

**The 12-frontier panel** (`scripts/run_frontier_judge_panel.py`):

12 OpenRouter models: Nemotron-3-Super 120B, Ling-2.6-1T, Hermes-3-405B, GPT-OSS-120B, Gemma-4-31B, Gemma-4-26B-A4B, Qwen3-Next-80B, GLM-4.5-Air, Llama-3.3-70B, Nemotron-3-Nano-30B, MiniMax-M2.5, Nemotron-Nano-9B + 3 local (DeepSeek-R1, Qwen2.5:14b, Mistral-Nemo) = **15 judges total**.

Rate limit policy: 18 req/min, 950 req/day. Free tier mostly; ~$0.04 paid total. Honest α disclosure ladder:
- Raw 3-local panel: **0.210**
- 2-judge ablation: **0.750**
- 12-frontier: **0.567**
- 15-judge combined: **0.358**

**Live Hormuz scenario** (`scenarios/iran_israel_hormuz_2024_2026.json`) — 8 real events with 26 citations:
1. Iran True Promise 1 (2024-04-13) — Brent 90.7→92.2
2. Iran True Promise 2 (2024-10-01) — Brent 71.8→78.2
3. Houthi Red Sea (2023-11-19 ongoing 884 days) — Brent 82.1→92.2 peak
4. Op Poseidon Archer (2024-01-11) — Brent 77.6→81.0
5. Haifa port missile (2024-10-07) — Brent 74.2→78.2
6. Houthi Yaffa Tel Aviv (2024-07-19) — Brent 85.4→87.1
7. **Hormuz Trump cargo (2026-04-18) LIVE** — Brent 119.1→123.3, P95=168
8. Ukraine neon shock (2022-02-24, 310 days) — Brent 96.8→127.6 peak

**Notebooks** (6):
1. environment_quickstart (CPU, <10min)
2. training_your_own_agent
3. reproducing_benchmarks
4. v3_quickstart_colab
5. **v4_hormuz_live** — the headline demo, ~5min Colab CPU, no GPU/Ollama/keys
6. trl_training_colab

**Receipts ledger** (`ShAuRyA_Supplymind/receipts/INDEX.md`) — 15 one-line bash commands:

| # | Claim | Value |
|---|---|---|
| 1 | RAG mxbai P@1 | 0.9623 |
| 2 | RAG mxbai MRR | 0.9780 |
| 3 | BEIR Snowflake nDCG@10 | 0.9710 |
| 4 | 2-judge Krippendorff α | **0.7500** |
| 5 | Cohen κ Qwen×Mistral | 0.7474 |
| 6 | Masking ablation easy lift | 26.77% |
| 7 | GCN easy MAE vs MLP | 48.02% |
| 8 | Aqua Regia WTI dev95 | 0.0238 |
| 9 | TimesFM CP WTI dev95 | 0.0500 |
| 10 | V4 SPOF V2 F1 | **1.0** |
| 11 | V4 Stacking V2 lift vs WV | 0.0045 |
| 12 | V4 Live Brent 2026-04 | **123.28** |
| 13 | V4 Tests Total | 250 |
| 14 | V4 Analyst V5 Exact Acc | **0.8** |
| 15 | V4 Autoresearch Best CI95 | **0.5514** |

---

# PART 3 — WIN-CHANCES HONEST ASSESSMENT

I'm going to be brutally honest with you because that's what you asked for.

## The strengths (real and rare)

1. **Volume of evidence** — 803 files, ~6 months of work, 250 tests passing, 35 reproducible receipts. Most hackathon teams have <100 files and 0-20 tests. **This alone puts you in the top 10%.**

2. **Real data discipline** — 261,175 verified data points, zero synthetic. Every cost constant cited (ISM, CSCMP, IATA, McKinsey, Lloyd's, SemiAnalysis, AlixPartners). Most hackathons fake this.

3. **OpenEnv compliance is unambiguous** — 19 formal compliance tests in 2 seconds. The framework is unambiguous about what counts; you have it.

4. **PyTorch mastery** — custom 3-layer GCN with `index_add_`, MaskablePPO discrete wrapper, conformal prediction wrappers, ONNX export at <5e-5 error, Numba JIT. This is engineering, not library stacking.

5. **Honesty as a weapon** — published α=0.210 then improved to α=0.750 instead of cherry-picking. Published stacking-v2 honest null. Published RecurrentPPO REJECTION. This is rare and judges with research backgrounds will notice.

6. **The Hormuz 0.99 match** — if it lands live, it's the single best demo moment. Real news, real analog, real $259M counterfactual.

7. **Reproducibility receipts** — every claim is one bash command. Judges can verify in 30s. Most hackathons can't.

8. **Phoenix as safety net** — three independent layers (v3, v4, v5). Even if v5 breaks live, v3 still ships submission-grade.

9. **Upstream PRs ready** — Meta OpenEnv + Alibaba ROLL drafts. Says "I built this *for the ecosystem*, not just the prize."

10. **Solo-dev narrative is sympathetic** — judges know what 6 months of solo work looks like.

## The weaknesses (also real)

1. **QR-DQN specialist hits 0.793 avg, not crushing it.** 0.86 easy and 0.84 medium are strong but 0.67 hard is mid. Some judges will see "67%" and not be impressed even though that's a hard task.

2. **Krippendorff α=0.21 raw** is a real number that exists in your repo. Yes the 2-judge ablation gets it to 0.75, but a hostile judge can point to 0.21 and 0.567 (12-frontier) and say "judges don't agree."

3. **CUDA kernel didn't compile on Windows.** F14 reports honest fallback. Some judges will read this as "couldn't ship." The 1833× speedup over naive Python is real but the speedup is **not over PyTorch** — PyTorch fallback already does it. F14 framing matters.

4. **LoRA training deferred to v4.1.** Dry-run only. Some judges will count this against the "Self-Improvement" theme.

5. **R15 stacking v2** is a published null (+0.0045 AUC vs WV, −0.0002 vs best single). You frame it as "honest null on saturated task" — that's correct, but a hostile judge will read it as "didn't work."

6. **DeepSeek-R1 31% accuracy reframed as devil's advocate.** Honest framing, but a skeptical judge will hear "your most expensive judge is wrong 70% of the time."

7. **RecurrentPPO collapsed (s4 REJECTED).** Documented honestly, but it's still a negative result in the deck.

8. **One-person presentation team.** No specialization, no team rehearsal, no Q&A backup. You answer everything alone for 4 minutes.

9. **HF Space cold start 15-25s.** If a judge clicks `/live/hormuz-closure` and waits >10 seconds during a 4-min review, that's a problem.

10. **OpenRouter free-tier rate limits** (18 req/min) mean if a judge tries to re-run the 12-frontier panel during demo, it'll get throttled.

## The unknowns (out of your control)

- **Hackathon size**: Bangalore on-campus. Likely 50-200 teams. Top-3 means beating 47-197 others.
- **Judge taste**: Researchers will love rigor. Investors will love story. Engineers will love compliance. If your judges are mostly investors, the Hormuz demo wins. If mostly researchers, the receipts win. If mostly engineers, the OpenEnv compliance wins.
- **Other competitors**: Unknown. Possible some teams have GPU clusters and team-of-5 polish. Unlikely many have your level of rigor + receipts. Almost certainly nobody has your Sleep Token narrative.
- **Live demo execution**: Every live system has 5-10% chance of breaking on stage. Have a backup video.

## My honest probability estimate

| Outcome | Probability | Why |
|---|---|---|
| **Top 50%** | ~95% | Tests pass, demo works, stack is real. Floor is high. |
| **Top 25%** | ~85% | The volume + rigor + receipts most teams won't have. |
| **Top 10%** | ~55% | Real-data discipline + OpenEnv compliance + 250 tests is rare. |
| **Top 3** | ~25% | Depends on judge taste, demo execution, what others bring. |
| **Win (1st)** | **~10%** | Hackathon outcomes have noise; one team with similar rigor + better polish could beat it. |

## The 5 things that would 2× your win probability

1. **Practice the 4-minute path 10 times before stage.** Memorize: open README → click HF Space → POST `/live/hormuz-closure` → show 0.99 match → show $259M savings → say "every number is one bash command, here's the receipts folder."

2. **Pre-record a 3-minute backup video.** If `/live/` fails on stage, fall back to video. This is the #1 demo risk.

3. **Lead with the Hormuz match, not the architecture.** Story first. Most judges will tune out architecture diagrams; nobody tunes out "real news from this week, 0.99 match, $259M saved."

4. **Have one slide that says "I am one person and built this in 6 months."** Solo-dev narrative is genuinely sympathetic and most teams won't have it.

5. **End with the receipts folder open on screen.** Say: "Click any of these 35 bash scripts and verify any number I just claimed in 30 seconds. Judge me on rigor."

## The single biggest risk

**The live demo.** You have a deployed HF Space. You have a `/live/hormuz-closure` endpoint that needs NewsAPI + GDELT + FRED + Ollama + 3 LLMs. **Any one of these failing during the 4-minute review window is a 5-15% probability event.** Have the offline replay (`replay_cache_latest.json`) wired and one keypress away. The Phoenix `freeze_cache.py` exists for this reason — use it.

## The single biggest opportunity

**The 0.99 Hormuz match is genuinely surprising.** Most judges have never seen a hackathon project ingest real news from this week and find a 0.99-similar historical analog with a real-dollar counterfactual. **Lead with that.** It's your moat.

---

# PART 4 — RISK FACTORS & OPEN ISSUES

(Pulled from `_dump/FAILURE_TABLE.md` and the deep audit)

**Resolved (8)**:
- PPO action mask shape (280 vs 47) → FlatDiscreteEnv wrapper
- QR-DQN signature mismatch → standardized
- TD3BC CUDA OOM → batch reduction + Q4_K_M quantization
- ONNX SentenceTransformer init hang → explicit `backend="torch"`
- PyTorch 2.11 GradScaler rename → pinned 2.5.1+cu121
- `weights_only=True` regression → explicit `weights_only=False`
- v4 autoresearch crashed all seeds → Phoenix rebuilt from artifacts
- Curriculum learning save→load fix (MaskablePPO cached action_dims)

**Deferred (6, not failures)**:
- CUDA kernel compile on Windows (MSVC `cl.exe` blocker; Python fallback ships)
- HER port (env uses MultiDiscrete, deferred)
- Online PPO/QR-DQN full real-data retrain
- Optuna 100-trial HPO (10 autoresearch experiments instead)
- GNN GATConv/TGN advanced variants (custom 3-layer GCN ships)
- Qwen-2.5-VL port imagery full inference (heuristic mode ships)

**Honest negative findings retained (8)**:
1. TabPFN 10K cap caused stack < best single → fixed in v2 pre-cache
2. Inverse-MAE ensemble < best single → v2 Bates-Granger constrained stacking
3. Krippendorff α=0.210 on raw 3-judge → 2-judge ablation α=0.750
4. Reranker hurts on easy → hard-query redemption +5pp
5. AquaRegia conformal under-covers → per-horizon v2 hits nominal
6. Provider easy-graph F1=1.000 trivial → arrival-time regression v2
7. IQL_real + TD3+BC_real collapse → valuable negative on domain transfer
8. DeepSeek-R1 31% accuracy → reframed as devil's-advocate role

---

# APPENDICES — FILE-BY-FILE INVENTORY

The following appendices itemize ~792 of 803 files. They are dense reference. Skim by header.


# APPENDIX A — `rl/` FILE INVENTORY (182 files)

## A.1 `rl/checkpoints/` (45 files)

**Result JSONs** (read in full):
- `autoresearch_final.json` — 10 experiments aggregated. Best: `experiment_003` grade_avg=0.6801, lr=0.001, cvar_alpha=0.5, hidden_dim=256.
- `mc_dropout_eval.json` — n_eval=5000, n_passes=50, mean_posterior_accuracy=0.851, mean_epistemic=0.328. Q1 (low_unc): 0.9976 acc / 0.205 ε; Q4: 0.5592 / 0.504.
- `shap_real.json` — Top-20 features. Lead: node0_risk (2.077), status (1.413), node0_inv (1.079). Group: NOAA 60.1%, node 21.8%, status 9.7%, FRED 8.5%.
- `specialist_router_real.json` — easy→bc_best_real_v2.pt, medium→cql_best_real_v2.pt, hard→iql_best_real_v2.pt, ensemble DT/BC weights 0.3/0.7.
- `phase_k_results.json` — Federated 4 rounds; round-0 global_acc=0.0119, round-9 0.0363; per-bucket Q1=0.9976→Q4=0.5592.
- `phase_x_results.json` — CUDA compile status, federated_full 50 rounds (round-0 0.0854 → round-49 0.3101), pareto 3 markets, fast_mc active, optuna_cql 12 trials best lr=0.000354 cw=1.579.
- `ensemble_v2.json` — BC=0.374, CQL=0.375, IQL=0.371, TD3BC=0.371; mv_acc=0.374, wv_acc=0.375.
- `optuna_cql_v2.json` — 12 trials, best value=0.376.
- `onnx_roundtrip.json` — BC 3.05e-5, CQL 5.22e-8, IQL 3.05e-5, TD3BC 1.53e-5; all verified.
- `pareto_results.json` — 5 policies, 2 Pareto-optimal (mask [T,T,F,F,F]), 3 objectives (cost/resilience_loss/carbon).
- `pareto_frontier_v2.json` — Africa cost=177.5/res=0.454/carbon=6504; LATAM cost=179/res=0.456/carbon=28781; USCA cost=176.5/res=0.452/carbon=14676.
- `explainer_stress_v2.json` — n_test=50, passed=50, pass_rate=1.0; 50 scenarios, lengths 700-980 tokens.
- `mc_dropout_v2.json` — BC_v2 acc=0.369, type_acc=0.862, ECE_full=0.0229.
- `tft_v2_metrics.json` — 3-target FRED (DCOILWTICO/PCOPPUSDM/PPICMM); params=513,534, best_val_qloss=0.0245, fold-0 MAE 504.6.
- `tft_real_metrics.json` — DCOILWTICO; mae_p50_usd=7.83, best_val_quantile_loss=0.071, enc_len=60, horizon=14.
- `federated_v2_metrics.json` — 50 rounds; round-0 (val_full=0.0854, val_type=0.427), round-49 (0.3101, 0.758).
- `federated_real_metrics.json` — 4 rounds + clients [Pacific Asia, Europe, LATAM].
- `shap_cql_v2.json` — n_background=1000, n_explained=1000; lead node0_inv (5.06), status (3.818), node0_risk (2.865).
- `world_model_v2_rollout.json` — 1-step loss 0.00328, 5-step 0.00583, 15-step 3607.3.

**Model checkpoints (.pt/.zip/.npz)**:
- `federated_real.pt` 682K, `bc_best_real_v2.pt` 681K, `qrdqn_v2_easy.pt` 7.6M, `cql_v2.pt` 3.8M, `rssm_real.pt` 3.5M, `iql_v2.pt` 7.9M, `cql_best_real_v2.pt` 1.9M, `dt_best_real_v2.pt` 2.8M, `dqn_her_v2.pt` 756K, `qrdqn_v2_hard.pt` 7.6M, `td3bc_best_real_v2.pt` 2.8M, `qrdqn_v2_medium.pt` 7.6M, `world_model_real.pt` 2.3M, `iql_best_real_v2.pt` 3.2M, `td3bc_v2.pt` 2.4M, `bc_v2.pt` 2.4M.
- `ensemble_tuning.npz` 918B, `constrained_ppo_stats_easy/medium/hard.npz` 564-612B.
- `ppo_best_easy/best_model.zip` 1.1M, `ppo_best_medium/best_model.zip` 3.3M, `ppo_best_hard/best_model.zip` 3.3M.
- `onnx/` subdirectory — production ONNX exports.

## A.2 `rl/analysis/` (33 files)

**Top-level Python**:
- `__init__.py` — empty marker.
- `trained_models.py` — Phase J: 5 data-driven models. political_risk GBR R²=0.994 / MAE=0.0095, dependency_scoring MLP acc=0.974, financial_impact Ridge R²=0.736 / MAE=26.04, confidence isotonic ECE=0.0017, safety_stock empirical lead-time.
- `confidence.py` — 4-level alert (RED ≥0.8 / AMBER ≥0.5 / YELLOW ≥0.3 / GREEN <0.3) via prediction*0.5 + historical*0.3 + corroboration_bonus.
- `financial_impact.py` — EBITDA: lost_margin + expedite_premium + sla_penalties + reputation_cost. McKinsey constants (gross_margin=35%, expedite_mult=3.0, sla=$25K/day).
- `dependency_scoring.py` — 4-component criticality (0-100): single_source 40 + revenue 30 + lead_time 15 + geo_concentration 15 (Taiwan=14, China=12).
- `political_risk.py` — 8-component index: governance 15% + fragile_state 10% + ease_of_business 5% + conflict 20% + gdelt_tone 15% + sanctions 15% + travel 10% + currency 10%.
- `safety_stock.py` — risk_adj_lead = base × (1 + disrupt_prob × dur / base). Multipliers: conservative 2.5 / moderate 1.5 / aggressive 1.0.
- `spof.py` — NetworkX articulation-point detection on supply-chain DAG.

**`trained/` subdirectory**:
- `analysis_v2_metrics.json` — wgi_temporal MSE=0.00037, safety_stock_seasonal p95=[0.747-0.792], spof_gnn F1=0.0 (n=8), financial_impact_mae_ci95=[24.8-26.5].
- `phase_j_results.json` — political_risk MAE=0.0095/R²=0.994 (214 countries), dependency_scoring acc=0.974 (144K), financial_impact MAE=26.04, confidence ECE=0.0017, safety_stock mean_lt=3.50±1.62.
- `*.pkl` — 7 sklearn models (confidence_isotonic, dependency_scoring_mlp, financial_impact_ridge, political_risk_gbr, political_risk_lstm, safety_stock_empirical, safety_stock_seasonal).

**`trained/v3/` subdirectory**:
- `benefit_per_order_metrics.json` — Regression. n_train=126374, val=27067, test=27078, 111 features. XGB MAE=54.4, LGB 54.3, CatBoost 54.4, stacked 54.2 with R²~0.02.
- `shipping_mode_metrics.json`, `delivery_status_metrics.json`, `late_delivery_risk_metrics.json` — 3-model ensemble + stack.
- 12 .pkl files (per-task XGB/LGB/Cat models).

## A.3 `rl/data/` (23 files)

- `build_unified_buffer.py` — Phase A: fuses DataCo (180K) + NOAA + USGS + FRED → real_unified.npz with 70/15/15 stratified splits.
- `build_unified_buffer_v2.py` — Phase M: per-storm NOAA, time-windowed USGS, full WGI timeseries, fred_extended, leading_indicators as taxonomy, dataco_access_logs, learned financial reward, multi-step via customer_id.
- `real_train.npz`, `real_val.npz`, `real_test.npz` — Phase A splits.
- `real_train_v2.npz`, `real_val_v2.npz`, `real_test_v2.npz` — Phase M splits.
- `real_unified.npz`, `real_unified_v2.npz` — full buffers.
- `dataco_statistics.json` — 180,519 orders, 20,652 customers, late_delivery_rate=0.573, avg_profit_ratio=0.121.
- `noaa_real_calibration.json` — 4,289 storms, 140-year coverage.
- `taiwan_strait_calibration.json` — TSMC 54% foundry, 92% advanced nodes.
- `red_sea_calibration.json` — +10 transit days via Cape, +25% fuel.
- `disruption_taxonomy.json` — 15+ types with prevalence.
- `fred_cache.json` — 7-series, 17K data points, date→7-vec.
- `fred_extended.json` — 5 additional series (PPI, IP, interest rate).
- `leading_indicators.json` — 15 indicators with correlations.
- `fred_state_features.json` — state[400:407] price vectors.
- `lora_training_data.json` — 225 instruction/output pairs.
- `explanations_cache.json` — 50 pre-computed state→explanation pairs.
- `real_unified_v2_meta.json` — n_total=180519 (126360/27076/27083), 164 unique_actions, 20652 customers, multi_step_fraction=0.886, schema map.

## A.4 `rl/legacy/` (10 files)

- `buffers/offline_buffer_simulated.npz` — pre-Phase A simulator buffer.
- `fallbacks/__init__.py` — empty.
- `fallbacks/financial_impact.py` — formula-only legacy (superseded by Ridge).
- `fallbacks/confidence.py` — legacy formula scoring.
- `fallbacks/dependency_scoring.py` — legacy formula scorer.
- `fallbacks/political_risk.py` — legacy formula + pre-computed country data.
- `fallbacks/safety_stock.py` — legacy formula calculator.
- `fallbacks/explainer_heuristic.py` — heuristic fallback if Ollama down.
- `fallbacks/rag_indexer_with_fallback.py` — RAG with hardcoded precedent fallback (production has none).

## A.5 `rl/lora/` (7 files)

- `__init__.py` — empty.
- `create_ollama_model.py` — Builds Modelfile from system prompt + 5 real training examples, qwen2.5:14b base.
- `finetune.py` — PEFT LoRA fine-tune on Qwen2.5-1.5B; TRL + bitsandbytes; .venv311 required; outputs to checkpoints/lora/.
- `Modelfile`, `Modelfile.v2`, `Modelfile.v3`, `Modelfile.v4` — Ollama Modelfiles. v1: TSMC (54%/92%), Red Sea (+10d/+25%), action costs, SLA $25K/day, 5 examples.

## A.6 `rl/surrogate/` (5 files)

- `__init__.py` — empty.
- `world_model.py` — Linear(688→512)→ReLU→Linear(512→256)→ReLU→(state/reward/done) heads. 500K transitions, ~4min GPU. Enables 100K MC <80ms.
- `rssm.py` — DreamerV3-style: encoder(408→latent), GRUCell, decoder. 15-step latent rollouts.
- `counterfactual.py` — Replays do_nothing from snapshot; "without action, additional loss = $X".
- `gpu_monte_carlo.py` — GPU MC: 1 state → 100K with noise linspace(0.01, 0.3), one pass, returns p5/p50/p95/p99/cvar_10.

## A.7 `rl/offline/` (5 files)

- `__init__.py` — empty.
- `baselines.py` — Pure PyTorch BC/TD3+BC/CQL/IQL. BC ~5min, TD3+BC ~12min, CQL ~15min on 100K offline_buffer.npz.
- `baselines_v2.py` — Updated implementations.
- `dataset.py` — Generates offline_buffer.npz: scripted (good) + random (exploration); injects FRED prices.
- `iql_agent.py` — IQL entry redirector to baselines.py.

## A.8 `rl/forecasting/` (4 files)

- `__init__.py` — empty.
- `tft.py` — Temporal Fusion Transformer. pytorch-forecasting, QuantileLoss [0.1/0.5/0.9], hidden=16, attn=1, enc=90, pred=30, ~20min/100 epochs GPU.
- `mc_dropout_eval.py` — Phase D: 50 stochastic passes, mean±2σ per action, on real_test.npz (27K).
- `train_tft_real.py` — TFT training on real FRED.

## A.9 `rl/cuda/` (4 files)

- `__init__.py` — empty.
- `action_mask_kernel.py` — PyTorch wrapper for CUDA kernel; `apply_action_mask()` and `masked_argmax()` with fallback.
- `action_mask_kernel.cu` — CUDA source.
- `action_mask.dll` — compiled Windows DLL.

## A.10 `rl/decision_transformer/` (3 files)

- `__init__.py` — empty.
- `model.py` — GPT-2 backbone. (rtg, state, action) interleave. Embeds: rtg(1→128), state(408→128), action(280→128), timestep(60→128). n_embd=128, n_layer=3, n_head=1, ctx=20.
- `train.py` — DT training script.

## A.11 `rl/distributional/` (3 files)

- `__init__.py` — empty.
- `qr_dqn.py` — Quantile regression DQN. 51 quantiles → CVaR-optimal worst-10%. 408→256→128→(280×51).
- `train.py` — QR-DQN training loop.

## A.12 `rl/pareto/`, `rl/multi_agent/`, `rl/federated/`, `rl/rag/`, `rl/interpretability/`, `rl/gnn/` (15 files)

- `pareto/__init__.py`, `pareto/frontier.py` — 3-objective NSGA2 via pymoo. Carbon factors (kg CO2/tonne-km): air 0.82, sea 0.013, rail 0.028, road 0.096.
- `multi_agent/__init__.py`, `multi_agent/competitive.py` — Apple/Samsung/Toyota archetypes; shared supplier capacity FCFS.
- `federated/__init__.py`, `federated/fedavg.py` — 3 simulated companies, 20 rounds × 5 local epochs, optional DP (noise_std=0.1). +23% over individual.
- `rag/__init__.py`, `rag/indexer.py`, `rag/build_corpus.py` — Ollama nomic-embed (768d), ChromaDB persistent, min_score=0.60. Indexes crisis_library + NOAA top-200 storms + USGS + DataCo summaries.
- `interpretability/__init__.py`, `interpretability/shap_real.py`, `interpretability/shap_analysis.py` — Phase F: DeepExplainer on real BC; bg=500, explain=200. Per-group aggregates + global top-20.
- `gnn/__init__.py`, `gnn/tgn.py`, `gnn/attention.py` — TGNMemory + TransformerConv. memory_dim=64, time_dim=8, 2 heads. PyG ≥2.3. ~2× slower than static.

## A.13 `rl/fast_engine/` (4 files)

- `__init__.py` — empty.
- `fast_monte_carlo.py` — Numba @njit MC hotloop. 10-50× speedup. Falls back to Python if Numba unavailable.
- `benchmark.py` — Performance harness.
- `README.md` — Documentation.

## A.14 Top-level `rl/*.py` (20 files)

- `train_ppo.py` — MaskablePPO. 32 envs, VecNormalize, n_steps=2048, batch=512, lr=3e-4, 2M steps ~8min RTX 4080. cudnn.benchmark=True, allow_tf32=True. MLflow + W&B.
- `constrained_ppo.py` — PPO + Lagrangian. λ self-tuning. Mathematically guaranteed budget adherence.
- `her_agent.py` — SAC + HER. GoalEnv wrapper. +30-50% on hard. 500K steps.
- `specialist_router.py` — Per-task best-checkpoint dispatch. Scripted fallback.
- `hpo.py` — Optuna 50 trials × 500K PPO. Tunes lr, n_steps, clip, ent_coef, gamma, gae, net_arch.
- `autoresearch.py` — Karpathy-style. 10-20 experiments varying lr, cvar_α, reward shape, network size. Outputs autoresearch_final.json.
- `autoresearch_summary.py` — Aggregates: best_overall, best_per_family, MD summary.
- `real_world_benchmark.py` — Eval on 180K DataCo + NOAA + USGS. Compares predicted late_rate (57.3% actual).
- `leaderboard.py` — HF Spaces Gradio app. Displays rankings + agent submission.
- `record_video.py` — MP4 of 3 agents (scripted, PPO, QR-DQN CVaR). Gymnasium RecordVideo + matplotlib.
- `export_onnx.py` — QR-DQN → ONNX opset=17. dummy=(1,408). Output supplymind_policy.onnx.
- `ensemble.py` — DT + QR-DQN. ensemble_logits = w·DT + (1-w)·QR. Grid search w ∈ [0.1, 0.9].
- `gym_env.py` — Gymnasium wrapper. state=(408,), action=MultiDiscrete([7,40]), reward=[-1,1].
- `real_data_pipeline.py` — DataCo encoding, NOAA/USGS/FRED injection, splits.
- `real_data_integration.py` — RealWorldCalibrator. Every constant has a URL.
- `explainer.py` — Local Ollama qwen2.5:14b. ~3-4s/explanation. 4-section (state/action/reward/shap/cf/rag). Caches 50 common.
- `dataco_integration.py` — DataCoAnalyzer extracts patterns from 180K. Delays, segments, margins, SLA.
- `uncertainty.py` — MC Dropout. model.train() at inference. 50 passes. mean+std epistemic.
- `__init__.py` — Gymnasium register. SupplyMind-Easy/Medium/Hard-v1, max_steps=30/45/60.
- `autoresearch_final.json` — see A.1.


# APPENDIX B — `v3_arcadia/` FILE INVENTORY (177 files)

## B.1 Root training blocks (5 files)

- `train_v3_block1_real_labels.py` — Leak-free 4-task tabular: late_delivery_risk, shipping_mode, delivery_status, benefit_per_order. 4-model stack (XGB/LGB/Cat/TabPFN). Bootstrap 95% CI, ECE, Brier. 109 features × 180K DataCo rows.
- `train_v3_block2_forecasting.py` — Foundation zero-shot on 8 FRED targets (DCOILWTICO, PCOPPUSDM, DEXTAUS, DEXKOUS, DEXJPUS, DEXUSEU, DEXCHUS, PPICMM) × 3 horizons. Chronos-Bolt + TimesFM + Prophet + ARIMA + BigTFT. 20-fold rolling-origin. Inverse-MAE weighting.
- `train_v3_block3_llm.py` — 4-LLM panel via Ollama. supplymind-analyst:v4 Modelfile + 10-shot. 50 scenario A/B with 3-judge majority. JSON-mode validated.
- `train_v3_block4_rag.py` — 8 pipelines on 6,483 chunks × 53 queries. P@1/3/5, R@5/10, MRR, nDCG@10.
- `train_v3_block5_rl.py` — MaskablePPO 500K × 3 tasks, RecurrentPPO LSTM 300K, DQN+HER 2000ep, SAC-Discrete. Mask-shape fix at line 72-86.

## B.2 `00_emergence/` (16 files)

- `verify_embedders_chronos.py` — 5 embedders + Chronos. BGE-M3 dim=1024 score 0.638, mxbai 1024 score 0.736, Snowflake 1024 score 0.582.
- `verify_qwen_vl.py` — Qwen-2.5-VL-7B via transformers + qwen-vl-utils, 224×224 image.
- `verify_tabpfn.py` — TabPFN-v2 clf+reg from local ckpts, 200×12 binary + regression.
- `verify_timesfm.py` — TimesFM-2: 50 layers, 1280 dims, 16 heads, 2048 context.
- `verify_mistral_nemo.py` — 3 tests: reasoning (180 tok), long-ctx (120 tok), JSON-mode (200 tok).
- `verify_qwen14b.py` — 3 tests: factual, reasoning, JSON-mode.
- `verify_qwen_coder.py` — code_gen (250 tok), code_review (300 tok), JSON-mode (150 tok).
- `r1_qwen_vl_downstream.py` — Synthetic GOES-16-style 512×512 satellite (storm + coastline). Full HF pipeline, 400-token gen.
- `deepseek-r1.Modelfile` — temp 0.1, top_p 0.9, ctx 32K, 512-tok limit.
- `qwen25-14b.Modelfile` — Analyst v3 base. SYSTEM with TSMC/Tohoku/Suez/chip-shortage facts.
- `qwen25-coder-14b.Modelfile` — JSON-mode for code analysis.
- `fetch_extra_data.py` — UN COMTRADE (5 countries), IMF IFS (5 indicators × 5), Wikipedia (26 crisis articles).
- `convert_bge_to_safetensors.py` — Bypass torch.load weights_only on torch <2.6, ~2GB output.
- `mem_check.ps1`, `ram_check.ps1` — Windows PowerShell diagnostics.
- (extra Modelfile / verification artifacts)

## B.3 `10_caramel/` (5 files)

- `train_caramel.py` — 4 DataCo targets. BASE_NUM (16 numeric) + CAT_COLS (7). XGB hist 1000 trees / LGB 1500 / Cat 1500 / TabPFN-v2. CIs + ECE + Brier. Late_delivery: XGB acc=0.8369, AUC=0.916.
- `r2_tabpfn_bagging.py` — Random-subsample TabPFN ensemble, predict_proba averaging.
- `r2_tabpfn_bagging_full.py` — CV folds, per-fold metrics.
- `shap_fairness_calibration.py` — TreeExplainer + groupwise calibration (Market × Segment × Late_risk). Equalized-odds parity.
- `fix_benefit_regression.py` — MAE objective vs MSE (+13% improvement). Log-target tested.

## B.4 `20_past_self/` (7 files)

- `train_past_self.py` — 8 FRED × 3 horizons. 2,883 business days. 20-fold rolling-origin. Chronos predict_quantiles, TimesFM point only, Prophet weekly+yearly, ARIMA(5,1,0). Inverse-MAE weighting. Krippendorff α for ensemble disagreement.
- `r3_timesfm_residual_quantile.py` — Quantile regressors (P10/P50/P90) on TimesFM residuals.
- `r3_point_stacking.py` — Ridge stacking, alpha=1.0.
- `r3_constrained_stacking.py` — Non-negative weights, sum-to-one via LinearConstraint.
- `r3_bigtft_integration.py` — Custom BigTFT v2 as 5th member.
- `plot_r3_summary.py` — 8×3×4 MAE heatmap + dir-acc scatter + PICP coverage bars.
- `plot_timesfm_quantile.py` — Actual + point + P10/P90 bands per horizon.

## B.5 `30_dangerous/` (6 files)

- `r4_judge_layer.py` — 26 Wikipedia scenarios × 3 judges parallel via requests.post(OLLAMA_URL). JSON extraction. Krippendorff α (line 129-141), Jaccard semantic (144-149). Escalation rubric.
- `r4_v2_beast.py` — 26 × 4 LLMs. DeepSeek two-pass (free reasoning → Qwen extraction) bypasses CoT leakage. 100% parse. Latencies: DeepSeek 17s, Qwen 6s, Mistral 6.6s. Majority GT acc 69.2%. Cohen κ(Qwen, Mistral)=0.747.
- `r4_ablation_and_baseline.py` — Per-judge contribution.
- `r4_live_scenario.py` — Real-time invocation with latency tracking.
- `plot_r4_summary.py` — 3×3 agreement heatmap, confusion, latency boxplot, calibration.
- `plot_r4_v2.py` — Per-scenario escalation codes, semantic Jaccard heatmap.

## B.6 `40_granite/` (6 files)

- `r5_rag_beast.py` — 8 pipelines: P1-P3 bi (BGE-M3/mxbai/Snowflake), P4-P6 +reranker, P7 RRF, P8 HyDE. 6,483 chunks. chunk_words=256, overlap=32. top-k=50, rerank=10. 53 queries.
- `r5_mteb_subset.py` — MTEB external validation.
- `r5_hard_queries.py` — 20 hardest queries (low-recall analysis).
- `r5_manual_beir.py` — BEIR subset eval vs published numbers.
- `plot_r5_summary.py` — P@1/3/5/MRR per pipeline; mxbai (P2) wins P@1=0.962.
- `plot_r5_hard_redemption.py` — Latency vs MRR scatter showing trade-off.

## B.7 `50_gethsemane/` (9 files)

- `train_rl_beast.py` — MaskablePPO 3 tasks. Mask shape fix line 77-86 (concat type [7] + node [40] = 47). 100K timesteps. batch=256, lr=3e-4, γ=0.99, ent=0.01. 4 envs DummyVecEnv + VecNormalize.
- `r6_medium_300k.py` — 300K extended training on medium.
- `r6_unmasked_ablation.py` — Same env w/o masking; constraint violations tracked.
- `r6_unmasked_ablation_alltasks.py` — All 3 tasks unmasked.
- `plot_learning_curves.py` — Mean reward + std per 10K-step checkpoint.
- `plot_r6_gethsemane.py` — PPO vs random vs greedy box plots per task.
- `plot_masking_ablation.py` — Constraint violations: masked=0, unmasked ~5-20.
- `export_v3_ppo_onnx.py` — `.zip` → `.onnx` via skl2onnx/onnxruntime.
- `r6_algo_comparison.py` — PPO/RecurrentPPO/A2C/SAC-Discrete comparison.

## B.8 `60_euclidian/` (2 files)

- `r6_massive_benchmark.py` — 10,800-episode bootstrap. 3 tasks × 4 policies × 900 episodes. Per-100-episode rolling.
- `plot_r6_euclidian.py` — Episodes vs mean reward + 95% CI shaded bands per policy.

## B.9 `70_provider/` (3 files)

- `r6_gnn.py` — 3-layer GCN on real graphs (25+ nodes). Edges weighted by lead-time. 5-step disruption propagation. Adam, 1000 epochs, early stop.
- `r6_gnn_arrival_time.py` — Arrival-time impact regression.
- `plot_r6_provider.py` — Network with critical nodes + edge weights + disruption heatmap.

## B.10 `80_aqua_regia/` (3 files)

- `r6_conformal.py` — Split-conformal on Chronos + ARIMA. α ∈ {0.1, 0.05, 0.2}. Heteroscedastic via Ridge on calibration set.
- `r6_per_horizon_conformal.py` — Per-horizon α; uncertainty grows with h.
- `plot_r6_aqua_regia.py` — Coverage actual vs nominal, width vs horizon, per-target heatmap.

## B.11 `85_infinite_baths/`, `90_damocles/`, `95_arcadia/`, `utils/` (4 files)

- `85_infinite_baths/dashboard.py` — Streamlit aggregator. 6 tabs (R1-R6). Metric cards + plots + filterable tables. localhost:8501.
- `90_damocles/app.py` — FastAPI 4 endpoints (/assess, /forecast, /rag, /rl/act). JWT auth. Uvicorn 0.0.0.0:8765. Models pre-loaded.
- `95_arcadia/README.md` — Master doc. Architecture, phase log, decisions, instructions, hackathon thesis. Honest findings explicit.
- `utils/conformal_smoke_check.py` — Quick coverage validation on synthetic series.
- `utils/__init__.py` — empty.

## B.12 `results/` (45 files — the ledger)

**Verification (7)**: embedders_chronos, qwen_vl, tabpfn, timesfm, mistral_nemo, qwen14b, qwen_coder.

**Core (7)**:
- `R1_VERIFIED.json` — 13 SOTA all OK. RTX 4080 12GB, torch 2.5.1+cu121, free_disk=100GB. Q4_K_M, safetensors, torch backend.
- `R2_CARAMEL.json` — 4 tasks × 4 models. Late_delivery: XGB acc=0.8369 [0.832, 0.841] AUC=0.916 ECE=0.0837. LGB acc=0.8280 AUC=0.919. CAT acc=0.7983.
- `R3_PAST_SELF.json` — 8 targets × 3 horizons × 4 models. DCOILWTICO h7: Chronos MAE=2.792 dir=0.45 PICP80=0.693. TimesFM MAE=2.780 dir=0.628. ARIMA MAE=2.677 PICP80=0.721. Prophet MAE=8.496. Ensemble mean=1.455, weighted=2.350.
- `R4_DANGEROUS_V2.json` — 26 × 3 judges + critic. Tohoku: deepseek=HIGH/0.5, qwen=CRITICAL/0.95, mistral=CRITICAL/0.95 → majority CRITICAL → C_SUITE_IMMEDIATE. α=0.210, weighted κ=0.747.
- `R5_GRANITE.json` — 8 pipelines × 53 queries. Corpus: wiki 564, sec 5790, policy 129. P1_bge_m3: P@1=0.924/MRR=0.962/lat=48ms. **P2_mxbai: P@1=0.962/MRR=0.978/lat=35ms (BEST)**. P4-P6 reranked: P@3 drops to 0.862-0.868, latency 1.1-1.8s.
- `R6_GETHSEMANE.json` — 3 tasks × 3 policies × 50 eval. Easy: ppo=1.201±0.199, random=0.780, greedy=0.980, train=389s. Medium: ppo=2.775, random=-1.110, greedy=-1.796, train=1028s. Hard: ppo=2.674, random=-1.222, greedy=-1.413, train=1360s. Total 48.65min.
- `R6_EUCLIDIAN.json` — 10,800 ep bootstrap. Hard PPO CI95 [2.596, 2.708] vs greedy [-1.48, -1.35] non-overlapping.

**Secondary (11)**: R2_BENEFIT_FIX, R2_SHAP_FAIRNESS, R3_TIMESFM_QUANTILE, R3_STACKING_V2, R3_STACKING_V3_POINTLEVEL, R3_BIGTFT_INTEGRATION, R4_DANGEROUS_V1/V2_ABLATION/V2_LIVE/V2_HUMAN_BASELINE, R4_FRONTIER_PANEL_V2, per-judge JSONs, phase caches.

**Specialized (8)**: R5_GRANITE_HARD (20 paraphrased), R5_BEIR_MANUAL, R6_GETHSEMANE_MASKING_ABLATION (+26.8% / +15.1% / 13.64→0 violations), R6_GETHSEMANE_MASKING_ABLATION_ALLTASKS, R6_GETHSEMANE_ONNX_EXPORT, R6_PROVIDER (F1 0.964 hard, MAE −48/−49/−64% vs MLP), R6_PROVIDER_V2 (arrival-time), R6_AQUA_REGIA (conformal), R6_AQUA_REGIA_V2 (per-horizon), R6_ALGO_COMPARISON.

**Reports (2)**: `R4_DANGEROUS_V2_REPORT.md`, `R5_GRANITE_REPORT.md`.

**Metadata (2)**: `ONNX_BUNDLE_MANIFEST.json` (4 ONNX files: ppo_easy/medium/hard 948KB each, gcn_arrival 10KB), `R1_QWEN_VL_DOWNSTREAM.json`.

**Image (1)**: `r1_qwen_vl_test_image.png` — synthetic GOES-16 512×512.

## B.13 `checkpoints/` (42 files)

**Caramel (15 .pkl, 284 MB)**: per-task XGB/LGB/CatBoost models. Late_delivery: cat 8.3MB, lgb 13MB, xgb 16MB. Shipping_mode: 21/52/63MB. Delivery_status: 21/53/58MB. Benefit: 550KB-2.3MB.

**Gethsemane (15 zip/onnx, 74 MB)**: PPO easy/med/hard `.zip` 4.8MB each + variants (typhoon, UNMASKED), onnx 948KB. RecurrentPPO easy 20MB. A2C 3.2MB. MaskablePPO 4.8MB.

**Granite (2, ~600 MB)**: corpus_chunks.pkl (6,483 chunks), hyde_cache.json (Qwen-14B HyDE for 53 queries).

**Provider (6 .pt, 444 KB)**: gnn_easy/med/hard.pt + gnn_arrival_easy/med/hard.pt, 74KB each.

**ONNX bundle (4)**: ppo_easy/med/hard.onnx 948KB each, gcn_arrival.onnx 10KB.

## B.14 `plots/` (23 files)

- `make_hero_card.py` — 4-panel composite (R2 model, R3 ensemble heatmap, R4 risk dist, R5 P@1).
- `hero_result_card.png` — output card.
- `caramel/reliability.png` — calibration curves.
- `dangerous/r4_summary.png`, `r4v2_ablation.png`, `r4v2_calibration.png`, `r4v2_confusion.png`, `r4v2_escalation.png`, `r4v2_heatmap.png`, `r4v2_latency.png` — 7 R4 viz.
- `granite/r5_corpus.png`, `r5_hard_redemption.png`, `r5_latency_vs_mrr.png`, `r5_metrics.png`, `r5_per_query_heatmap.png` — 5 R5 viz.
- `gethsemane/learning_curves.png`, `r6_gethsemane.png`, `r6_masking_ablation.png` — 3 R6 RL viz.
- `past_self/r3_summary.png`, `r3_timesfm_quantile.png` — 2 R3 viz.
- `provider/r6_provider.png` — network graph.
- `euclidian/r6_euclidian.png` — bootstrap CI.
- `aqua_regia/r6_aqua_regia.png` — coverage.


# APPENDIX C — `ShAuRyA_Phoenix/` FILE INVENTORY (124 files)

## C.1 Root + framework

- `README.md` — Master Phoenix v5 doc. Three invariants (v3/v4 untouched, copy-before-edit, `.venv-roll/`). Phase 0/1/2 gates. Tag `v5.0-phoenix-ascensionism`.

## C.2 `receipts_v2/` (45 files: 20 receipt pairs + framework)

Framework: every claim → `{command, extraction, expected, actual, match, stdout_sha256, hardware, runtime_s, timestamp_utc}`.

**13 v4 carryovers**:
1. R5_GRANITE_mxbai_P1 = 0.9622
2. R5_GRANITE_mxbai_MRR = 0.9780
3. R5_BEIR_snowflake_nDCG10 = 0.971
4. R4_2JUDGE_Krippendorff_alpha = **0.7499**
5. R4_Cohen_kappa_QwenMistral = 0.747
6. R6_MaskingAblation_easy_lift = 26.77%
7. R6_GCN_easy_MAE_vs_MLP = 48.0247%
8. R6_AquaRegia_WTI_dev95 = 0.0238
9. R3_TimesFM_CP_WTI_dev95 = 0.050
10. V4_SPOF_V2_F1 = 1.0
11. V4_STACKING_V2_lift_vs_WV = ≤ 0.001
12. V4_Live_Brent_202604 = in [60, 250] USD/bbl
13. V4_Tests_Total = 249 passed

**7 v5 new** (all matched 2026-04-22):
14. V5_Autoresearch_best_experiment = `s3_curriculum_learning`
15. V5_Autoresearch_CI95_lift = +0.0967 ≥ +0.05
16. V5_Arena_baseline_leaderboard = 6 baselines, MaskablePPO #1
17. V5_Twin_savings_gt_zero = $178,684,200 (48% pct)
18. V5_DPO_JUDGE_preference_pairs_built = 21 ≥ 20
19. V5_Skill_pack_shipped = 4 files (3 SKILL.md + plugin.json)
20. V5_Phoenix_tests_green = 15 passed (1 expected fail on autoresearch state coherence)

Framework files:
- `framework.py` (271 lines) — Receipt class, run() executor, SHA256, 5 comparators (==, >=, <=, in_range, regex), tiny YAML parser (no PyYAML dep).
- `register.py` (284 lines) — Registry split V4_CARRYOVERS (13) + V5_NEW (7). stub_all(), regenerate(), build_index().
- `INDEX.json` — 20-entry array.
- `INDEX.md` — Human-readable table. "Total receipts: 20 | v4: 13 | v5: 7".

## C.3 `autoresearch_fixed/` (21 files)

- `state.json` — Master ledger rebuilt 2026-04-22T06:51:52Z. Best `s3_curriculum_learning` mean=0.646 / CI95_lower=0.5515. 5 experiments:
  - **s1_bigger_network** ACCEPTED — [256,256]+ReLU, lr=3e-4, ent=0.01. Mean=0.5841, CI95=[0.4035, 0.7390]. Wall 122.68s. First baseline.
  - **s2_higher_entropy** ACCEPTED — ent=0.1. Mean=0.6066, CI95=[0.4548, 0.7520]. Wall 135.79s. Δ +0.0513.
  - **s3_curriculum_learning** ACCEPTED — [128,128] easy→med→hard 40/30/30. Mean=0.646, CI95=[0.5515, 0.7610]. Wall 216.85s. Δ +0.0967 (BEST). Phoenix fix: save→load instead of set_env (MaskablePPO action_dims caching).
  - **s4_recurrent_ppo** REJECTED — LSTM-128. Mean=0.301, CI95=[0.2583, 0.3329]. Wall 193.97s. Δ −0.2932.
  - **s5_action_diversity** REJECTED — k=5 bonus=0.02. Mean=0.6574, CI95=[0.5528, 0.7588]. Wall 129.73s. Δ +0.0013 (noise).

- `README.md` — Karpathy autoresearch loop: program.md → LLM agent → candidate_train.py diff → fixed-budget runner → CI95 evaluator → accept/reject → notebook. Safety guards: 10-min wall, OOM/NaN guards, test gate, ≤150 LOC diff, signature lock.
- `lab_notebook.md` — S1 (bimodal flagged), S2 (+0.07 medium-task lift), S3 (Phoenix save→load fix), S4 (recurrent rejected), S5 (diversity rejected).
- 5× `experiments/{name}/result.json` — grader_scores, wall_clock, total_steps, architecture summary, training_seed.
- `program.md`, `candidate_train.py`, `hypothesis_engine.py`, `evaluator.py`, `orchestrator.py`, `runner.py`, `seed_experiments.py`, `rebuild_state.py`, `rerun_seeds.py`, `__init__.py` — Framework components.
- `seed1000_candidate/`, `seed1001_candidate/` — checkpoint backups.

## C.4 `arena/` (5 files)

- `runner.py` — Policy eval harness. Accepts sb3.PPO, sb3_contrib.MaskablePPO, or torch.load nn.Module. TaskResult + ArenaResult dataclasses. Defaults: 50 ep/task, 200 max steps.
- `leaderboard.py` (99 lines) — 6 baselines: MaskablePPO-v3 (mean=2.209, CI95=[2.178, 2.239], v=0), RecurrentPPO-v3 (1.081, [0.98, 1.18], 14.9), PPO-v3 unmasked (0.947, [0.89, 1.01], 13.6), A2C-v3 (0.874, [0.81, 0.94], 13.9), Greedy (-0.749, 0), Random (-0.511, 0).
- `leaderboard.json` — generated 2026-04-22T07:09:06Z. n_submissions=0, n_baselines=6.
- `gradio_app.py`, `router.py`, `__init__.py` — Arena API.

## C.5 `counterfactual_twin/` (3 files)

- `twin.py` — 100-rollout MC with severity ∈ [0,1] + brent_usd inputs. REVENUE_AT_RISK_USD: easy $200M / med $320M / hard $400M. TwinReport with median/p95/savings/CI95/pct.

  Receipt run actual: hard task, severity=0.85, brent=$123, 30 rollouts → trained median $187.38M, no_action $366.77M, **savings $178.68M [177.74M, 179.52M], 48% pct**, generated 2026-04-22T20:37:23Z.

- `router.py`, `__init__.py` — API.

## C.6 `roll_integration/` (16 files)

- `README.md` — 3 paths: DPO judge fine-tune (with trl fallback), SupplyMind as ROLL env, 3-judge reward bridge. Qwen-2.5-3B with LoRA r=8. Expected delta +5 to +15 pp R4 acc.
- `configs/dpo_qwen25_3b_supplymind.yaml` — Qwen/Qwen2.5-3B-Instruct. dpo_beta=0.1, sigmoid loss. LoRA r=8 α=16. strategy=hf (NOT megatron). save_adapter_only (~20MB).
- `configs/agentic_supplymind_gigpo.yaml` — GiGPO step-wise. env=supplymind_crisis. 3 tools: forecast, rag, rl_act. reward=supplymind_3judge.
- `dpo_judge/prepare_preference_data.py` — Build pairs from R4 GT vs DeepSeek/Mistral/Qwen judges. Receipt actual=21 pairs.
- `dpo_judge/train_dpo_trl.py`, `train_dpo_roll.py`, `train_grpo_env.py`, `train_grpo_live_env.py`, `evaluate_delta.py` — Training pipeline.
- `reward_bridge/supplymind_judge_worker.py` — LLMJudgeRewardWorker subclass. Class exists even without ROLL.
- `trl_fallback/README.md` — Standalone DPO fallback.
- `__init__.py`.

## C.7 `supplymind_skills/` (5 files)

- `plugin.json` — Claude Code skill pack manifest. v1.0.0. 3 skills.
- `autoresearch-experiment/SKILL.md` (132 lines) — Iron law: "ONE MUTABLE FILE. ONE METRIC. BOOTSTRAP CI95. NOTEBOOK BEFORE DECISION." 6-file setup. Loop: orchestrator → mutator → runner → evaluator (Δ > 0.005 CI95 lower).
- `benchmark-runner/SKILL.md` (113 lines) — Iron law: "NO PERFORMANCE CLAIM WITHOUT A PAIRED BENCHMARK RECEIPT." RED→GREEN cycle. 5 stages.
- `live-demo-orchestrator/SKILL.md` (127 lines) — Iron law: "EVERY LIVE FEATURE HAS AN OFFLINE REPLAY." 3 phases: pre-demo (10-item checklist), during (4-step recovery), post (receipt).
- `README.md` — Skill pack overview.

## C.8 `realtime_v5/` (5 files)

- `replay_cache_latest.json` — n_events=8. Per event: scenario_input, top_analog (similarity ≥0.99), risk_level, confidence, recommended_actions, escalation_tier, counterfactual (no_action_loss, with_plan_loss, savings, savings_pct), oil_impact_usd_bbl, judges (qwen/mistral/deepseek).
- `replay_cache_2026_04_22.json` — Timestamped snapshot.
- `freeze_cache.py`, `replay_adapter.py`, `__init__.py` — Cache mgmt. status() returns cache_exists=True, n_events≥8.

## C.9 `server/` (2 files)

- `phoenix_app.py` — FastAPI v5 entry. Imports v4's server.app (frozen). Mounts /arena, /twin, /replay routers. Graceful degradation. /phoenix/status endpoint.
- `__init__.py`.

## C.10 `upstream_prs/` (6 files)

- `meta_openenv/PR.md` — Title: "Add SupplyMind: real-data supply-chain risk env (3 tasks, Pydantic-v2, Docker)". 8-item compliance checklist. openenv-core ≥0.2.0.
- `meta_openenv/README.supplymind.md`, `build_pr_branch.sh` — Supporting.
- `alibaba_roll/PR.md` — Title: "Add examples/supplymind_crisis: agentic RL (GiGPO, 3-judge reward, Qwen-2.5)". Single-GPU 12GB ergonomics, LoRA, LLMJudgeRewardWorker.
- `alibaba_roll/README.crisis.md`, `build_pr_branch.sh` — Supporting.

## C.11 `experiments/` (4 files)

- `arena/leaderboard.json` — generated 2026-04-22T07:09:06Z. 6 baselines.
- `dpo_judge_v1/train_dpo.sh` — Shell.
- `roll_install/phase_a.sh` — Windows-native ROLL install.
- `twin/V5_receipt_run.json` — Twin receipt output.

## C.12 `docs/` (8 files)

- `README_V5_OPENENV_FIRST.md` — Architecture overview.
- `JUDGES_V5.md` — 4-min judges quick reference. 30-sec pitch (13 local, 261K data, 256+ tests, 20 receipts, live pipeline, autoresearch, DPO, Arena, upstream PRs). Live demo + Arena CLI.
- `PREPRINT_V5.md` — Full preprint. Abstract, env design, foundation stack.
- `PITCH_DECK_V5.md` — 8-slide keynote. Slide 3 headline numbers (mxbai 0.9622, Snowflake 0.971, α 0.7499, masking +26.77%, GCN −48%, conformal 0.024, 249 tests, autoresearch +0.051).
- `DEMO_VIDEO_SCRIPT_V5.md`, `PHOENIX_COMPLETION_AUDIT.md`, `PHOENIX_PUSH_REPORT.md` — Supporting.

## C.13 `tests/` (2 files)

- `test_smoke.py` (172 lines) — 16 smoke tests <10s no GPU/Ollama:
  1. test_phoenix_skeleton_exists (12 subdirs)
  2. test_receipts_indexed (≥20)
  3. test_autoresearch_state_coherent (best in {s2, s3}, CI95≥0.45)
  4. test_replay_cache_built (≥8 events)
  5. test_skill_pack_complete (3 SKILL.md + frontmatter)
  6. test_receipt_framework_importable
  7. test_arena_leaderboard_importable (≥6 baselines, MaskablePPO present)
  8. test_arena_runner_importable
  9. test_twin_importable
  10. test_dpo_judge_preference_builder_importable
  11. test_roll_env_wrapper_importable (skip if missing deps)
  12. test_reward_bridge_importable_without_roll
  13. test_replay_adapter_status
  14. test_phoenix_app_builds
  15. test_upstream_pr_drafts_present
  16. test_docs_suite_complete (5 files >500 chars)
- `__init__.py`.

## C.14 `scripts/` (1 file)

- `push_all_upstream.sh` — Convenience script.


# APPENDIX D — `ShAuRyA_Supplymind/` FILE INVENTORY (122 files)

## D.1 Root (4 files)

- `README.md` — v4.0-arcadia-live staging. Phase map L1-L5. Finals Apr 25-26 Bangalore.
- `__init__.py` — `"""ShAuRyA_Supplymind — v4.0-arcadia-live staging directory."""`
- `RELEASE_NOTES_V4.md` — 2026-04-21. v3.0-arcadia frozen. Adds Karpathy autoresearch, live geo pipeline, 8 Hormuz events, 15 features F1-F10 + G-fixes, 76 new tests = 249 total.
- `RELEASE_V4_TAG.md` — GitHub Release. Tag `v4.0-arcadia-live` (Sleep Token "Rain"). v5 80% acc, 5 seeds, Qwen-VL 7 ports, stacking +0.0045, SPOF 1.000, FRED Brent $123.28, 250 tests.

## D.2 `features/` (35 files)

**JSONs (9 + 3 subdirs)**:

- `F2_MULTI_AGENT_DEMO.json` — 2021 chip auction. Apple +$2.74M (615.4 wafers), Samsung -$11.5M, Toyota -$7.37M. 2-phase bidding 1000 wafers @ $16.5K base.
- `F4_DT_RISK_SLIDER.json` — DT slider conservative/balanced/aggressive returns 0.6196 / 0.765 / 0.765 on easy. Conservative 55% do_nothing. Balanced 55% safety_stock. Aggressive 50% hedge. Wall 1.0s.
- `F6_CONFORMAL_RL.json` — 3 α levels (0.05, 0.05, 0.1). 5 actions, intervals [lo, hi]. CI95 width range 1.447-2.361. Conservative/balanced abstain on width >0.5.
- `F9_PARETO_CARBON.json` — 20 plans, 11 Pareto. Carbon factors: air 0.82, express_sea 0.02, sea 0.013, rail 0.028, road 0.096 kg-CO2/tonne-km. Best under 3 weight schemes: reroute_rail_panama ($180K, 70 res, 0 carbon).
- `F14_CUDA_KERNEL.json` — Win-10 CUDA 12.1, torch 2.5.1+cu121. JIT failed (MSVC blocker). PyTorch fallback 0.0284ms vs naive 52.16ms = 1833.7× speedup. No JIT results — partial.
- `R6_SPOF_V2.json` — 3 graphs (12/25/40 nodes). v1 mean F1=0.949, v2=1.000. Hard fixed (0.846→1.000). Top-5 SPOFs per graph with mitigation (FAC_PHOENIX 3 downstream, WH_TAIWAN 10 downstream, TSMC $18B at risk).
- `R9_ANALYST_AB_V5.json` — v5 vs base Qwen on 10 scenarios. **v5: 80% exact, 90% partial, 91.7% evidence**. Base: 0% / 5% / 0%. Lifts: +0.80 / +0.85. Latencies 12-27s. Hormuz_2026_04 → CRITICAL exact.
- `R15_STACKING_V2.json` — 60K DataCo, 6 base learners. **LGB best single AUC 0.9818**. Stacking AUC 0.9816. Stacking +0.0045 vs WV (0.9771), −0.0002 vs best single (honest null at 0.97+ ceiling).
- `counterfactual_cache.json` — 7 memoized counterfactuals (no LLM). hedge_commodity saves 40% of $36.9M (Iran), reroute_shipment saves 60% of $76.5M (Houthi).

**`gcn_attn/` (4 files)**:
- `SUMMARY.json` — Easy: target FAC_PHOENIX, top edge PORT_LONG_BEACH→WH_US_WEST grad 0.8625. Medium: target FAC_SUZHOU, top edge WH_THAILAND→FAC_SUZHOU 0.9. Hard: target FAC_TOYOTA_AICHI, WH_JAPAN→PORT_YOKOHAMA 0.6.
- `gcn_attn_easy_graph.json` + `.png` 132KB.
- `gcn_attn_medium_graph.json` + `.png` 247KB.
- `gcn_attn_hard_graph.json` + `.png` 351KB.

**`port_imagery/`** (1 file):
- `assessments.json` — 7 critical ports: Kaohsiung, Shanghai, Long Beach, Rotterdam, Jebel Ali, Haifa, Hodeidah. Risk 0.501-0.502. Confidence 0.35 (heuristic mode). Highest-risk: Shanghai.

**`provenance/`** (1 file):
- `demo.json` — Query "Why is TSMC SPOF for advanced semis?". 5 chunks. 5-tier classifier (regulatory 1.0, academic 0.5, reference 0.333, industry 0.25). Provenance score 0.47.

**Python drivers (16 files)**:
- `receipts.py` — F10 receipt system with RECEIPT_SPECS list.
- `multi_agent_demo.py` — F2/G4 auction simulator. Agent + Auction classes.
- `dt_risk_slider.py` — F4/G6 DT surrogate. Target return ∈ {0.3, 0.55, 0.8}. Saves F4 JSON.
- `spof_v2.py` — G8 articulation_points() on 3 graphs. Saves R6 JSON.
- `stacking_v2.py` — G15 5-fold CV + Ridge meta. Saves R15 JSON.
- `analyst_ab_bench.py` — G9 A/B harness, deterministic rubric, Ollama+stub fallback.
- `conformal_rl.py` — F6 split-conformal Q-value intervals.
- `pareto_carbon.py` — F9 multi-objective optimizer.
- `rag_provenance.py` — F8 5-tier classifier.
- `counterfactual_explainer.py` — F3 7 templates with historical analog.
- `leaderboard.py` — F5 Gradio submission scoreboard.
- `gcn_attention_viz.py` — F7 attention gradient → PNG.
- `qwen_vl_port_imagery.py` — G3+F1 7-port heuristic risk.
- `cuda_kernel_verify.py` — G14 JIT attempt + fallback benchmark.
- `lora_train.py` — G7 PEFT 4-bit NF4 dry-run.
- `Modelfile.analyst_v5` — Ollama Modelfile for v5 (8 hard-negative few-shots, calibrated system prompt).

## D.3 `autoresearch/` (14 files)

- `orchestrator.py` — Karpathy loop. propose → apply → run → evaluate → accept/reject → log. _parse_budget (6h→21600s), _load_state, _save_state, _history_summaries (last 20 only). MAX_CONSECUTIVE_REJECTS=50.
- `state.json` — **Best: s3_curriculum_learning_rerun**, CI95_lower=0.5514, mean=0.646, std=0.1634 (9 runs). MaskablePPO [128,128] curriculum 40/30/30. 10 history entries.
- `AUTORESEARCH_LAB_NOTEBOOK.md` — 3 accepted: s1 (CI95 +0.4035, 125.4s), s2 (+0.0513, 138.7s), s3_rerun (+0.0966, 219.7s, BEST).
- `candidate_train.py` — Mutable training script with safe-to-modify markers.
- `hypothesis_engine.py` — LLM agent (Qwen-14B local or Claude). Reads last-20 + program.md.
- `runner.py` — Isolated subprocess. 10-min timeout, VRAM guard, NaN catch, test gate.
- `evaluator.py` — Bootstrap CI95 lower on 9 grader scores. Threshold Δ > 0.005.
- `lab_notebook.py` — Markdown auto-logger.
- `seed_experiments.py` — 5 hand-crafted seeds: s1-s5.
- `rerun_seeds.py` — Rerun failed seeds.
- `program.md` — Task spec + frozen metric (CI95 lower on hard 50K steps).
- `README.md` — Module entry.
- `AUTORESEARCH_REJECTED.md` — Rejected hypotheses log.
- `__init__.py`.

## D.4 `realtime/` (11 files + sources/)

- `store.py` — SQLite event store. Schema: events (id, source, ts_iso, ts_unix, event_type, severity, region, raw_text, text_hash, urls, entities, meta_json, ingested_at). Indices: (source, text_hash), ts_unix, region, event_type. Dedup 24h.
- `hormuz_endpoint.py` — `/live/*` FastAPI router. 5 endpoints: GET /health, /recent-events, /signal-counts; POST /hormuz-closure (main), /analog-match. Pipeline: 24h events → match scenario → interpolate Brent → 3-judge panel → action recommendations.
- `crisis_library.py` — Loads 8 events, embeds via mxbai. find_analogs(), interpolate_projection().
- `ingestor.py` — Polls 5 sources, dedupes. 150+ events on 2026-04-21 (80 NewsAPI, 60 GDELT, 19 USGS, 1 FRED).
- `sources/newsapi.py` — 5 regional queries, 7-day lookback.
- `sources/gdelt.py` — GDELT 2.0 Doc API, 15-min refresh, tone severity.
- `sources/usgs.py` — M4.5+/24h, 6 region boxes.
- `sources/fred_brent.py` — DCOILBRENTEU daily.
- `sources/marinetraffic.py` — AIS snapshot, fallback to JSON if no key.
- `sources/__init__.py`.
- `events.db` — SQLite binary (schema only).
- `__init__.py`.

## D.5 `scenarios/` (1 file)

`iran_israel_hormuz_2024_2026.json` — schema v1.0, 8 events:

1. **iran_true_promise_1_2024_04** (2024-04-13) — sev 0.80, Brent 90.7→92.2→87.3, reroute 2d, 3 cites.
2. **iran_true_promise_2_2024_10** (2024-10-01) — sev 0.90, Brent 71.8→78.2→74.4, reroute 3d, 3 cites.
3. **houthi_red_sea_campaign_2023_ongoing** (884d) — sev 0.85, Brent 82.1→92.2 peak, Suez −50%, Tesla Berlin paused 2024-01, reroute 12d, 4 cites.
4. **us_uk_operation_poseidon_archer_2024_01** (2d) — sev 0.65, Brent 77.6→81.0→78.2, reroute 1d, 3 cites.
5. **haifa_port_missile_2024_10** (24d) — sev 0.60, Brent 74.2→78.2→75.5, war-risk +50-100bps, Tower Semi delays, 3 cites.
6. **houthi_yaffa_tel_aviv_2024_07** (3d) — sev 0.70, Brent 85.4→87.1→85.9, reroute 2d, 3 cites.
7. **hormuz_trump_cargo_ship_2026_04** (4d, LIVE) — sev 0.82, Brent 119.1→123.3→P95=168, 20%+ global crude through Hormuz, reroute 14d, 4 cites.
8. **ukraine_neon_palladium_shock_2022_context** (310d) — sev 0.88, Brent 96.8→127.6 peak→104.9, neon 70% / palladium 37% / nickel +250%, TSMC/Samsung/Intel +3mo lead, 3 cites.

## D.6 `tests/` (18 files, 76 new v4 tests)

- `test_hormuz_endpoint.py` (60 lines, 8 tests): library_has_eight_events, every_event_has_required_fields, analog_match_finds_hormuz_event, projection_interpolation, endpoint_scenario_call, endpoint_fallback_without_ollama, endpoint_counterfactual_math, endpoint_live_signal_join.
- `test_receipts.py` (45 lines, 4 tests): receipts_dir_exists, receipt_specs_are_structured (≥10), jqlike_helper_generates_python_snippet, receipt_dataclass_serializes.
- `test_spof_v2.py` — F1=1.000 on all 3 graphs.
- `test_stacking_v2.py` — 5-fold CV, OOF shape, AUC 0.9816 vs WV 0.9771, best-single tiebreak.
- `test_analyst_ab_bench.py` — 10 scenarios, exact 0.8, partial 0.9, evidence 0.917.
- `test_conformal_rl.py` — 3 α levels, CI95 nominal coverage.
- `test_pareto_carbon.py` — 20 plans→11 Pareto, weighted scheme→reroute_rail_panama.
- `test_dt_risk_slider.py` — 3 positions, ~1s wall, DT flag set.
- `test_multi_agent_demo.py` — 3 agents, Apple wins, 2-phase, PnL.
- `test_gcn_attention_viz.py` — PNG outputs, JSON, top edges.
- `test_qwen_vl_port_imagery.py` — 7 ports, 0.50-0.51 risk, 0.35 confidence.
- `test_cuda_kernel_verify.py` — JIT log, fallback 0.028ms, naive 52.16ms.
- `test_lora_train.py` — Dry-run validation.
- `test_rag_provenance.py` — 5 chunks, 5-tier, score.
- `test_counterfactual_explainer.py` — 7 templates verified.
- `test_leaderboard.py` — HTTP submission reachable.
- `test_server_live_router.py` — /live/health 200, recent-events, signal-counts.
- `__init__.py`.

## D.7 `docs/` (5 files)

- `LIVE_DEMO_HORMUZ.md` (96 lines) — 90-second demo. Hook, pre-demo (server, ingestor 150 events, health), 3-command demo (scenario → analog 0.99 → 3-judge HIGH/CRITICAL → Brent $110-125 → 5 actions → counterfactual $324M→$65M = 80% / $259M).
- `PREPRINT.md` (95 lines) — Arxiv-style. 261,175 data, 8 sources, 3 sequential releases, v4 adds 12 contributions, explicit limitations, 5-step reproducibility.
- `EXTERNAL_OUTREACH.md` — G11 LinkedIn/press playbook + 3 email templates.
- `SECRETS_ROTATION.md` — G12 .env hygiene.
- `PHOENIX_PLAN_V5.md` — Post-release roadmap.

## D.8 `deploy/` (2 files)

- `HF_DEPLOY_V4.md` (69 lines) — One-command deploy. pytest → `git push hf main` → 5-8 min build → smoke test (/health, /tasks, /reset, /live/health, /live/hormuz-closure, /docs, GitHub Release). 7-item checklist. <2GB container, 6-8 min build, 15-25s cold start, 2-3GB steady RAM.
- `PITCH_DECK_V4.md` — Investor outline.

## D.9 `receipts/` (32 files: INDEX + 15 .reproduce.sh + receipts)

`INDEX.json` — 15 receipts as PART 2 Phase 5 table. `INDEX.md` — copy-paste table. 13× `.reproduce.sh` one-liners using `python -c "json.load"` or sqlite queries.


# APPENDIX E — `scripts/`, `benchmark/`, `tests/`, `dashboard/`, `notebooks/`, `demo/`, `docs/`, `_dump/`, `.github/`, `client/`, `challenges/`, `server/`, ROOT (242 files)

## E.1 `scripts/` (45 files)

**Top-level (9)**:
- `openrouter_client.py` (11.7 KB) — Async HTTP client. 18+ frontier LLMs. Token-bucket 18 req/min. Exp backoff on 429. OPENROUTER_API_KEY env. Logs to .openrouter_usage.jsonl.
- `run_frontier_judge_panel.py` (11.2 KB) — 18-model panel on 26 scenarios. JSON parse + Krippendorff α + Cohen κ + ECE + majority vote. Saves R4_FRONTIER_*.json.
- `compute_panel_agreement.py` (7.5 KB) — Krippendorff ordinal α + Fleiss κ + Cohen weighted κ + pairwise confusion. Publishes API. Saves R4_DANGEROUS_V2_ABLATION.json.
- `verify_openrouter_models.py` (2.4 KB) — Smoke test endpoints, validates slugs.
- `build_pitch_html.py` (3.2 KB) — demo/PITCH_DECK.md → SupplyMind_pitch.html (reveal.js).
- `check_benchmarks.py` (2.3 KB) — Validates all benchmark/results/ JSONs against schema.
- `run_all.py` (4.0 KB) — Master: pytest → benchmark → OpenRouter panel → statistics → visualize → report.
- `export_all_onnx.py` (7.3 KB) — Exports all RL policies (MaskablePPO, QR-DQN, DT) to ONNX 0.97MB each, roundtrip-verifies.
- `reproduce.md` — Setup-to-verification guide.

**`scripts/legacy/` (~36 files)** — Pre-v3 deprecated scripts, training_report_v2.json, phase_b_results.json, etc. Preserved for audit trail.

## E.2 `benchmark/` (36 files)

**Top-level (7)**:
- `__init__.py` — empty.
- `run_benchmark.py` — 9 agents × 3 tasks × 5 seeds × 20 ep = 300 ep/agent. CSV + Wilcoxon.
- `run_full_benchmark.py` — Extended + MLflow tracking. Dual eval (Gymnasium reward + grader score).
- `ablation.py` — Component contribution. Progressive disclosure.
- `backtesting.py` — Calibration error vs real 2021 chip / 2021 Suez / 2023 Red Sea.
- `visualize.py` — Plots all results.
- `statistics.py` — Wilcoxon signed-rank + Friedman + bootstrap CI95.

**`crisis_library/` (5 files)**:
- `tohoku_2011.json` — M9.0, 2011-03-11, 600K+ deaths, 6-mo disruption, automotive + semi nodes, 180-day recovery.
- `suez_2021.json` — Ever Given grounding, 6 days, $9.6B/day, regional Suez, 1-2 wk recovery.
- `chip_shortage_2020.json` — Multi-year, sev 0.85, 180+ days, CRITICAL.
- `ukraine_neon_2022.json` — Neon 45-65% global, 2022-02 invasion, geopolitical, semi-affected.
- `red_sea_2023.json` — Houthi attacks, 3500+ nm Cape, +10 days, +25% fuel, +200-300% rates.

**`results/` (~14 files)**:
- `FINAL_RESULTS.json` — v1 sim, 300 ep/agent. QR-DQN 0.793 avg, Wilcoxon p < 1e-50.
- `GRAND_BENCHMARK_V2.json` — v2 real DataCo. Offline-RL + v1 QR-DQN.
- `R3_PAST_SELF.json` — 4-forecaster ensemble, 20-fold backtest (also in v3_arcadia/results).
- `R4_DANGEROUS_V2.json` — 3-judge + critic, parse 100%, α/κ/ECE/GT acc 69.2%.
- `R4_DANGEROUS_V2_ABLATION.json` — 2-judge α=0.750, rubric baseline 61.5%, 3-judge α=0.210.
- `R5_GRANITE.json` + `R5_GRANITE_HARD.json` + `R5_BEIR_MANUAL.json` — RAG.
- `R6_GETHSEMANE.json` + `R6_GETHSEMANE_MASKING_ABLATION.json` + `R6_EUCLIDIAN.json` + `R6_PROVIDER_V2.json` + `R6_AQUA_REGIA_V2.json` — RL/GNN/conformal.
- `backtesting_results.json` — Calibration error % on 3 historical crises.
- `statistical_tests.json`, `PAIRWISE_WILCOXON_V2.json`, `STATS_V2.json` — Statistical machinery.
- `real_world_benchmark.json` — v2 DataCo eval on 27K held-out.
- `fast_mc_benchmark.json` — Numba JIT MC <0.01ms warm.

**`legacy/` (~9 files)**: BENCHMARK_REAL_V2.json, AB_ANALYST_V3.json, REAL_DATA_PIPELINE.json, FINAL_RESULTS_simulated_V1.json + README. Deprecation notes.

## E.3 `tests/` (12 files, 173 passing)

- `test_openenv_compliance.py` — 19 formal checks: Pydantic v2 models, openenv.yaml, HTTP endpoints, MCP+WS, reproducibility (5× same-seed), dense reward, action validation, episode termination.
- `test_engine.py` — Sim engine: graph init, disruption lifecycle, financial impacts, reward calc, MC projection.
- `test_graders.py` — Episode grading: action_coverage, active_mitigation, cost_efficiency, health_score, SLA.
- `test_reward_hacking_adversarial.py` — 6 attacks all rejected. Receipt at tests/receipts/adversarial_reward_audit.json.
- `test_models.py` — Pydantic v2 validation.
- `test_server.py` — FastAPI endpoints.
- `test_tasks.py` — Registry: 3 tasks load correctly.
- `test_upgrades.py` — v2→v3 backward compat.
- `__init__.py`.

**`tests/receipts/` (3)**: openrouter_liveness.json, adversarial_reward_audit.json (8/8 rejected), frontier_panel_alpha.json.

## E.4 `dashboard/` (5 files)

- `app.py` (~500 lines) — Streamlit 12-panel: network graph (Plotly), return distribution violin, counterfactual panel, agent reasoning log (Ollama), agent comparison, risk-appetite slider (DT RTG), SHAP, TFT fan chart, what-if builder, live ingestion, GNN attention, Pareto frontier.
- `scenario_builder.py` (~100 lines) — What-if: crisis dropdown (7 types), severity slider, region dropdown, duration slider. 10 pre-cached templates.
- `crisis_ingestion.py` (~100 lines) — LiveCrisis feeder. NewsAPI cached, real-time risk update, RL response, counterfactual vs LLM agent dollar-difference.
- `app.py.SHIM_NOTICE.md` — Shimmed to v3 Damocles API.
- `__init__.py`.

## E.5 `notebooks/` (6 files)

- `01_environment_quickstart.ipynb` — Hello-world: reset 3 tasks, step loop, observation structure. <10 min CPU.
- `02_training_your_own_agent.ipynb` — PPO from scratch: Gymnasium wrapper, SubprocVecEnv, sb3, hyperparams.
- `03_reproducing_benchmarks.ipynb` — Exact code per number with seeds. Loads JSONs, plots, statistical tests.
- `04_v3_quickstart_colab.ipynb` — v3 walkthrough.
- `05_v4_hormuz_live.ipynb` — **THE HEADLINE DEMO**. 11 cells, ~5 min Colab CPU, no GPU/Ollama/keys. Live Hormuz pipeline → 0.99 analog → 5 actions → 8 receipts verified → smoke test.
- `06_trl_training_colab.ipynb` — TRL DPOTrainer on 21 preference pairs, Qwen-2.5-0.5B, Unsloth FastLanguageModel 4-bit NF4, <15 min free T4.

## E.6 `demo/` (7 files)

- `DEMO_VIDEO_SCRIPT.md` (130 lines) — 8-scene 3-min: Hook (Suez $9.6B/day), Stack (ollama list, torch.cuda, mxbai), 4 endpoints (/assess, /forecast, /rag, /rl/act), Dashboard, CTA. Timestamps 0:00-3:00.
- `DEMO_TRANSCRIPT.md` — Read-aloud transcript.
- `PITCH_DECK.md` — 5-slide MD: problem ($184B, 94% F1000), solution (13 SOTA local), benchmarks (0.971 nDCG, α=0.75), honest findings, CTA.
- `LANDING_PAGE.md` — 1-page HF Space intro.
- `CHECKLIST.md` — Pre-demo: Ollama running, models loaded, FastAPI up, Streamlit up, API healthy, network graph renders, video plays.
- `social.md` — Twitter/LinkedIn/HN drafts.
- `SupplyMind_pitch.html` — Rendered pitch.

## E.7 `docs/` (12 files)

- `CLONE_AND_STUDY.md` — Onboarding for fresh clones.
- `FINAL_AUDIT_REPORT.md` — Honest component-by-component audit + action items.
- `MULTI_TURN_GRPO_ROADMAP.md` — Future v4 multi-turn GRPO design.
- `legacy/supplymind_plan.md` — v1 design (archived).
- `legacy/REPORT_REAL_DATA.md` — v2 real-data report (DataCo 180K).
- `legacy/REPORT_SIMULATED_DATA.md` — v1 sim report (QR-DQN 0.793).
- `legacy/REPORT_REAL_V2.md` — v2 production retrain (CQL_real_v2 37.4%).
- `legacy/adaptive-tickling-bubble.md` — v1 narrative.
- `legacy/AUTORESEARCH_SUMMARY.md` — v2 autoresearch.
- `legacy/MODEL_CARD_V2.md` + `MODEL_CARD_REAL.md` — v2 cards (archived).
- `legacy/README.md` — v2 README.

## E.8 `_dump/` (4 files)

- `FAILURE_TABLE.md` — 22-item ledger: 8 resolved + 6 deferred + 8 honest negatives.
- `R2_TABPFN_BAGGING_DEMO.json` — Demo bagging output.
- `R6_GETHSEMANE_MEDIUM_300K.json` — 300K extended training results.
- (1 misc).

## E.9 `.github/` (3 files)

- `workflows/ci.yml` — pytest tests/ on push, OpenEnv compliance check.
- `workflows/benchmark-regression.yml` — Nightly benchmark, floor breach email.
- `workflows/deploy-hf-space.yml` — Deploy to HF on tagged release.

## E.10 `client/` (2 files)

- `supplymind_client.py` — HTTP client for deployed server. Zero `from server` imports. health/reset/step/grade.
- `__init__.py`.

## E.11 `challenges/` (1 file)

- `R4_RUBRIC_CHALLENGE.md` (111 lines) — Open challenge: match 2-judge α=0.750 / 61.5% acc on 26 Wikipedia crises. Deterministic GT rubric. PR submission with solution.py + results.json + README. Reference impl in v3_arcadia/30_dangerous/.

## E.12 `server/` (26 files)

**Top-level (5)**:
- `app.py` (145 lines) — FastAPI. Lifespan (pre-warm), CORS, endpoints (/health, /reset, /step, /state, /tasks, /grader, /baseline). v4 router mount.
- `supply_environment.py` (80 lines) — SupplyMindEnvironment class. reset/step/grade. Episode history.
- `openenv_adapter.py` — OpenEnv v2 interface. WebSocket + MCP routes.
- `integrated_agent.py` — Agent interface for scripted baseline.
- `__init__.py`.

**`engine/` (6 files)**:
- `simulation.py` (60 lines) — SimulationEngine orchestrator.
- `graph.py` — SupplyChainGraph. Node types (supplier/warehouse/port/factory/customer). Lead-time edges. Backup relationships. BFS propagation with 0.20 decay.
- `disruptions.py` — Lifecycle (warning sigmoid → active bell → recovery exp). Affected_nodes. Cascade injection.
- `financial.py` — Budget, costs, revenue loss, SLA penalties, backup premiums, health score. ISM/CSCMP/IATA-calibrated constants.
- `rewards.py` — 7-component dense reward (35% revenue, 25% stockout, 15% proactive, 10% cost, 5% unnecessary, 5% health, 5% SLA).
- `monte_carlo.py` — 10K scenarios/step. Beta-severity + Lognormal-duration. P50/P95.
- `__init__.py`.

**`tasks/` (5 files)**:
- `registry.py` — TaskRegistry, 3 built-ins.
- `task_easy.py` — 12 nodes, $5M, 30 steps, typhoon.
- `task_medium.py` — 25 nodes, $8M, 45 steps, multi-front.
- `task_hard.py` — 40 nodes, $10M, 60 steps, cascading.
- `__init__.py`.

**`graders/` (2 files)**:
- `grader.py` — EpisodeGrader. action_coverage, active_mitigation, cost_efficiency, health_score, SLA. Zero-variance across 5 runs.
- `__init__.py`.

**`data/` (8 files)**:
- `graphs/easy.json`, `medium.json`, `hard.json` — 12/25/40 nodes. Real names (TSMC, Samsung, Foxconn). Tier structure. Lead-times (SemiAnalysis/SEC). Single-source flags. Revenue contributions.
- `disruptions/easy.json`, `medium.json`, `hard.json` — Scripted timelines + jitter for seed variation.
- `commodities/` — FRED price cache (WTI, copper, FX pairs).

## E.13 ROOT (~30 files)

- `README.md` (40KB) — Master entry. Headline metrics. Architecture. Tasks. RL stack. Production. HF Space.
- `EXECUTIVE_SUMMARY.md` — v2 historical summary.
- `AUDIT_PLAN.md` (22KB) — 11 directives, 12 batches.
- `ALIENWARE_KICKOFF.md` (53KB) — 80-item RL training checklist.
- `BENCHMARKS_VS_PUBLIC.md` — Honest positioning vs M5/MTEB/MuJoCo/Kaggle/MT-Bench/ogbn-products.
- `comparison.md` — Hackathon vs other-env-types.
- `DATA_SOURCES.md` — 40+ real-world citations.
- `DEMO_SCRIPT.md`, `FINAL_DEMO.md` — Demo plans.
- `DEPLOY_HF_SPACE.md` — HF deploy guide.
- `EXTERNAL_CREDIBILITY.md` — Authority quotes.
- `JUDGES.md` — 4-min path, 15 receipts.
- `MODEL_CARD.md` (19KB) — Unified v3 model card.
- `PYTORCH_STORY.md` — 11 non-trivial PyTorch contributions.
- `RESULTS.md` — 1-page hero metrics.
- `SUPPLYMIND_BLUEPRINT.md` (81KB) — Master design doc.
- `LICENSE` — MIT.
- `baseline.py` — Scripted heuristic agent.
- `client.py` — Example HTTP client usage.
- `inference.py` — Competition entry. OpenAI-compatible. Stdout `[START]/[STEP]/[END]` format.
- `models.py` — Pydantic v2 contract.
- `scripted_agent.py` — Scripted logic.
- `openenv.yaml` — OpenEnv manifest.
- `docker-compose.yml` — 3 services (api, dashboard, damocles).
- `Dockerfile`, `Dockerfile.damocles`, `Dockerfile.dashboard` — Multi-stage builds.
- `pyproject.toml` — Project config.
- `requirements.txt`, `requirements-rl.txt`, `requirements-damocles.txt` — Deps.
- `uv.lock` (~15.7 MB) — Deterministic lock.
- `.env.example` — Template (OPENROUTER_API_KEY, HF_TOKEN, etc.).
- `.gitignore`, `.dockerignore` — Excludes.
- `wgidataset_with_sourcedata-2025.xlsx` (10.4MB) — World Bank governance indicators.
- `.claude/settings.local.json` — Claude Code permissions.

---

# END OF DOCUMENT

