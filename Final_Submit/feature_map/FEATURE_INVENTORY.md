# Feature Inventory — All 250 Features Mapped to Submission Use

This document maps every one of the 250+ unique features in the project to a specific use in the hackathon submission. **No feature wasted.**

Format: **Feature → File → Use in submission → Judging criterion served**

---

## A. Foundation Models & Fine-Tuning (24 features)

| # | Feature | File | Submission use | Criterion |
|---|---|---|---|---|
| 1 | DeepSeek-R1-Q4 (devil's advocate) | `rl/lora/Modelfile` | 3-judge panel for reward grading | Innovation 40% |
| 2 | Qwen-2.5-14B-local | `rl/lora/Modelfile` | Primary judge in Hormuz live demo | Innovation, Storytelling |
| 3 | Mistral-Nemo-local | (Ollama Modelfile) | 128K-ctx primary judge | Innovation |
| 4 | Qwen-2.5-Coder-14B | (Ollama) | Critic-pass JSON validator | Pipeline 10% |
| 5 | Chronos-Bolt | (HF) | Forecasting in env disruption modeling | Innovation |
| 6 | TimesFM-2 | (HF) | Forecasting with conformal residual quantile | Innovation |
| 7 | TabPFN-v2 (clf+reg) | `v3_arcadia/00_emergence/` | Tabular baseline for DataCo | Improvement 20% |
| 8 | BGE-M3 embeddings | `rl/rag/` | RAG corpus indexing | Innovation |
| 9 | mxbai-embed-large (winner) | `crisis_library.py` | Live Hormuz analog matching → 0.99 sim | Innovation, Storytelling |
| 10 | Snowflake-Arctic-Embed-L | `v3_arcadia/40_granite/` | OOD validation, nDCG@10=0.971 | Improvement |
| 11 | BGE-reranker-v2-m3 | `v3_arcadia/40_granite/` | Honest finding (reranker hurts on ceiling) | Innovation |
| 12 | Qwen-2.5-VL-7B | `ShAuRyA_Supplymind/features/qwen_vl_port_imagery.py` | Port imagery feature F1 | Innovation |
| 13 | `supplymind-analyst:v1` Modelfile | `rl/lora/Modelfile` | Phase-1 analyst with TSMC facts | Pipeline |
| 14 | `supplymind-analyst:v2` | `rl/lora/Modelfile.v2` | Improved domain knowledge | Pipeline |
| 15 | `supplymind-analyst:v3` | `rl/lora/Modelfile.v3` | Action costs added | Pipeline |
| 16 | `supplymind-analyst:v4` | `rl/lora/Modelfile.v4` | Phase 4 R3 LLM block | Improvement |
| 17 | **`supplymind-analyst:v5`** with 8 hard-negative few-shots | `ShAuRyA_Supplymind/features/Modelfile.analyst_v5` | **Wins 80% exact-risk vs base 0%** — receipt #14 | Improvement, Storytelling |
| 18 | LoRA fine-tune of Qwen-2.5-1.5B (PEFT) | `rl/lora/finetune.py` | Explanation generation | Pipeline |
| 19 | 225 instruction/output pairs | `rl/lora/lora_training_data.json` | LoRA training data | Improvement |
| 20 | DPO judge fine-tune Qwen-2.5-3B | `ShAuRyA_Phoenix/roll_integration/dpo_judge/` | Self-bootstrapping judge | Innovation |
| 21 | 21 DPO preference pairs | `ShAuRyA_Phoenix/roll_integration/dpo_judge/data/` | Receipt #18 | Improvement |
| 22 | Q4_K_M quantization | (config) | Fits 14B in 12GB GPU | Pipeline |
| 23 | Safetensors conversion (BGE-M3) | `v3_arcadia/00_emergence/convert_bge_to_safetensors.py` | CVE-2025-32434 workaround | Pipeline |
| 24 | TRL fallback for ROLL | `ShAuRyA_Phoenix/roll_integration/trl_fallback/` | Resilience under heavy deps | Pipeline |

---

## B. The Game Engine — Environment (28 features)

| # | Feature | File | Submission use | Criterion |
|---|---|---|---|---|
| 25 | 3 difficulty tasks (easy/med/hard) | `openenv.yaml` | Curriculum learning + theme #2 fit | Innovation 40% |
| 26 | 7-action discrete space | `models.py` | OpenEnv compliance + theme #3 | Innovation |
| 27 | MultiDiscrete[7,40] → Discrete(280) wrapper | `rl/gym_env.py` | MaskablePPO compatibility | Pipeline |
| 28 | 7-component dense reward | `server/engine/rewards.py` | Layered hacking-resistant reward | Pipeline 10% |
| 29 | Sigmoid disruption curve (warning) | `server/engine/disruptions.py:72-99` | Realistic ramp-up | Innovation |
| 30 | Bell disruption curve (active) | (same) | Realistic peak with dips | Innovation |
| 31 | Exponential decay (recovery) | (same) | Real long-tail recovery | Innovation |
| 32 | BFS propagation severity decay 0.20/hop | `server/engine/graph.py:29` | Multi-tier cascade modeling | Innovation |
| 33 | Time-discounted proactive bonus | `server/engine/rewards.py:130-132` | Encourages early action | Pipeline |
| 34 | One-per-episode bonus spam guard | `server/engine/rewards.py:123-127` | Anti-reward-hacking | Pipeline |
| 35 | Jitter within determinism | `server/supply_environment.py:72-77` | Reproducible + varied | Pipeline |
| 36 | 5 historical crisis scenarios | `benchmark/crisis_library/` | Real-world calibration | Innovation |
| 37 | Tohoku 2011 crisis | `tohoku_2011.json` | Real M9.0, $235B, 180-day | Innovation |
| 38 | Suez 2021 crisis | `suez_2021.json` | $9.6B/day, 6-day blockage | Innovation |
| 39 | Chip shortage 2020 | `chip_shortage_2020.json` | 2-year severity 0.85 | Innovation |
| 40 | Ukraine neon 2022 | `ukraine_neon_2022.json` | 70% global neon | Innovation |
| 41 | Red Sea 2023 | `red_sea_2023.json` | Houthi +10d/+25% fuel | Innovation |
| 42 | Real cost: $150K backup (ISM) | `server/engine/financial.py` | Calibration credibility | Innovation |
| 43 | 25% inventory carrying (CSCMP) | (same) | Calibration credibility | Innovation |
| 44 | 10× air vs sea (IATA) | (same) | Calibration credibility | Innovation |
| 45 | $9.6B/day Suez (Lloyd's) | (same) | Calibration credibility | Innovation |
| 46 | Anti-reward-hacking 6-attack suite | `tests/test_reward_hacking_adversarial.py` | All 6 rejected, receipt published | Pipeline 10%, Innovation |
| 47 | Session pool LRU eviction (max 20) | `server/app.py:188-198` | Concurrent judge isolation | Pipeline |
| 48 | Pre-warming on FastAPI startup | `server/app.py:49-68` | <100ms first reset | Pipeline |
| 49 | CORS allow_origins=["*"] | `server/app.py` | HF Space iframe support | Pipeline |
| 50 | OpenEnv MCP JSON-RPC | `server/openenv_adapter.py:195-212` | Standard compliance | Innovation, Pipeline |
| 51 | OpenEnv WebSocket route | (same) | Standard compliance | Innovation |
| 52 | Pydantic v2 typed contracts | `models.py` | OpenEnv compliance | Pipeline |

---

## C. RL Players (15 features)

| # | Feature | File | Use in submission | Criterion |
|---|---|---|---|---|
| 53 | **GRPO via TRL** (this submission's headline) | `Final_Submit/training/train_grpo_supplymind.py` | The required hackathon training stack | All 4 |
| 54 | **Unsloth 4-bit NF4** | (same) | Required hackathon stack | Pipeline |
| 55 | MaskablePPO | `rl/train_ppo.py` | Best RL baseline for ablation comparison | Improvement |
| 56 | Constrained PPO Lagrangian | `rl/constrained_ppo.py` | Math-guaranteed budget feature | Innovation |
| 57 | QR-DQN (51 quantiles, CVaR α=0.5) | `rl/distributional/qr_dqn.py` | Worst-10% optimization, avg=0.793 | Improvement |
| 58 | HER (Hindsight Experience Replay) | `rl/her_agent.py` | Sparse-reward fix on hard task | Pipeline |
| 59 | Decision Transformer | `rl/decision_transformer/model.py` | Risk-slider feature F4 | Innovation |
| 60 | BC (Behavior Cloning) | `rl/offline/baselines.py` | Easy task specialist | Improvement |
| 61 | CQL (Conservative Q-Learning) | (same) | Medium task specialist | Improvement |
| 62 | IQL (Implicit Q-Learning) | (same) | Hard task specialist | Improvement |
| 63 | TD3+BC | (same) | Offline RL baseline | Improvement |
| 64 | RecurrentPPO | (autoresearch s4) | Honest negative finding (collapsed) | Storytelling |
| 65 | A2C / SAC-Discrete / MBRL Dyna | `v3_arcadia/train_v3_block5_rl.py` | 8-algorithm comparison | Improvement |
| 66 | Specialist router | `rl/specialist_router.py` | Per-task best-checkpoint dispatch | Innovation |
| 67 | ONNX export <5e-5 roundtrip (4 models) | `rl/checkpoints/onnx/`, `onnx_roundtrip.json` | Production deployment | Pipeline |

---

## D. Forecasting & Statistics (16 features)

| # | Feature | File | Use | Criterion |
|---|---|---|---|---|
| 68 | TFT 90K params on real WTI | `rl/forecasting/tft.py` | MAE $7.83/bbl receipt | Improvement |
| 69 | TFT 513K params on 3-target | `rl/checkpoints/tft_v2_metrics.json` | Forecasting baseline | Improvement |
| 70 | Bates-Granger constrained stacking | `v3_arcadia/20_past_self/r3_constrained_stacking.py` | 9/21 wins (1969 method) | Innovation |
| 71 | Per-horizon split-conformal | `v3_arcadia/80_aqua_regia/` | Hero metric: WTI dev=0.024 | Innovation |
| 72 | TimesFM-CP residual quantile | `v3_arcadia/20_past_self/r3_timesfm_residual_quantile.py` | Receipt #9 | Innovation |
| 73 | 20-fold rolling-origin backtest | (same) | Statistical rigor | Pipeline |
| 74 | 8 FRED targets × 3 horizons | `v3_arcadia/20_past_self/train_past_self.py` | Forecasting breadth | Improvement |
| 75 | PICP@80/90/95% calibration | (same) | Coverage validation | Pipeline |
| 76 | Wilcoxon signed-rank | `benchmark/statistics.py` | p<1e-50 pairwise | Improvement |
| 77 | Friedman test | (same) | Multi-agent rigor | Improvement |
| 78 | Bootstrap CI95 (paired+unpaired) | (same) | Counterfactual twin's CI95 | Innovation |
| 79 | Krippendorff α (ordinal) | `scripts/compute_panel_agreement.py` | 0.21→0.75→0.567→0.358 disclosure | Storytelling |
| 80 | Cohen κ weighted | (same) | 0.747 receipt #5 | Improvement |
| 81 | 10,800-episode bootstrap | `v3_arcadia/60_euclidian/r6_massive_benchmark.py` | Non-overlapping CI95 | Improvement |
| 82 | ECE + Brier calibration | `v3_arcadia/10_caramel/` | Tabular reliability | Pipeline |
| 83 | Honest disclosure ladder | `MODEL_CARD.md` W1 | Storytelling integrity | Storytelling |

---

## E. Uncertainty & Interpretability (10 features)

| # | Feature | File | Use | Criterion |
|---|---|---|---|---|
| 84 | MC Dropout (50 forward passes) | `rl/uncertainty.py` | Q1=99.76% / Q4=55.92% calibration | Innovation |
| 85 | Conformal RL on Q-values | `ShAuRyA_Supplymind/features/conformal_rl.py` | Feature F6 abstention | Innovation |
| 86 | Confidence-damped projection | `crisis_library.py:187-189` | Live demo integrity guard | Innovation |
| 87 | Beta-severity + Lognormal-duration MC | `server/engine/monte_carlo.py` | Realistic uncertainty | Pipeline |
| 88 | Numba JIT MC (10-50× speedup) | `rl/fast_engine/fast_monte_carlo.py` | Fast inference | Pipeline |
| 89 | GPU MC 100K scenarios <80ms | `rl/surrogate/gpu_monte_carlo.py` | Real-time risk | Innovation |
| 90 | SHAP DeepExplainer on RL policy | `rl/interpretability/shap_real.py` | NOAA 60.1% importance | Innovation |
| 91 | LLM-RL hybrid explainer | `rl/explainer.py` | 4-section structured explanations | Innovation |
| 92 | 4-section output (Decision/Evidence/Counterfactual/Precedent) | (same) | 100% pass rate on 50 stress tests | Pipeline |
| 93 | Provenance 5-tier trust classifier | `ShAuRyA_Supplymind/features/rag_provenance.py` | Source lineage | Innovation |

---

## F. RAG & Knowledge (12 features)

| # | Feature | File | Use | Criterion |
|---|---|---|---|---|
| 94 | 8-pipeline RAG comparison | `v3_arcadia/40_granite/r5_rag_beast.py` | Honest finding (mxbai wins) | Storytelling |
| 95 | mxbai bi-encoder (winner) | (same) | P@1=0.962 receipt #1 | Improvement |
| 96 | RRF ensemble pipeline | (same) | Multi-encoder fusion | Innovation |
| 97 | HyDE via Qwen-14B | (same) | Honest null finding | Storytelling |
| 98 | 6,483-chunk corpus | (RAG corpus) | Real knowledge base | Innovation |
| 99 | ChromaDB persistent storage | `rl/rag/indexer.py` | Production RAG | Pipeline |
| 100 | Corpus SHA-256 caching | `crisis_library.py:100-120` | Smart re-embed | Pipeline |
| 101 | TF-IDF cosine fallback | `crisis_library.py:54-76` | Demo never fails | Pipeline |
| 102 | min_score=0.60 threshold | `rl/rag/indexer.py` | Quality gate | Pipeline |
| 103 | 53 precise queries + 20 hard | `R5_GRANITE.json` + `R5_GRANITE_HARD.json` | Eval rigor | Improvement |
| 104 | BEIR external validation | `v3_arcadia/40_granite/r5_manual_beir.py` | Snowflake nDCG=0.971 | Improvement |
| 105 | "Reranker hurts" honest publish | `R5_GRANITE_REPORT.md` | Storytelling integrity | Storytelling |

---

## G. GNN & Graph Modeling (8 features)

| # | Feature | File | Use | Criterion |
|---|---|---|---|---|
| 106 | Custom 3-layer GCN (50 lines) | `rl/gnn/tgn.py` | F1=0.964 receipt #7 | Innovation |
| 107 | `index_add_` message passing | (same) | No torch_geometric | Pipeline |
| 108 | TGN per-node memory + GRU | (same) | Temporal modeling | Innovation |
| 109 | TransformerConv (PyG ≥2.3) | (same) | Modern attention | Innovation |
| 110 | 5-day risk trajectory prediction | (same) | Multi-step forecasting | Innovation |
| 111 | 3 graphs (12/25/40 nodes) | `server/data/graphs/` | Difficulty progression | Innovation |
| 112 | GNN attention edge weights | `ShAuRyA_Supplymind/features/gcn_attention_viz.py` | Feature F7 viz | Innovation |
| 113 | gnn_arrival.onnx (10KB) | `v3_arcadia/checkpoints/onnx_bundle/` | Edge deployment | Pipeline |

---

## H. Federated, Multi-Agent, Pareto (10 features)

| # | Feature | File | Use | Criterion |
|---|---|---|---|---|
| 114 | FedAvg 3-client (Apple/Samsung/Toyota) | `rl/federated/fedavg.py` | +263% accuracy lift | Innovation |
| 115 | Optional differential privacy | (same) | DP noise std=0.1 | Innovation |
| 116 | 20 rounds × 5 local epochs | (same) | Federated rigor | Pipeline |
| 117 | Multi-agent Apple/Samsung/Toyota auction | `rl/multi_agent/competitive.py` | Feature F2 demo | Innovation |
| 118 | NSGA2 Pareto frontier (3 obj) | `rl/pareto/frontier.py` | Feature F9 | Innovation |
| 119 | Carbon factors (EPA/IMO/ICAO) | (same) | Calibration credibility | Innovation |
| 120 | 11 Pareto-frontier plans | (same) | Multi-objective tradeoff | Innovation |
| 121 | 3D Plotly dashboard | (same) | Visualization | Storytelling |
| 122 | Bates-Granger 9/21 wins | (forecasting) | Honest stacking | Improvement |
| 123 | Stacking v2 honest null (+0.0045 vs WV) | `R15_STACKING_V2.json` | Receipt #11 | Storytelling |

---

## I. Live Data & Hormuz Demo (15 features)

| # | Feature | File | Use | Criterion |
|---|---|---|---|---|
| 124 | NewsAPI ingestion (5 keyword groups) | `realtime/sources/newsapi.py` | Hormuz live demo | Innovation, Storytelling |
| 125 | GDELT 2.0 Doc API | `realtime/sources/gdelt.py` | Government reports | Innovation |
| 126 | USGS earthquake feed (M4.5+) | `realtime/sources/usgs.py` | 19 real events | Innovation |
| 127 | FRED Brent (DCOILBRENTEU) | `realtime/sources/fred_brent.py` | $123.28 receipt #12 | Innovation |
| 128 | MarineTraffic AIS (optional) | `realtime/sources/marinetraffic.py` | 5th data source | Innovation |
| 129 | SQLite event store | `realtime/store.py` | Append-only audit trail | Pipeline |
| 130 | SHA-256 dedup hash | (same) | Tamper-evident | Pipeline |
| 131 | 24h dedup window | (same) | Anti-spam | Pipeline |
| 132 | KNOWN_ENTITIES extraction | `realtime/sources/newsapi.py:140-154` | Cheap NER | Pipeline |
| 133 | Severity from keyword weights | `realtime/sources/newsapi.py:48-60` | NewsAPI severity | Pipeline |
| 134 | Severity from price-spike formula | `realtime/sources/fred_brent.py:28-30` | FRED severity | Pipeline |
| 135 | 8-event crisis library | `scenarios/iran_israel_hormuz_2024_2026.json` | Memory book for matching | Innovation |
| 136 | ≥3 citations per event | (same, curation policy line 183) | Audit trail | Storytelling |
| 137 | mxbai analog matching → 0.99 | `crisis_library.py:140-184` | Live demo showstopper | Storytelling |
| 138 | Counterfactual: $324M → $65M | `hormuz_endpoint.py:262-282` | Showstopper math | Storytelling |

---

## J. Phoenix v5 Layer (15 features)

| # | Feature | File | Use | Criterion |
|---|---|---|---|---|
| 139 | Counterfactual digital twin | `ShAuRyA_Phoenix/counterfactual_twin/twin.py` | Demo #2 (CI95 [177.74,179.52]) | Innovation, Storytelling |
| 140 | 100-rollout MC | (same) | Statistical confidence | Improvement |
| 141 | Paired bootstrap CI95 | (same) | Saving range certainty | Improvement |
| 142 | OpenEnv Arena leaderboard | `ShAuRyA_Phoenix/arena/` | Drop-in policy harness | Innovation |
| 143 | 6 baselines pre-seeded | `arena/leaderboard.py` | MaskablePPO #1 mean=2.209 | Improvement |
| 144 | 3 callable Claude Code skills | `supplymind_skills/` | benchmark-runner, autoresearch, demo-orch | Innovation |
| 145 | plugin.json v1.0.0 manifest | (same) | Skill marketplace | Innovation |
| 146 | Replay cache (8 events frozen) | `realtime_v5/replay_cache_latest.json` | Demo never fails | Pipeline |
| 147 | freeze_cache.py | `realtime_v5/freeze_cache.py` | Generate offline replay | Pipeline |
| 148 | replay_adapter.py | `realtime_v5/replay_adapter.py` | Status + load | Pipeline |
| 149 | ROLL integration | `roll_integration/` | Alibaba framework | Innovation |
| 150 | DPO judge worker | `roll_integration/reward_bridge/` | Custom ROLL extension | Innovation |
| 151 | Two upstream PRs ready | `upstream_prs/{meta_openenv,alibaba_roll}/` | Ecosystem contribution | Innovation, Storytelling |
| 152 | Phoenix isolation guarantee | (architecture) | v3+v4 untouched safety net | Pipeline |
| 153 | Copy-before-edit discipline | (architecture) | Multi-layer safety | Pipeline |

---

## K. Autoresearch System (10 features)

| # | Feature | File | Use | Criterion |
|---|---|---|---|---|
| 154 | Karpathy autoresearch loop | `rl/autoresearch.py` | Demo #4 (theme #4 fit) | Innovation |
| 155 | LLM hypothesis generation | (same) | Autonomous research | Innovation |
| 156 | Mutable candidate_train.py | `autoresearch_fixed/candidate_train.py` | Safe-to-modify markers | Pipeline |
| 157 | 10-min wall-clock kill | `runner.py` | Safety guard | Pipeline |
| 158 | OOM/NaN/test gate | (same) | Robustness | Pipeline |
| 159 | ≤150 LOC diff limit | (same) | Bounded changes | Pipeline |
| 160 | Bootstrap CI95 lower threshold | `evaluator.py` | Δ>0.005 acceptance | Improvement |
| 161 | 9 random seeds per experiment | (same) | Statistical rigor | Improvement |
| 162 | Lab notebook auto-Markdown | `lab_notebook.py` | Research transparency | Storytelling |
| 163 | s3 curriculum +0.0967 lift | `state.json` | Receipt #15 | Improvement |

---

## L. Test Suite & Receipts (12 features)

| # | Feature | File | Use | Criterion |
|---|---|---|---|---|
| 164 | 19 OpenEnv compliance tests | `tests/test_openenv_compliance.py` | Receipt — formal compliance | Pipeline |
| 165 | 6 adversarial reward attacks | `tests/test_reward_hacking_adversarial.py` | All rejected | Pipeline 10% |
| 166 | 173 v3 core tests | (test suite) | Foundation reliability | Pipeline |
| 167 | 76 v4 new feature tests | `ShAuRyA_Supplymind/tests/` | Feature coverage | Pipeline |
| 168 | 7+ Phoenix smoke tests | `ShAuRyA_Phoenix/tests/test_smoke.py` | v5 sanity | Pipeline |
| 169 | 250 total tests in 2m38s | (full suite) | Receipt #13 | Pipeline |
| 170 | 35 reproducibility receipts | `receipts/INDEX.md` + Phoenix v5 | One-bash-command audit trail | Storytelling, Pipeline |
| 171 | SHA-256 stdout tracking | Receipt framework | Tamper-evident | Pipeline |
| 172 | Hardware capture (CUDA) | (same) | Reproducibility | Pipeline |
| 173 | 5 comparator types | (same) | ==, >=, <=, in_range, regex | Pipeline |
| 174 | INDEX.json + INDEX.md auto | `register.py` | Self-updating ledger | Pipeline |
| 175 | check_benchmarks.py CI guard | `scripts/check_benchmarks.py` | Regression protection | Pipeline |

---

## M. Tabular ML & Trained Analysis Models (15 features)

| # | Feature | File | Use | Criterion |
|---|---|---|---|---|
| 176 | XGBoost (GPU hist) | `v3_arcadia/10_caramel/train_caramel.py` | Tabular baseline | Improvement |
| 177 | LightGBM (winner) | (same) | AUC=0.9818, F1=0.9724 | Improvement |
| 178 | CatBoost | (same) | 3rd ensemble member | Improvement |
| 179 | TabPFN-v2 zero-shot | (same) | 4th ensemble member | Improvement |
| 180 | TabPFN bagging (random subsample) | `r2_tabpfn_bagging.py` | Confidence intervals | Improvement |
| 181 | Stacking with Ridge meta | `stacking_v2.py` | Honest null finding | Storytelling |
| 182 | 5-fold OOF CV | (same) | Proper stacking | Pipeline |
| 183 | 4 leak-free DataCo tasks | `train_v3_block1_real_labels.py` | Anti-leakage discipline | Pipeline |
| 184 | Political risk GBR (R²=0.994) | `rl/analysis/trained_models.py` | 214 countries | Innovation |
| 185 | Dependency MLP (97.45% acc) | (same) | 144K samples | Innovation |
| 186 | Financial impact Ridge (R²=0.736) | (same) | $25.66 MAE | Innovation |
| 187 | Confidence isotonic (ECE=0.0017) | (same) | Calibration | Pipeline |
| 188 | Safety stock empirical | (same) | Lead-time data-driven | Innovation |
| 189 | Articulation-point SPOF v2 (F1=1.0) | `spof_v2.py` | Receipt #10 | Improvement |
| 190 | 4-component dependency score | `rl/analysis/dependency_scoring.py` | Real index | Innovation |

---

## N. LLM Judging Infrastructure (15 features)

| # | Feature | File | Use | Criterion |
|---|---|---|---|---|
| 191 | 3-judge local panel | `v3_arcadia/30_dangerous/r4_judge_layer.py` | Demo #3 component | Innovation |
| 192 | 12-frontier OpenRouter panel | `scripts/run_frontier_judge_panel.py` | Demo #3 (15 judges total) | Innovation, Storytelling |
| 193 | NVIDIA Nemotron-3 120B (judge) | `openrouter_client.py:51-106` | Free tier | Innovation |
| 194 | Hermes-3 Llama 405B (judge) | (same) | Paid ~₹3 total | Innovation |
| 195 | Llama-3.3 70B (judge) | (same) | Meta SOTA | Innovation |
| 196 | GPT-OSS 120B (judge) | (same) | OpenAI open | Innovation |
| 197 | 4 Gemma + Qwen variants (judges) | (same) | Multi-lab diversity | Innovation |
| 198 | 26 real Wikipedia crisis scenarios | `R4_DANGEROUS_V2.json` | Judging evaluation set | Improvement |
| 199 | DeepSeek devil's-advocate role | `r4_v2_beast.py` | Honest reframing | Storytelling |
| 200 | Two-pass DeepSeek extraction | (same) | 100% parse rate | Pipeline |
| 201 | Token-bucket rate limiter (18/min) | `openrouter_client.py` | Free tier discipline | Pipeline |
| 202 | API call caching `.openrouter_cache/` | (same) | Cost control | Pipeline |
| 203 | `.openrouter_usage.jsonl` | (same) | Spend tracking | Pipeline |
| 204 | Total OpenRouter spend ₹3 | (verifiable) | Cost-effectiveness story | Storytelling |
| 205 | Phase A/B caching (resume-safe) | `R4_DANGEROUS_V2_phaseA_cache.json` | 26-scenario reliability | Pipeline |

---

## O. Documentation & Narrative (20 features)

| # | Feature | File | Use | Criterion |
|---|---|---|---|---|
| 206 | README.md (40KB) | `../README.md` | Master doc | Storytelling 30% |
| 207 | SUPPLYMIND_BLUEPRINT.md (81KB) | `../SUPPLYMIND_BLUEPRINT.md` | Design depth | Storytelling |
| 208 | ALIENWARE_KICKOFF.md (53KB) | `../ALIENWARE_KICKOFF.md` | 80-item checklist | Storytelling |
| 209 | AUDIT_PLAN.md (22KB) | `../AUDIT_PLAN.md` | 11 directives | Storytelling |
| 210 | MODEL_CARD.md (W1-W10) | `../MODEL_CARD.md` | 10 design wins | Storytelling |
| 211 | PYTORCH_STORY.md | `../PYTORCH_STORY.md` | 11 PyTorch contributions | Storytelling |
| 212 | BENCHMARKS_VS_PUBLIC.md | `../BENCHMARKS_VS_PUBLIC.md` | Honest positioning | Storytelling |
| 213 | DATA_SOURCES.md (40+ citations) | `../DATA_SOURCES.md` | Calibration audit | Storytelling |
| 214 | EXTERNAL_CREDIBILITY.md | `../EXTERNAL_CREDIBILITY.md` | Authority quotes | Storytelling |
| 215 | JUDGES.md (4-min path) | `../JUDGES.md` | Judge journey | Storytelling |
| 216 | FINAL_DEMO.md | `../FINAL_DEMO.md` | Top-3 plan | Storytelling |
| 217 | PREPRINT.md (Arxiv-style) | `ShAuRyA_Supplymind/docs/PREPRINT.md` | Research paper | Storytelling |
| 218 | LIVE_DEMO_HORMUZ.md (90-sec) | `ShAuRyA_Supplymind/docs/LIVE_DEMO_HORMUZ.md` | Demo script | Storytelling |
| 219 | 6 Colab notebooks | `notebooks/` | Re-runnable tutorials | Improvement, Storytelling |
| 220 | 12 Sleep Token-named research stages | `v3_arcadia/00_emergence/` → `95_arcadia/` | Organizing narrative | Storytelling |
| 221 | _dump/FAILURE_TABLE.md | `_dump/FAILURE_TABLE.md` | 8 honest negatives | Storytelling |
| 222 | R4_RUBRIC_CHALLENGE.md | `challenges/R4_RUBRIC_CHALLENGE.md` | Open challenge | Storytelling |
| 223 | EXECUTIVE_SUMMARY.md | `../EXECUTIVE_SUMMARY.md` | 1-page summary | Storytelling |
| 224 | DEMO_VIDEO_SCRIPT.md (8-scene 3-min) | `demo/DEMO_VIDEO_SCRIPT.md` | Video plan | Storytelling |
| 225 | RESULTS.md (10-number hero) | `../RESULTS.md` | Quick reference | Storytelling |

---

## P. Visualization (15 features)

| # | Feature | File | Use | Criterion |
|---|---|---|---|---|
| 226 | Hero result card (10-number 2×5) | `v3_arcadia/plots/hero_result_card.png` | README hero | Storytelling |
| 227 | make_hero_card.py | `v3_arcadia/plots/make_hero_card.py` | Generator script | Pipeline |
| 228 | R4 ablation heatmap | `r4v2_heatmap.png` | Storytelling visual | Storytelling |
| 229 | R5 latency vs MRR scatter | `r5_latency_vs_mrr.png` | Tradeoff plot | Storytelling |
| 230 | R5 per-query heatmap | `r5_per_query_heatmap.png` | Granular results | Improvement |
| 231 | R6 learning curves | `learning_curves.png` | Training evidence | Improvement 20% |
| 232 | R6 masking ablation | `r6_masking_ablation.png` | +26.8% lift visualization | Improvement |
| 233 | R6 provider network graph | `r6_provider.png` | GNN visualization | Storytelling |
| 234 | R6 euclidian bootstrap CI | `r6_euclidian.png` | Non-overlapping CI95 | Improvement |
| 235 | R3 summary heatmap | `r3_summary.png` | 8 targets × 3 horizons | Improvement |
| 236 | TimesFM quantile lines | `r3_timesfm_quantile.png` | Forecast visualization | Improvement |
| 237 | Aqua regia coverage plot | `r6_aqua_regia.png` | Conformal calibration | Innovation |
| 238 | GCN attention heatmaps (3 graphs) | `gcn_attn_easy/medium/hard_graph.png` | Edge importance | Innovation |
| 239 | Streamlit dashboard 12 panels | `dashboard/app.py` | Interactive demo | Storytelling |
| 240 | Pareto 3D scatter | `pareto/frontier.py` Plotly | Multi-objective viz | Innovation |

---

## Q. Production Infrastructure (10 features)

| # | Feature | File | Use | Criterion |
|---|---|---|---|---|
| 241 | 3 Dockerfiles | `Dockerfile`, `Dockerfile.dashboard`, `Dockerfile.damocles` | Modular containers | Pipeline |
| 242 | docker-compose with healthcheck | `docker-compose.yml` | Service orchestration | Pipeline |
| 243 | HF Space deployed | `huggingface.co/spaces/Shaurya-Noodle/Supplymind` | Live submission requirement | Storytelling |
| 244 | <2GB container, 6-8min build | (deploy spec) | Cold start <25s | Pipeline |
| 245 | Multi-stage Python 3.11-slim | `Dockerfile` | Slim production | Pipeline |
| 246 | Non-root appuser (UID 1000) | (same) | Security | Pipeline |
| 247 | HEALTHCHECK curl /health 30s | (same) | Liveness probe | Pipeline |
| 248 | /docs Swagger UI live | `server/app.py` (FastAPI auto) | Judge interactive doc | Storytelling |
| 249 | CUDA action mask kernel attempt | `rl/cuda/action_mask_kernel.py` + `.dll` | 1833× PyTorch fallback | Pipeline |
| 250 | MSVC blocker honest disclosure | `_dump/FAILURE_TABLE.md` | Engineering integrity | Storytelling |

---

## Final tally

**250 features mapped, 0 wasted.**

Each feature is used in:
- ✅ The OpenEnv environment itself (28)
- ✅ The training pipeline / model layer (15 RL + 24 fine-tuning + 16 forecasting = 55)
- ✅ The 4 demos (15 live + 15 Phoenix + 10 autoresearch = 40)
- ✅ The presentation layer (20 docs + 15 plots = 35)
- ✅ The infrastructure (12 tests/receipts + 10 production = 22)
- ✅ The credibility layer (15 LLM judging + 15 stats + 10 uncertainty + 12 RAG + 8 GNN + 10 multi-agent + 15 trained models = 85)

**Coverage by judging criterion**:
- Innovation 40%: ~120 features serving this
- Storytelling 30%: ~75 features serving this
- Improvement 20%: ~40 features serving this
- Pipeline 10%: ~80 features serving this (overlap with above)

**No feature is sitting unused. Every line of work goes into the submission.**
