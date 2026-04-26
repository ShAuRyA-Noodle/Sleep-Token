# API KEYS TO GET — direct signup links

Each link goes to the official signup page. All FREE tier. Most are instant. Once you have a key, append to `.env` in repo root (never commit).

## Tier 1 — HIGH IMPACT (closes documented gaps)

### 1. FRED (Federal Reserve Economic Data) — closes L9
- **Link**: https://fred.stlouisfed.org/docs/api/api_key.html
- **Cost**: FREE, instant, 120,000 req/day
- **Closes**: HONEST_LIMITATIONS L9 (synthetic Brent pre-history). Real DCOILBRENTEU 200-day pre-event slices for 8 documented events.
- **.env line**: `FRED_API_KEY=<paste_key>`
- **What this unlocks**: ensemble Brent backtest using REAL FRED Brent history instead of AR(1)+sinusoid synthetic. Drops `prehistory_source` from `synthetic` to `fred_dcoilbrenteu_real`. Tightens median rel error 3.32% → expected <2.5%.

### 2. NewsAPI — closes G4
- **Link**: https://newsapi.org/register
- **Cost**: FREE, instant, 100 req/day developer tier (100 articles/req)
- **Closes**: HONEST_LIMITATIONS G4 (NewsAPI key missing → OpenRouter substitute path). Real news ingest for chained_live_demo Stage C.
- **.env line**: `NEWS_API_KEY=<paste_key>`
- **What this unlocks**: live geopolitical event polling (Hormuz / Iran / Israel / Red Sea / Suez) into `realtime/news_ingest.py`. Feeds RAG corpus with real headlines instead of GDELT-only.

### 3. NOAA Climate Data Online — adds tropical cyclone real fetch
- **Link**: https://www.ncdc.noaa.gov/cdo-web/token
- **Cost**: FREE, email arrives in <60s
- **Closes**: G4 NewsAPI substitute path + adds tropical cyclone tracking real fetch (typhoon scenarios in `easy_typhoon_response`)
- **.env line**: `NOAA_TOKEN=<paste_token>`
- **What this unlocks**: `realtime/noaa.py` ingests real tropical cyclone IBTrACS for typhoon-response task. Closes M-section data gap.

### 4. Weights & Biases (W&B) — experiment tracking dashboard
- **Link**: https://wandb.ai/authorize
- **Cost**: FREE academic / personal, instant
- **Closes**: V8 W&B-style logs gap (currently locally-styled, not actually wandb)
- **.env line**: `WANDB_API_KEY=<paste_key>`
- **What this unlocks**: link from README to live W&B dashboard showing reward curve / loss / wallclock for every training run. Judges can check live, not just rely on PNG.

## Tier 2 — INNOVATION SURFACE (adds new data sources)

### 5. ACLED (Armed Conflict Location & Event Data) — political conflict
- **Link**: https://acleddata.com/access-data/data-and-resources/
- **Cost**: FREE for academic / non-commercial. Application needs short purpose statement (~1 paragraph).
- **Closes**: M-section gap. Replaces `ACLED unavailable` honest disclosure in `HONEST_LIMITATIONS.md` L10.
- **.env line**: `ACLED_API_KEY=<paste>` + `ACLED_EMAIL=<your_email>`
- **What this unlocks**: live political conflict events for Hormuz / Israel / Sudan / Houthi corridors. Adds 12K+ events to crisis library.

### 6. Alpha Vantage — supplementary commodity / FX
- **Link**: https://www.alphavantage.co/support/#api-key
- **Cost**: FREE, instant, 25 req/day
- **Closes**: Backup to FRED for commodity price (Brent, WTI, Copper, Gold, Soybeans).
- **.env line**: `ALPHA_VANTAGE_KEY=<paste>`
- **What this unlocks**: redundant commodity feed if FRED rate-limits.

### 7. Polygon.io — equity + commodity backup
- **Link**: https://polygon.io/dashboard/signup
- **Cost**: FREE tier 5 req/min
- **Closes**: Backup commodity feed.
- **.env line**: `POLYGON_API_KEY=<paste>`
- **What this unlocks**: equity tickers (Apple/Samsung/Toyota stock-price reaction to disruptions for war-room).

### 8. Exa.ai (semantic search) — real-time RAG
- **Link**: https://exa.ai/
- **Cost**: FREE tier, instant
- **Closes**: G4 NewsAPI alt + RAG live retrieval upgrade.
- **.env line**: `EXA_API_KEY=<paste>`
- **What this unlocks**: semantic web search (not keyword) for crisis events. Lifts RAG quality vs GDELT keyword search.

### 9. Brave Search API — search backup
- **Link**: https://brave.com/search/api/
- **Cost**: FREE 2,000 req/month
- **.env line**: `BRAVE_API_KEY=<paste>`
- **What this unlocks**: alternative search source for chained demo if Exa rate-limits.

### 10. Sentinel Hub (Copernicus EU satellite) — port imagery
- **Link**: https://www.sentinel-hub.com/
- **Cost**: FREE tier 30,000 processing units/month
- **.env line**: `SENTINEL_HUB_INSTANCE_ID=<paste>` + `SENTINEL_HUB_CLIENT_ID=<paste>` + `SENTINEL_HUB_CLIENT_SECRET=<paste>`
- **What this unlocks**: real satellite imagery overlay for port congestion (Suez / Hormuz / Long Beach) in war-room visual.

## Tier 3 — OPTIONAL (innovation upside)

### 11. HuggingFace Hub Token (write scope) — model checkpoint upload
- **Link**: https://huggingface.co/settings/tokens — create with **write** scope
- **Cost**: FREE
- **Closes**: U32 (HF Hub model upload). Currently HF_TOKEN may be read-only.
- **.env line**: replace existing `HF_TOKEN=<new_write_scope_token>`
- **What this unlocks**: push REINFORCE-trained Wordle policy + GRPO-trained LLaMA-1B to `Shaurya-Noodle/supplymind-reinforce-v2` and `Shaurya-Noodle/supplymind-llama-grpo`.

### 12. OpenAI / Anthropic / Google API (paid, optional)
- Currently substituted by local Ollama (qwen2.5:14b, deepseek-r1, supplymind-analyst:v5 — 20 models).
- Skip unless you want to add a frontier-model judge to the panel.

---

## After you add any key

```bash
# verify .env loaded
python -c "from dotenv import load_dotenv; load_dotenv(); import os; print({k:('set' if os.environ.get(k) else 'missing') for k in ['FRED_API_KEY','NEWS_API_KEY','NOAA_TOKEN','WANDB_API_KEY','ACLED_API_KEY','EXA_API_KEY']})"
```

Then I will run `pass28_real_data_ingest.py` to verify each key is live and emit a sha256-stamped receipt.

---

## What you ALREADY have (confirmed in .env)

- ✅ OPENROUTER_API_KEY (we will not use to save credit)
- ✅ EIA_API_KEY (live, $91.06/bbl WTI verified)
- ✅ NASA_FIRMS_MAP_KEY (live, 3986 csv lines verified)
- ✅ GFW_API_TOKEN (key authenticated, 503 transient honestly disclosed)
- ✅ HF_TOKEN (likely read scope; check write scope)
- ✅ Ollama LOCAL (20 models including qwen2.5:14b + supplymind-analyst:v5 14B + deepseek-r1)

---

## Priority order for MAXIMUM VICTORY LIFT

1. **FRED** (15 min compute lift, closes biggest data-source honest limitation L9)
2. **NewsAPI** (5 min code lift, closes G4)
3. **NOAA** (5 min code lift, closes M typhoon)
4. **WANDB** (10 min lift, makes training judge-verifiable live)
5. **ACLED** (90 min lift, adds 12K+ conflict events)
6. **Exa.ai** (15 min lift, semantic RAG)
7. Rest are upside, not critical

Total: 7 keys × ~2 min signup each = ~15 min of your time. Each lifts the submission probability by ~0.5-2pp.

End API keys checklist.
