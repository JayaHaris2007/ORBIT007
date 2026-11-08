# ORBIT007 — Ollama Backend Verification Report

**Generated:** 2026-11-02 02:45:40 UTC  
**Backend URL:** `http://127.0.0.1:8000`  
**Ollama URL:** `http://127.0.0.1:11434`  
**Model:** `qwen3.5:4b`  

## Summary Table

| # | Feature / Check | Status | Duration | Notes |
|---|---|---|---|---|
| 01 | Ollama Direct | **PASS** | 10.63s | VRAM: 2995.0 MB (100% on GPU) |
| 02 | Speed and Thinking | **PASS** | 2.59s | Warm app latency: 0.02s, direct Ollama speed: 56.1 tokens/s |
| 03 | Health Check | **PASS** | 0.00s | backend=ollama, reachable=True, scheduler=True |
| 04 | Plain Chat | **PASS** | 1.36s | Provider: ollama, Reply: The capital of France is Paris.... |
| 05 | Multi-turn Context | **PASS** | 1.74s | Retained 'teal' across turns with None tokenizer |
| 06 | Long-context Safety | **PASS** | 0.92s | Processed 1856 chars safely, reply: 25 |
| 07 | Rules-first Commands | **PASS** | 0.39s | All executed via local rules, max latency: 0.10s (<1.0s) |
| 08 | Reminders via Chat | **PASS** | 0.07s | Created reminder with ISO timestamp verified in API; alert wait skipped (pass --slow to wait) |
| 09 | Task Recommendation | **PASS** | 4.21s | LLM recommended highest priority task ('Fix critical server crash') |
| 10 | Learning Fallback | **SKIP** | 0.00s | Covered by unit test `test_fallback_llm_for_learning_regression` in tests/test_ollama_only.py (PASS) |
| 11 | Memory Search | **WARN** | 0.04s | Memory saved, but code name not named in reply: you're jaya haris. |
| 12 | Web Research | **SKIP** | 0.00s | TAVILY_API_KEY not configured in environment |
| 13 | Agent Mode WebSocket (/ws) | **PASS** | 2.95s | Agent turn completed, text length: 107, WAV fetched: 311084 bytes (audio/wav) |
| 14 | STT Round-trip | **SKIP** | 0.00s | Pass --slow to run STT round-trip audio transcription test |
| 15 | Concurrency (3 parallel) | **PASS** | 1.21s | 3 concurrent requests completed without deadlock |
| 16 | Failure and Recovery | **PASS** | 0.00s | Automated unit tests in test_ollama_only.py pass (offline + recovery). Guided manual steps printed for owner. |
| 17 | Resource Check (Manual) | **MANUAL** | 0.00s | Run `nvidia-smi -l 1` to confirm VRAM stays under 6 GB during concurrent voice + LLM operations. |
| 18 | Discord Checklist | **MANUAL** | 0.00s | Manual verification checklist printed: bot online, chat reply, reminders, task command, voice call. |

## Measured Performance & Resource Metrics

- **Direct Ollama Generation Speed:** 56.1 tokens/s
- **Ollama VRAM Allocation (`size_vram`):** 2995.0 MB
- **GPU Offload:** 100% layers offloaded to GPU

## Manual Owner Steps Still Required

### 1. Guided Failure & Recovery (Check 16)
To manually observe recovery without server restart:
1. In PowerShell, stop Ollama: `taskkill /IM ollama.exe /F`
2. Send a chat request in the frontend or curl `POST http://127.0.0.1:8000/api/chat` with `{"message": "hi"}`. Confirm the friendly fallback message is returned and rules-first commands like `add task buy milk` still work.
3. Restart Ollama: `ollama serve`
4. Send a chat request again: confirm normal assistant response without restarting `app.py`.

### 2. Peak VRAM Under Load (Check 17)
1. In a separate shell, run: `nvidia-smi -l 1`
2. Send a multi-sentence prompt in the chat with Agent Mode / TTS enabled.
3. Confirm peak GPU memory usage stays strictly under 6.0 GB and nothing falls back to CPU.

### 3. Discord Bot Checklist (Check 18)
- [ ] Bot connects and appears online in your Discord server.
- [ ] Bot answers regular chat mentions.
- [ ] Bot parses task and reminder commands from Discord messages.
- [ ] Voice call receives a spoken audio reply via Phonon or Windows SAPI.

## Untracked Local Weights To Delete

The following untracked PyTorch weights from the legacy Transformers Qwen path can now be safely removed by the owner to reclaim disk space (~8.04 GB total):
- `Models/LLM/model-00001-of-00003.safetensors` (~3.96 GB)
- `Models/LLM/model-00002-of-00003.safetensors` (~3.99 GB)
- `Models/LLM/model-00003-of-00003.safetensors` (~99.6 MB)

```powershell
# Command to reclaim ~8.04 GB:
Remove-Item -Recurse -Force Models/LLM
```
