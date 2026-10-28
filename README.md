# Local AI Voice Interface

This project combines the local models and modern voice pipeline:

- `Models/LLM/` contains a local Qwen3-1.7B instruction model for generating text.
- `voice/` contains the complete voice engine powered by **OmniVoice** (`k2-fsa/OmniVoice`, 0.6B params) for text-to-speech, and `SpeechRecognition` for speech-to-text.
- `Models/voice/` contains legacy Kokoro v1.0 files (superseded by OmniVoice).

`app.py` is the FastAPI backend and terminal interface.

## OmniVoice System (TTS & STT)

The S.P.I.D.E.Y voice system provides real-time speech synthesis, speech recognition, live hands-free conversation, and Discord voice bridge.

> [!IMPORTANT]
> **OmniVoice Weights License Notice**: The OmniVoice model weights (`k2-fsa/OmniVoice`) are licensed under **CC-BY-NC (Creative Commons Attribution-NonCommercial 4.0 International)** for personal and non-commercial use only. Do not clone anyone's voice without authorization; use voice design or your own reference audio recording.

### Architecture & Safeguards
- **In-Memory Audio Streaming**: Audio is synthesized and streamed in memory as 24 kHz mono PCM / WAV. No audio is persisted to disk by default. Any transient files use `voice/tmp/` (git-ignored), which is completely purged on application startup and voice session reset.
- **Fail-Safe Chat Operation**: Text chat and voice are strictly decoupled. If OmniVoice fails to load, runs out of VRAM, or encounters an error, text chat continues working uninterrupted and voice features degrade gracefully with a user notification.
- **Automatic Device Fallback**: The TTS engine attempts to load on `TTS_DEVICE` (`cuda:0` in `float16`), fitting within the ~3.4 GB VRAM headroom remaining beside Qwen3 4-bit NF4. If CUDA OOM occurs, it automatically falls back to CPU `float32`.
- **Thread-Safe Cancellation**: When the user interrupts or clicks Stop, synthesis cancellation is checked between sentence chunks and during generation, instantly silencing output and freeing buffers.

### Voice Configuration Options
Environment variables (optional, in `.env`):
- `TTS_DEVICE`: `cuda:0` (default) or `cpu`.
- `TTS_DTYPE`: `float16` (default for CUDA) or `float32` (default for CPU).
- `TTS_MODEL_ID`: Hugging Face repo ID or local path (default: `k2-fsa/OmniVoice`).
- `TTS_REFERENCE_AUDIO`: Optional path to custom reference audio clip for voice cloning (e.g. `TTS/audio.wav`).
- `TTS_INSTRUCT`: Voice design description (e.g. `A clear, friendly, articulate assistant`).
- `STT_ENGINE`: Speech-to-text engine (default: `whisper_local`).
- `STT_ALLOW_CLOUD`: Defaults to `false`. Set to `true` only if you explicitly choose to allow cloud speech recognition (e.g. Google Speech API).
- `DISCORD_VOICE_ALLOWED_USER_IDS`: Comma-separated list of Discord user IDs authorized to use voice call features.

### Frontend Voice Controls (Web UI)
1. **"Speak replies" Toggle Button**:
   - Located in the Assistant subheader beside chat controls.
   - When enabled, incoming assistant messages are automatically synthesized and spoken aloud sentence-by-sentence via an audio queue with seamless prefetching.
2. **"Listen" Button (Speech to Text)**:
   - Located in the chat composer bar.
   - Click to speak; browser captures audio via AudioWorklet (16 kHz mono 16-bit PCM) and sends it for speech transcription. Transcribed text is placed directly in the composer.
3. **"Stop" Button**:
   - Appears whenever audio is synthesizing or playing.
   - Immediately cancels the current turn, halts speech playback, and clears transient buffers.
4. **"Live" Mode (Gemini Live Experience)**:
   - Click the **Live** button in the Assistant header to open the full-screen interactive voice session.
   - Hands-free, continuous conversation with real-time barge-in detection (speaking interrupts the bot immediately).
   - Dynamic animated voice orb reflecting states: `IDLE`, `LISTENING`, `THINKING`, `SPEAKING`.
   - Real-time turn latency metrics and headphones recommendation for echo prevention.

### Discord Voice Call (Text In, Voice Out)
S.P.I.D.E.Y can join a Discord voice channel and speak answers while you type messages in text:
- **Join Call**: In any Discord server channel, join a voice channel and type:
  ```text
  !call join
  ```
  S.P.I.D.E.Y joins your voice channel and binds voice responses to the current text channel.
- **Text In, Voice Out**: Type any message (e.g. `!ask What's on my schedule today?`). The bot replies in text AND speaks the response in the voice channel. Spoken output is capped at 500 characters for clarity.
- **Voice Interruption**: Type `stop`, `stop talking`, or `!call stop` to immediately silence current speech.
- **Toggle Voice**: `!call voice off` disables speech output (text only); `!call voice on` re-enables it.
- **Leave Call**: Type `!call leave` to disconnect the bot from voice.
- **Encryption Requirement**: Uses Discord's new DAVE End-to-End Voice Encryption powered by `discord.py[voice]` (`PyNaCl` + `davey`).

### Voice Troubleshooting
- **Microphone access denied in browser**: Ensure your browser allows microphone permissions on `http://localhost:5173`.
- **CUDA Out of Memory during voice synthesis**: Set `TTS_DEVICE=cpu` in `.env` to run TTS entirely on CPU, or let the automatic fallback take effect.
- **Discord voice audio not playing**: Verify that `PyNaCl` and `davey` are installed via `pip install "discord.py[voice]"` and that the bot has `Connect` and `Speak` permissions in the voice channel.

## Setup

Use Python 3.10, 3.11, or 3.12 in a virtual environment. Kokoro 0.9.x
requires Python 3.10 through 3.12; Python 3.13 and 3.14 cannot install it.
The current machine's Python 3.14 environment must be replaced, not upgraded
with another `pip` command.

```powershell
# Check available interpreters. The command below assumes Python 3.11 is listed.
py -0p
if (Test-Path .venv) { Remove-Item -LiteralPath .venv -Recurse -Force }
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

The LLM uses Qwen's tokenizer and causal language model interface for
text-only input.

For NVIDIA GPU acceleration, install the CUDA-enabled PyTorch build for your
system from the official PyTorch selector at https://pytorch.org/get-started/locally/
after activating the environment. A normal PyPI install can leave you with a
CPU-only build. Verify the result before running the app:

```powershell
python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available())"
```

The final value should be `True` for the recommended 4-bit GPU mode. The app
stops with an actionable error if CUDA is unavailable instead of silently
running the LLM on the CPU. To explicitly allow slow CPU mode for
troubleshooting, use:

```powershell
$env:AI_ALLOW_CPU_LLM = "1"
python app.py
```

If `py -3.11` is not available, install Python 3.11 or 3.12 from
https://www.python.org/downloads/windows/ and select **Add python.exe to
PATH**. Then reopen PowerShell and repeat the commands above, replacing
`3.11` with the installed version.

Kokoro also requires the `espeak-ng` system executable for phonemization. On
Windows, install it separately and make sure `espeak-ng` is available on
`PATH`.

The Qwen3 checkpoint is several gigabytes. The application loads it only from
the local `Models/LLM/` directory and does not download a second LLM. On the
6 GB RTX 4050 profile, it uses 4-bit NF4 bitsandbytes quantization, reserves
up to approximately 5 GiB of GPU memory and 6 GiB of CPU memory, and does not
use disk offload. If these limits are insufficient, startup stops with an
error instead of exhausting system RAM or paging the model to disk.

The local checkpoint is **Qwen3-1.7B** (`Qwen/Qwen3-1.7B`), with about
1.7 billion parameters. The LLM uses CUDA when available, while
Kokoro voice synthesis is deliberately kept on the CPU.

### Laptop resource safeguards

The default settings are conservative for an RTX 4050 6 GB laptop GPU:

- The LLM uses 4-bit NF4 quantization and is limited to about 5 GiB of VRAM,
  leaving room for the display driver and generation workspace.
- CPU placement is limited to 6 GiB. Disk offload is disabled because paging
  the model on a 16 GB system caused RAM exhaustion and 100% SSD activity.
- Text generation is capped at 128 new tokens and prompts at 2,000 characters.
- Kokoro runs on the CPU so it cannot compete with the LLM for GPU memory.
- CPU inference is limited to four threads and one inter-op thread.
- CUDA temporary allocations are configured to use expandable segments, which
  helps reduce memory fragmentation.
- Startup prints the model device map, placement distribution, load time, and
  GPU memory allocation. Disk placement is reported as a warning and should
  normally be absent.

These limits may make generation slower, but they reduce the likelihood of a
GPU out-of-memory crash. Close other GPU-heavy applications before starting the
program, and use a shorter prompt if the program reports an out-of-memory
error.

## Run

Start the FastAPI HTTP server. The local Qwen model is loaded once when this
process starts, so restarting the server loads the model again; requests reuse
the already-loaded model:

```powershell
python app.py --server
```

For backend development, enable automatic reload explicitly:

```powershell
python -m uvicorn app:app --host 127.0.0.1 --port 8000 --reload
```

Use the direct Uvicorn command for reload mode. It prevents the launcher
process from loading the model before Uvicorn starts its reload worker. Each
reload still starts a fresh worker and loads the model once for that worker;
normal requests never reload the model. Do not use `--reload` in production.

The server listens on `http://127.0.0.1:8000`. Check its status:

```powershell
Invoke-RestMethod http://127.0.0.1:8000/health
```

Generate a response:

```powershell
Invoke-RestMethod -Method Post `
  -Uri http://127.0.0.1:8000/generate `
  -ContentType "application/json" `
  -Body '{"prompt":"Explain RAM usage in one sentence","max_new_tokens":64}'
```

Use `--host` and `--port` to change the bind address or port. Keep
`--host 127.0.0.1` when the service should remain local.

The original terminal interface is still available:

```powershell
python app.py
```

Enter a request at the `You:` prompt. The
prompt can also be passed directly for one-shot usage:

```powershell
python app.py "Explain RAM usage in one sentence"
```

Responses are limited to 128 new tokens by default. Use
`--max-new-tokens` to select a lower limit, for example:

```powershell
python app.py --max-new-tokens 64 "Write a short joke"
```

Type `exit` or `quit` to stop interactive mode. The
default Kokoro voice is `af_heart`; all available local voices are the `.pt`
files in `Models/voice/voices/`.

## Unified React UI

The single frontend in `frontend/` is the S.P.I.D.E.Y workspace: a
productivity dashboard and a personal assistant in one panel. It includes
habits, analytics, tasks, local chat, reminders, notifications, memories,
streaks, XP, and achievements.

Install the frontend dependencies and start the Vite development server in a
second terminal while the FastAPI server is running:

```powershell
cd frontend
npm install
npm run dev
```

Open `http://localhost:5173` in a browser.

The frontend connects directly to FastAPI by default at
`http://127.0.0.1:8000`. To use another backend address, create
`frontend/.env.local` with:

```text
VITE_SPIDEY_API_BASE_URL=http://your-backend:8000
```

FastAPI allows the local development origins and exposes both `/health` and
`/api/health`.

Without `--reload`, after changing backend routes or reminder behavior, restart
the FastAPI process:

```powershell
# Stop any old FastAPI process with Ctrl+C, then run this from the project root:
.\.venv\Scripts\python.exe app.py --server
```

If the frontend reports an HTML/JSON parsing error, confirm that the FastAPI
server is running on port 8000 and that the Vite development server is running
from the `frontend` directory. Use the project `.venv` interpreter rather than
an IDE or system Python installation, because the latter may not contain
`dateparser` and the other backend dependencies. API errors are returned as
JSON by the backend so the dashboard can display the actual failure.

## Persistent chats and memories

The application now stores chats, messages, and explicitly saved personal
facts in `assistant.db` beside `app.py`. The database is created and migrated
automatically at startup. Recent messages are bounded by a token budget before
being passed through Qwen's chat template, so separate chats do not share
short-term context.

The React sidebar can create, switch, and delete chats and can save or delete
explicit memories. Memory retrieval is local and currently uses a bounded
keyword fallback. The optional `sentence-transformers` dependency is included
for a future local semantic retriever; no cloud embedding or vector database
is used. Do not treat every conversation message as a personal memory.

The dashboard also exposes the SQLite-backed task and reminder APIs. These
records are persistent, but reminder execution is not yet an always-on worker;
the FastAPI process must remain running for delivery. Reminders created from chat
are attached to that conversation and appear as new S.P.I.D.E.Y messages when they
fire. For example:

```text
Remind me to study Java at 7 PM
Remind me to review goals at 8 AM, every day
Remind me to stretch at 6 PM, every 2 hours
Set a reminder in 1 minute for to start the learning course
Hey Spidey, remind me tomorrow morning to water my plants
```

When a reminder uses a day part without a clock time, the scheduler uses
6:30 AM for morning, 12:30 PM for afternoon, 6:30 PM for evening, and
9:00 PM for night.

The frontend polls the notification endpoint every five seconds. When a
reminder fires, the local LLM writes a short, natural S.P.I.D.E.Y notification using
the reminder text and the current system date/time. If LLM generation fails,
the reminder still appears with a direct fallback notification. If the FastAPI
process or computer is stopped, delivery waits until a future worker is added.

Task commands are also handled deterministically before Qwen:

```text
Create task Finish the Python project priority high
Show my pending tasks
Complete task 1
Remind me in 10 minutes to take a break
Check my habits to complete
What habits do I need to complete today?
```

Completing a task from the dashboard or the S.P.I.D.E.Y assistant marks it complete.
Completed tasks remain available in the task history and completed filter until
you delete them.

Habits are different: they are persistent protocols. Creating or editing a
habit changes the protocol itself, while completing it records the selected
date in `habit_completions`; the same habit can therefore be completed again
on the next day without creating a duplicate habit.

Additional API routes:

```text
GET    /api/chats
POST   /api/chats
GET    /api/history/<chat_id>
DELETE /api/chats/<chat_id>
DELETE /api/chats
GET    /api/memories
POST   /api/memories              {"content":"My project is Qwen Assistant"}
PUT    /api/memories/<memory_id>  {"content":"Updated fact"}
DELETE /api/memories/<memory_id>
GET    /api/tasks?include_completed=1
POST   /api/tasks             {"title":"Study Java","priority":"high"}
PATCH  /api/tasks/<task_id>   {"completed":true}
DELETE /api/tasks/<task_id>
GET    /api/habits
POST   /api/habits             {"name":"Read","icon":"BookOpen","daily_goal":1}
PATCH  /api/habits/<habit_id>  {"name":"Read daily","icon":"BookOpen","daily_goal":1}
DELETE /api/habits/<habit_id>
PUT    /api/habits/<habit_id>/completions/<YYYY-MM-DD> {"completed":true}
GET    /api/gamification
GET    /api/reminders
POST   /api/reminders         {"text":"Study Java","run_at":"2026-10-24T13:30:00Z","chat_id":"..."}
DELETE /api/reminders/<id>
GET    /api/notifications/<chat_id>?after_id=0
```

The existing `POST /generate` contract remains compatible. It accepts the
original `prompt` and `max_new_tokens` fields and also accepts `chat_id`; the
response includes the selected chat ID and persisted messages. To remove all
local history and memories, stop the server and delete `assistant.db`.

## Discord bot

The first Discord integration keeps inference local: Discord messages are
handled by the backend and passed to the existing Qwen model. The bot stores a
separate conversation per Discord user in `assistant.db`; it does not expose
the local FastAPI API or send prompts to a cloud provider.

1. Create a Discord application and bot in the Discord Developer Portal.
2. Enable the **Message Content Intent** for the bot.
3. Invite it to your server with only the permissions it needs: **View
   Channels**, **Send Messages**, **Send Messages in Threads**, and **Read
   Message History** (`permissions=274877975552`). No administrator access is
   required.
4. Install dependencies, copy the environment template, and paste the bot
   token into it:

```powershell
python -m pip install -r requirements.txt
Copy-Item .env.example .env
notepad .env
```

`.env` holds `AI_DISCORD_APPLICATION_ID`, `AI_DISCORD_PUBLIC_KEY`, and
`AI_DISCORD_TOKEN`; [app.py](app.py) loads it on startup, so both the API and
the bot see the same values. The token is the only required value, and it must
stay out of source control and all frontend variables. Then start the bot:

```powershell
python discord_bot.py
```

Keep exactly one bot process running. A second copy that shares the token — for
example one from an IDE console and one from a terminal — competes for the same
gateway session and causes missed replies. The bot also refuses to start when
its local lock port is already taken.

Reminders created in Discord are delivered to you as DMs. The bot hosts the
reminder scheduler for Discord chats, and a reminder that was missed by up to
30 minutes is delivered when the bot starts again.

Use `!ask <message>`, mention the bot, or send any other `!command` such as
`!status` or `!add task ...`; unregistered commands are passed to the same
assistant pipeline as the web chat. Restricting `AI_DISCORD_ALLOWED_USER_IDS`
is strongly recommended, since an empty allowlist lets anyone use the bot. You
can also set `AI_DISCORD_ALLOWED_GUILD_IDS` to limit it to selected servers.

## Notes

Kokoro's Python package may cache its compatible model files on first use.
The application validates that the selected voice file exists locally before
starting synthesis. Generated audio is overwritten at `output.wav` after each
successful response.
