# S.P.I.D.E.Y (ORBIT007)

S.P.I.D.E.Y is a local AI assistant with Ollama chat, browser voice, and
Discord voice integration.

- **LLM:** Ollama `qwen3.5:4b`
- **TTS:** Gradium Phonon `phonon-31m-en` through `ptts`, CPU-only, with Windows
  SAPI as the automatic fallback
- **STT:** local Whisper / SpeechRecognition
- **Backend:** Python FastAPI

## Quick start on Windows

Install Ollama and pull the model:

```powershell
ollama pull qwen3.5:4b
ollama serve
```

Then run `start.bat`. It activates the available virtual environment, starts
the backend and frontend, and uses Phonon on CPU.

## Backend installation

```powershell
py -3.14 -m venv .venv314
.\.venv314\Scripts\python.exe -m pip install --upgrade pip
.\.venv314\Scripts\python.exe -m pip install -r requirements.gpu.txt
```

Phonon weights must be downloaded once with Hugging Face access enabled:

```powershell
$env:SPIDEY_ALLOW_HUB = "1"
.\.venv314\Scripts\python.exe -m ptts --lang en --quant q8 --voice Easton "Hello from Spidey" -o test.wav
```

After the weights are cached, normal startup may use offline mode. Configure
the backend with `TTS_BACKEND=phonon` (default), or `TTS_BACKEND=sapi` to force
Windows SAPI.

## TTS configuration

```env
TTS_BACKEND=phonon
PHONON_MODEL=hf://gradium/phonon-31m-en
PHONON_QUANT=q8
PHONON_LANG=en
SPIDEY_TTS_VOICE=Easton
```

The voice service synthesizes 24 kHz mono float32 audio. Browser playback
receives WAV audio, while Discord resamples it to 48 kHz stereo PCM.

## Tests

```powershell
.\.venv314\Scripts\python.exe -m pytest -q
```

The first Phonon load may download weights and take several minutes. If Phonon
cannot load, Windows SAPI remains available and the server continues running.
