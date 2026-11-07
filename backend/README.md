# SMS Spam Checker: FastAPI backend + UI

Shows the **base** and **QLoRA fine-tuned** Qwen2.5-0.5B decisions side by side for any SMS.
Uses the same prompt and the same scoring rule (`logit(spam) > logit(ham)`) as the notebook.

## 1. Get the adapter from Colab
After running the notebook, the last cells zip and download `adapter.zip`.
Unzip it so this folder exists:

```
backend/
  app.py
  index.html
  requirements.txt
  qlora-sms-spam-adapter/        <-- from Colab (contains adapter_config.json, adapter_model.safetensors, ...)
```

Without the adapter the server still starts, but shows only the base model (and a warning banner).

## 2. Install and run (Windows PowerShell)
Use **Python 3.11 or 3.12**. PyTorch often has no wheels yet for the newest Python (this PC has 3.14).

```powershell
cd C:\Users\Administrator\Desktop\Project\backend
py -3.12 -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
uvicorn app:app --port 8000
```

Open http://127.0.0.1:8000 for the UI, or http://127.0.0.1:8000/docs for the API docs.
The first start downloads the base model (about 1 GB) from Hugging Face. It runs on CPU if there is no GPU.

## API
| Method | Path | Purpose |
|---|---|---|
| GET | `/health` | model, device, whether the adapter is loaded |
| POST | `/predict` | body `{"text": "..."}` returns base and fine-tuned verdicts |

```powershell
curl -X POST http://127.0.0.1:8000/predict -H "Content-Type: application/json" -d "{\"text\": \"WIN a free prize, text 80085 now\"}"
```

`spam_probability` is `sigmoid(logit(spam) - logit(ham))`, i.e. P(spam) between the two answers.

## Note
The adapter was trained on a 4-bit quantised base in Colab. Here the base is loaded in normal precision
(float16 on GPU, float32 on CPU), the usual way to serve a QLoRA adapter. Tiny score differences versus
the notebook are expected.
