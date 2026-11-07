"""FastAPI backend for the SMS spam classifier (base vs QLoRA fine-tuned Qwen2.5-0.5B).

It uses exactly the same prompt and scoring rule as the notebook:
    spam  if  logit("spam") > logit("ham")  at the first answer token.

Run:  uvicorn app:app --port 8000      then open http://127.0.0.1:8000
"""
import math
import os
import threading
from contextlib import asynccontextmanager
from pathlib import Path

import torch
from fastapi import FastAPI
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field
from transformers import AutoModelForCausalLM, AutoTokenizer

HERE = Path(__file__).parent
MODEL_NAME = os.getenv("BASE_MODEL", "Qwen/Qwen2.5-0.5B-Instruct")


def find_adapter_dir() -> Path:
    """ADAPTER_DIR env var wins; otherwise use the first folder that contains adapter_config.json."""
    if os.getenv("ADAPTER_DIR"):
        return Path(os.environ["ADAPTER_DIR"])
    candidates = [HERE / "qlora-sms-spam-adapter",
                  HERE / "adapter" / "qlora-sms-spam-adapter",
                  HERE / "adapter"]
    return next((c for c in candidates if (c / "adapter_config.json").exists()), candidates[0])


ADAPTER_DIR = find_adapter_dir()
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.float16 if DEVICE == "cuda" else torch.float32

# Must be identical to the notebook (Section 3), otherwise results will not match.
SYSTEM = ("You are an SMS spam filter. Classify the SMS as 'spam' (unwanted promotions, "
          "prizes, scams, phishing, premium-rate services) or 'ham' (a legitimate, normal "
          "message). Answer with exactly one word: spam or ham.")

state = {}                       # tok, model, adapter_loaded, spam_id, ham_id
lock = threading.Lock()          # disable_adapter() changes model state -> one request at a time


def build_prompt(text: str) -> str:
    messages = [{"role": "system", "content": SYSTEM},
                {"role": "user", "content": f"SMS: {text[:600]}"}]
    return state["tok"].apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


@torch.no_grad()
def spam_margin(text: str) -> float:
    """logit(spam) - logit(ham) for the next token after the prompt."""
    enc = state["tok"](build_prompt(text), return_tensors="pt", add_special_tokens=False).to(DEVICE)
    logits = state["model"](**enc).logits[0, -1].float()
    return float(logits[state["spam_id"]] - logits[state["ham_id"]])


def verdict(margin: float) -> dict:
    return {"label": "spam" if margin > 0 else "ham",
            "margin": round(margin, 4),
            "spam_probability": round(1.0 / (1.0 + math.exp(-margin)), 4)}   # P(spam | spam or ham)


@asynccontextmanager
async def lifespan(app: FastAPI):
    print(f"Loading {MODEL_NAME} on {DEVICE} ...")
    tok = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, torch_dtype=DTYPE).to(DEVICE)

    adapter_loaded = False
    if (ADAPTER_DIR / "adapter_config.json").exists():
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, str(ADAPTER_DIR)).to(DEVICE)
        adapter_loaded = True
        print(f"Loaded LoRA adapter from {ADAPTER_DIR}")
    else:
        print(f"WARNING: no adapter at {ADAPTER_DIR}. Running base model only.")
    model.eval()

    state.update(tok=tok, model=model, adapter_loaded=adapter_loaded,
                 spam_id=tok("spam", add_special_tokens=False).input_ids[0],
                 ham_id=tok("ham", add_special_tokens=False).input_ids[0])
    assert state["spam_id"] != state["ham_id"]
    yield
    state.clear()


app = FastAPI(title="SMS Spam Classifier: Base vs QLoRA", lifespan=lifespan)


class PredictIn(BaseModel):
    text: str = Field(..., min_length=1, max_length=2000, description="The SMS message")


@app.get("/health")
def health():
    return {"status": "ok", "model": MODEL_NAME, "device": DEVICE,
            "adapter_loaded": state.get("adapter_loaded", False)}


@app.post("/predict")
def predict(body: PredictIn):
    text = body.text.strip()
    with lock:
        if state["adapter_loaded"]:
            with state["model"].disable_adapter():          # adapter off = the base model
                base = spam_margin(text)
            finetuned = spam_margin(text)                    # adapter on
        else:
            base, finetuned = spam_margin(text), None
    return {"text": text,
            "base": verdict(base),
            "finetuned": verdict(finetuned) if finetuned is not None else None,
            "adapter_loaded": state["adapter_loaded"]}


@app.get("/")
def index():
    return FileResponse(HERE / "index.html")
