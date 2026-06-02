# Quickstart

This is a fork of [Transformer Explainer](https://poloclub.github.io/transformer-explainer/) adapted to run `uer/gpt2-chinese-cluecorpussmall` — a Chinese GPT-2 model — in the browser.

---

## Prerequisites

- [Node.js](https://nodejs.org/) v18+
- [uv](https://docs.astral.sh/uv/) (Python package manager)
- Python 3.10+

---

## 1. Install Node dependencies

```bash
npm install --legacy-peer-deps
```

---

## 2. Set up Python environment

Create a virtual environment and install the packages needed for the model export pipeline:

```bash
uv venv .venv
source .venv/bin/activate

uv pip install torch --index-url https://download.pytorch.org/whl/cpu
uv pip install transformers onnx onnxscript httpx[socks]
```

> `httpx[socks]` is only needed if you're behind a SOCKS proxy.
> Use the CPU-only torch wheel to keep the install size manageable (~300MB vs ~2GB for CUDA).

---

## 3. Export the model to ONNX

Run from the **repo root**:

```bash
python src/utils/model/export_to_onnx.py
```

This downloads `uer/gpt2-chinese-cluecorpussmall` from HuggingFace (~400MB), loads it into the custom GPT class, and exports it to:

```
src/utils/model/params_output/gpt2-chinese.onnx
```

Expected output:
```
loading weights from pretrained gpt: uer/gpt2-chinese-cluecorpussmall
number of parameters: 101.28M
Model has been successfully exported to ONNX format.
```

---

## 4. Split the ONNX into browser-friendly chunks

The browser loads the model in 10MB chunks. Run from the **repo root**:

```bash
python src/utils/model/chunk.py
```

This produces 46 files at `static/model-v2/gpt2-chinese.onnx.partN`.

> The `static/model-v2/` folder is gitignored — you must run this step on every fresh clone.

---

## 5. Download the tokenizer

The tokenizer files are already committed to `static/tokenizer/`. If you need to regenerate them:

```bash
source .venv/bin/activate

PYTHONPATH=.venv/lib/python3.12/site-packages python3 -c "
from transformers import AutoTokenizer
import os, shutil

dest = 'static/tokenizer/uer/gpt2-chinese-cluecorpussmall'
os.makedirs(dest, exist_ok=True)
t = AutoTokenizer.from_pretrained('uer/gpt2-chinese-cluecorpussmall')
t.save_pretrained(dest)
print('Saved:', os.listdir(dest))
"
```

---

## 6. Start the dev server

```bash
npm run dev
```

Open `http://localhost:5173`. The app will load the 46 ONNX chunks (~460MB total) from the local server on first visit — this takes a few seconds. Subsequent visits use the browser cache.

---

## 7. Build for production

```bash
npm run build
npm run preview
```

---

## Python packages reference

| Package | Purpose |
|---|---|
| `torch` | Load and run the GPT-2 model |
| `transformers` | Load `uer/gpt2-chinese-cluecorpussmall` via HuggingFace |
| `onnx` | ONNX model format support |
| `onnxscript` | Required by PyTorch 2.9+ ONNX export |
| `httpx[socks]` | SOCKS proxy support for HuggingFace downloads |

---

## File overview

```
src/utils/model/
  model.py              # Custom GPT class (loads any HF GPT-2-compatible model)
  export_to_onnx.py     # Exports model to ONNX (run once)
  chunk.py              # Splits ONNX into 10MB chunks for the browser (run once)
  quantize.py           # Optional: quantize to INT8 for smaller size
  params_output/        # Generated ONNX output — gitignored, regenerate locally

static/
  model-v2/             # 46 × 10MB ONNX chunks served to the browser
  tokenizer/            # Tokenizer files served locally (no HuggingFace CDN needed)
```
