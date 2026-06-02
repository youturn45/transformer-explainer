# Chinese GPT-2 Swap Design

**Date:** 2026-06-02
**Scope:** Replace the base English `gpt2` model with `uer/gpt2-chinese-cluecorpussmall` across the full pipeline — Python export, ONNX chunking, browser tokenizer, and Svelte store.

---

## Decision Summary

- **Replacement strategy:** Full swap (Option A) — Chinese model replaces English `gpt2` entirely; no model switcher UI needed.
- **Model loader strategy:** Patch `from_pretrained` in `model.py` (Approach A) — make it accept any HF GPT-2-compatible model by reading config dynamically instead of hardcoding the 4 standard variants.

---

## Architecture

The pipeline is unchanged in structure; only the model and tokenizer identity changes at each stage:

```
HuggingFace (uer/gpt2-chinese-cluecorpussmall)
    ↓  model.py: from_pretrained (patched)
Custom GPT class (model.py)
    ↓  export_to_onnx.py
gpt2-chinese.onnx
    ↓  chunk.py
static/model-v2/gpt2-chinese.onnx.partN  (N chunks × 10MB)
    ↓  fetchChunks.js (browser)
ONNX InferenceSession (onnxruntime-web)
    ↑  tokenizer
AutoTokenizer.from_pretrained('uer/gpt2-chinese-cluecorpussmall')  [@xenova/transformers]
```

---

## Changes

### 1. `src/utils/model/model.py`

**`GPTConfig`**
- Remove hardcoded `vocab_size: int = 50304` default — it will be set dynamically per model.

**`GPT.from_pretrained`**
- Remove `assert model_type in {'gpt2', 'gpt2-medium', 'gpt2-large', 'gpt2-xl'}`.
- Remove the hardcoded `config_args` dict for the 4 standard variants.
- Load the HF model first: `model_hf = GPT2LMHeadModel.from_pretrained(model_type)`.
- Read `n_layer`, `n_head`, `n_embd`, `vocab_size` from `model_hf.config` directly.
- Remove the hardcoded `vocab_size=50257` and `block_size=1024` overrides — read from HF config (`n_positions` for block_size).
- Keep the Conv1D transpose logic unchanged — UER's model is stored in standard HF `GPT2LMHeadModel` format which still uses Conv1D weights.

### 2. `src/utils/model/export_to_onnx.py`

- Change `modelname` to `"uer/gpt2-chinese-cluecorpussmall"`.
- Add a sanitized filename variable: `filename = "gpt2-chinese"` used for the ONNX output path.
- Update `dummy_input` to use valid token IDs from the Chinese vocab (e.g. a short sequence of character IDs within the 21128 vocab range).
- Update `onnx_model_path` to use `filename` instead of `modelname`.

### 3. `src/utils/model/chunk.py`

- Change `modelname` to `"gpt2-chinese"` (the sanitized filename).
- Input: `src/utils/model/params_output/gpt2-chinese.onnx`
- Output: `static/model-v2/gpt2-chinese.onnx.partN`

### 4. `src/store/index.ts`

- Remove existing `gpt2`, `gpt2-medium`, `gpt2-large` entries from `modelMetaMap`.
- Add entry to `modelMetaMap`:
  ```ts
  'gpt2-chinese': { layer_num: 12, attention_head_num: 12, dimension: 768, chunkTotal: <N> }
  ```
  `chunkTotal` is filled in after the ONNX export and chunking step.
- Change `initialSelectedModel` from `'gpt2'` to `'gpt2-chinese'`.
- Replace `inputTextExample` with:
  ```ts
  export const inputTextExample = [
    '床前明月光，疑是',
    '春眠不觉晓，处处',
    '世上本没有路，走的人多了，也便',
    '横眉冷对千夫指，俯首甘为',
    '臣妾很想知足，可臣妾',
    '黑夜给了我黑色的眼睛，我却用它',
  ];
  ```

### 5. `src/routes/+page.svelte`

- Change tokenizer: `AutoTokenizer.from_pretrained('uer/gpt2-chinese-cluecorpussmall')`.
  - `@xenova/transformers` fetches tokenizer files (vocab, config) from HuggingFace CDN at runtime — no offline conversion needed.
- Update `fetchModel`:
  - Change chunk filename from `gpt2.onnx.partN` to `gpt2-chinese.onnx.partN`.
  - Update `chunkNum` to match the actual chunk count after export.

---

## Model Details

| Property | Value |
|---|---|
| HF model ID | `uer/gpt2-chinese-cluecorpussmall` |
| Architecture | GPT-2 (12 layers, 12 heads, 768 dim) |
| Vocab size | 21128 (Chinese BERT vocab, character-level) |
| Tokenizer type | BertTokenizer (supported by @xenova/transformers) |
| Block size | 1024 |

---

## Example Prompts

| Text | Source |
|---|---|
| `床前明月光，疑是` | Li Bai《静夜思》|
| `春眠不觉晓，处处` | Meng Haoran《春晓》|
| `世上本没有路，走的人多了，也便` | 鲁迅《故乡》|
| `横眉冷对千夫指，俯首甘为` | 鲁迅《自嘲》|
| `臣妾很想知足，可臣妾` | 甄嬛传 |
| `黑夜给了我黑色的眼睛，我却用它` | 顾城《一代人》|

---

## Implementation Order

1. Patch `model.py` → `from_pretrained`
2. Update `export_to_onnx.py` → run export to produce `gpt2-chinese.onnx`
3. Update `chunk.py` → run chunking to produce `static/model-v2/gpt2-chinese.onnx.partN`
4. Update `store/index.ts` with chunk count, new model meta, and example prompts
5. Update `+page.svelte` tokenizer and chunk URLs

---

## Out of Scope

- Model switcher UI (full replacement only)
- Quantization of Chinese model (can be done separately via `quantize.py`)
- Keeping English `gpt2` assets in the repo
