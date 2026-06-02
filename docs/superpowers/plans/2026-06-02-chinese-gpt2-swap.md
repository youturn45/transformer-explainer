# Chinese GPT-2 Swap Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the base English `gpt2` model with `uer/gpt2-chinese-cluecorpussmall` across the full stack — Python export pipeline, ONNX chunking, browser tokenizer, Svelte store, and UI copy.

**Architecture:** The existing pipeline structure is unchanged — only the model identity changes at each layer. `model.py`'s `from_pretrained` is patched to accept any HF GPT-2-compatible model by reading config dynamically. The frontend tokenizer swaps from `Xenova/gpt2` (BPE, English) to `uer/gpt2-chinese-cluecorpussmall` (BertTokenizer, Chinese). Stale English cached example data is removed entirely.

**Tech Stack:** Python + PyTorch + HuggingFace Transformers (export pipeline), SvelteKit + TypeScript + @xenova/transformers v2.17.x + onnxruntime-web (frontend).

---

## File Map

| File | Change |
|---|---|
| `src/utils/model/model.py` | Patch `from_pretrained` to read config dynamically from any HF model |
| `src/utils/model/export_to_onnx.py` | Set model name + sanitized filename + valid Chinese dummy input |
| `src/utils/model/chunk.py` | Update model name to `gpt2-chinese` |
| `src/utils/data.ts` | Fix `modelMetaMap.gpt2` hard-coded refs; add `add_special_tokens: false` |
| `src/store/index.ts` | New model meta entry, updated initial model/text/prompts, remove ex0 usage |
| `src/routes/+page.svelte` | New tokenizer, null guards, remove stale cache fallback, update chunk URLs |
| `src/utils/textbookPages.ts` | Update vocab count and model name strings |
| `src/components/InputForm.svelte` | Update GPT-2 prompt text |
| `src/components/article/Article.svelte` | Update vocab counts and model name references |
| `src/components/Popovers/LogitWeightPopover.svelte` | Update hard-coded 50,257 dimension labels |

---

## Task 1: Patch `model.py` to accept any HF GPT-2 model

**Files:**
- Modify: `src/utils/model/model.py`

- [ ] **Step 1: Update `GPTConfig` default vocab_size**

In `src/utils/model/model.py`, change the `GPTConfig` dataclass. Replace:

```python
@dataclass
class GPTConfig:
    block_size: int = 1024
    vocab_size: int = 50304 # GPT-2 vocab_size of 50257, padded up to nearest multiple of 64 for efficiency
    n_layer: int = 12
    n_head: int = 12
    n_embd: int = 768
    dropout: float = 0.0
    bias: bool = True # True: bias in Linears and LayerNorms, like GPT-2. False: a bit better and faster
```

With:

```python
@dataclass
class GPTConfig:
    block_size: int = 1024
    vocab_size: int = 50257
    n_layer: int = 12
    n_head: int = 12
    n_embd: int = 768
    dropout: float = 0.0
    bias: bool = True
```

- [ ] **Step 2: Replace `from_pretrained` method**

Replace the entire `from_pretrained` classmethod (lines 298–353) with:

```python
@classmethod
def from_pretrained(cls, model_type, override_args=None):
    override_args = override_args or {}
    assert all(k == 'dropout' for k in override_args)
    from transformers import GPT2LMHeadModel
    print("loading weights from pretrained gpt: %s" % model_type)

    model_hf = GPT2LMHeadModel.from_pretrained(model_type)

    config_args = dict(
        n_layer=model_hf.config.n_layer,
        n_head=model_hf.config.n_head,
        n_embd=model_hf.config.n_embd,
        vocab_size=model_hf.config.vocab_size,
        block_size=model_hf.config.n_positions,
        bias=True,
    )
    if 'dropout' in override_args:
        config_args['dropout'] = override_args['dropout']

    config = GPTConfig(**config_args)
    model = GPT(config)
    sd = model.state_dict()
    sd_keys = [k for k in sd.keys() if not k.endswith('.attn.bias')]

    sd_hf = model_hf.state_dict()
    sd_keys_hf = [k for k in sd_hf.keys() if not k.endswith('.attn.masked_bias')]
    sd_keys_hf = [k for k in sd_keys_hf if not k.endswith('.attn.bias')]
    transposed = ['attn.c_attn.weight', 'attn.c_proj.weight', 'mlp.c_fc.weight', 'mlp.c_proj.weight']
    assert len(sd_keys_hf) == len(sd_keys), f"mismatched keys: {len(sd_keys_hf)} != {len(sd_keys)}"
    for k in sd_keys_hf:
        if any(k.endswith(w) for w in transposed):
            assert sd_hf[k].shape[::-1] == sd[k].shape
            with torch.no_grad():
                sd[k].copy_(sd_hf[k].t())
        else:
            assert sd_hf[k].shape == sd[k].shape
            with torch.no_grad():
                sd[k].copy_(sd_hf[k])

    return model
```

- [ ] **Step 3: Commit**

```bash
git add src/utils/model/model.py
git commit -m "feat: patch model.py from_pretrained to accept any HF GPT-2 model"
```

---

## Task 2: Update export and chunk scripts

**Files:**
- Modify: `src/utils/model/export_to_onnx.py`
- Modify: `src/utils/model/chunk.py`

- [ ] **Step 1: Update `export_to_onnx.py`**

Replace the top of `src/utils/model/export_to_onnx.py`. Change:

```python
modelname="gpt2"
```

To:

```python
modelname = "uer/gpt2-chinese-cluecorpussmall"
filename = "gpt2-chinese"
```

Then change the dummy input line:

```python
dummy_input = torch.tensor([[6601, 32704, 795, 30132, 2985, 284]])
```

To:

```python
dummy_input = torch.tensor([[1435, 1184, 3209, 3362, 2769, 1526]])
```

Then change the ONNX output path line:

```python
onnx_model_path = "src/utils/model/params_output/"+ modelname +".onnx"
```

To:

```python
onnx_model_path = "src/utils/model/params_output/" + filename + ".onnx"
```

- [ ] **Step 2: Update `chunk.py`**

In `src/utils/model/chunk.py`, change:

```python
modelname="gpt2"
```

To:

```python
modelname="gpt2-chinese"
```

- [ ] **Step 3: Commit**

```bash
git add src/utils/model/export_to_onnx.py src/utils/model/chunk.py
git commit -m "feat: update export and chunk scripts for Chinese GPT-2"
```

---

## Task 3: Run the Python export pipeline

> This task produces the ONNX artifact that the frontend loads. It requires Python with `torch`, `transformers`, and `onnx` installed. The model download from HuggingFace is ~400MB. **Run all commands from the repo root.**

**Files:**
- Produces: `src/utils/model/params_output/gpt2-chinese.onnx`
- Produces: `static/model-v2/gpt2-chinese.onnx.partN`

- [ ] **Step 1: Export the model to ONNX**

Run from the repo root (scripts use repo-root-relative paths):

```bash
python src/utils/model/export_to_onnx.py
```

Expected output:
```
loading weights from pretrained gpt: uer/gpt2-chinese-cluecorpussmall
number of parameters: 102.07M
Model has been successfully exported to ONNX format.
```

The file `src/utils/model/params_output/gpt2-chinese.onnx` should now exist.

- [ ] **Step 2: Split into chunks**

```bash
python src/utils/model/chunk.py
```

Expected: files `static/model-v2/gpt2-chinese.onnx.part0` through `static/model-v2/gpt2-chinese.onnx.partN` are created.

- [ ] **Step 3: Count the chunks**

```bash
ls static/model-v2/gpt2-chinese.onnx.part* | wc -l
```

Note this number — you will use it as `chunkTotal` in Task 5 and as `chunkNum` in Task 6.

- [ ] **Step 4: Remove old English model chunks**

```bash
rm static/model-v2/gpt2.onnx.part*
```

- [ ] **Step 5: Commit new chunks**

```bash
git add static/model-v2/
git commit -m "feat: add Chinese GPT-2 ONNX chunks, remove English gpt2 chunks"
```

---

## Task 4: Update `src/utils/data.ts`

**Files:**
- Modify: `src/utils/data.ts`

- [ ] **Step 1: Fix hard-coded `modelMetaMap.gpt2` references**

In `src/utils/data.ts` at the `attentionTensors` definition (~line 337), replace:

```typescript
const attentionTensors = Array(modelMetaMap.gpt2.layer_num)
	.fill(0)
	.flatMap((_, i) => {
		return Array(modelMetaMap.gpt2.attention_head_num)
```

With:

```typescript
const attentionTensors = Array(modelMetaMap['gpt2-chinese'].layer_num)
	.fill(0)
	.flatMap((_, i) => {
		return Array(modelMetaMap['gpt2-chinese'].attention_head_num)
```

- [ ] **Step 2: Add `add_special_tokens: false` to tokenizer encode**

> Note on API: @xenova/transformers v2.x `encode` signature is `encode(text, text_pair, options)` mirroring the Python Transformers API. `null` is passed as `text_pair` since we are encoding a single string. If type-check fails on this call, verify the installed typings — the alternative form is `tokenizer(input, { add_special_tokens: false }).input_ids`.

In `src/utils/data.ts` at the `getTokenization` function (~line 116), replace:

```typescript
export const getTokenization = async (tokenizer: PreTrainedTokenizer, input: string) => {
	const token_ids = tokenizer.encode(input);
	const input_tokens = token_ids.map((id) => tokenizer.decode([id])).flat();
```

With:

```typescript
export const getTokenization = async (tokenizer: PreTrainedTokenizer, input: string) => {
	const token_ids = tokenizer.encode(input, null, { add_special_tokens: false });
	const input_tokens = token_ids.map((id) => tokenizer.decode([id])).flat();
```

- [ ] **Step 3: Run type check**

```bash
npm run check
```

Expected: no errors. If the `encode` call produces a type error, replace it with:
```typescript
const token_ids = (await tokenizer(input, { add_special_tokens: false })).input_ids.tolist()[0] as number[];
```

- [ ] **Step 4: Commit**

```bash
git add src/utils/data.ts
git commit -m "fix: update data.ts for gpt2-chinese model meta and BertTokenizer special tokens"
```

---

## Task 5: Update `src/store/index.ts`

**Files:**
- Modify: `src/store/index.ts`

- [ ] **Step 1: Remove the `ex0` import and update initial store state**

Replace:

```typescript
import { ex0 } from '~/constants/examples';
```

With nothing (delete the line).

Then replace the three initial writable calls:

```typescript
export const modelData = writable<ModelData>(ex0);
export const predictedToken = writable<Probability>();
export const tokens = writable<string[]>(ex0?.tokens);
export const tokenIds = writable<number[]>(ex0?.tokenIds);
```

With:

```typescript
export const modelData = writable<ModelData | null>(null);
export const predictedToken = writable<Probability>();
export const tokens = writable<string[]>([]);
export const tokenIds = writable<number[]>([]);
```

- [ ] **Step 2: Replace `modelMetaMap` entries**

Replace:

```typescript
export const modelMetaMap: Record<string, ModelMetaData> = {
	gpt2: { layer_num: 12, attention_head_num: 12, dimension: 768, chunkTotal: 63 },
	'gpt2-medium': { layer_num: 24, attention_head_num: 16, dimension: 1024 },
	'gpt2-large': { layer_num: 36, attention_head_num: 20, dimension: 1280 }
};
```

With (replace `<N>` with the chunk count from Task 3 Step 3):

```typescript
export const modelMetaMap: Record<string, ModelMetaData> = {
	'gpt2-chinese': { layer_num: 12, attention_head_num: 12, dimension: 768, chunkTotal: <N> }
};
```

- [ ] **Step 3: Update `initialSelectedModel` and `inputTextExample`**

Replace:

```typescript
export const inputTextExample = [
	'Data visualization empowers users to',
	'Artificial Intelligence is transforming the',
	'As the spaceship was approaching the',
	'On the deserted planet they discovered a',
	'IEEE VIS conference highlights the'
];
```

With:

```typescript
export const inputTextExample = [
	'床前明月光，疑是',
	'春眠不觉晓，处处',
	'世上本没有路，走的人多了，也便',
	'横眉冷对千夫指，俯首甘为',
	'臣妾很想知足，可臣妾',
	'黑夜给了我黑色的眼睛，我却用它',
];
```

Replace:

```typescript
const initialSelectedModel = 'gpt2';
```

With:

```typescript
const initialSelectedModel = 'gpt2-chinese';
```

- [ ] **Step 4: Run type check**

```bash
npm run check
```

Expected: no errors.

- [ ] **Step 5: Commit**

```bash
git add src/store/index.ts
git commit -m "feat: update store for Chinese GPT-2 — model meta, prompts, remove cached English data"
```

---

## Task 6: Update `src/routes/+page.svelte`

**Files:**
- Modify: `src/routes/+page.svelte`

- [ ] **Step 1: Update the tokenizer**

Replace:

```typescript
const gpt2Tokenizer = await AutoTokenizer.from_pretrained('Xenova/gpt2');
```

With:

```typescript
const gpt2Tokenizer = await AutoTokenizer.from_pretrained('uer/gpt2-chinese-cluecorpussmall');
```

- [ ] **Step 2: Remove stale cached data imports**

Remove this import line:

```typescript
import { ex0, ex1, ex2, ex3, ex4 } from '~/constants/examples';
```

Change this import from `~/utils/data`:

```typescript
import { adjustTemperature, runModel, fakeRunWithCachedData } from '~/utils/data';
```

To:

```typescript
import { adjustTemperature, runModel } from '~/utils/data';
```

- [ ] **Step 3: Remove `cachedDataMap` and the fake-run fallback**

Replace:

```typescript
// Subscribe inputs
const cachedDataMap = [ex0, ex1, ex2, ex3, ex4];
const subscribeInputs = (tokenizer: PreTrainedTokenizer) => {
    const runModelOrCache = () => {
        if ($isFetchingModel || !$modelSession) {
            const cachedData = cachedDataMap[$selectedExampleIdx];

            fakeRunWithCachedData({
                cachedData,
                tokenizer,
                temperature: $temperature,
                sampling: $sampling
            });
            return;
        }
        // run model when input has changed
        runModel({
            tokenizer,
            input: $inputText.trim(),
            temperature: $temperature,
            sampling: $sampling
        });
    };
```

With:

```typescript
// Subscribe inputs
const subscribeInputs = (tokenizer: PreTrainedTokenizer) => {
    const runModelOrCache = () => {
        if ($isFetchingModel || !$modelSession) {
            return;
        }
        runModel({
            tokenizer,
            input: $inputText.trim(),
            temperature: $temperature,
            sampling: $sampling
        });
    };
```

- [ ] **Step 4: Add null guards to temperature and sampling subscribers**

`$modelData` is now initialized to `null` and only set after inference runs. The temperature and sampling subscribers must not call `adjustTemperature` before that.

Find the temperature subscriber (immediately after `subscribeInputs`):

```typescript
let initialTemperature = true; // prevent initial redundant rendering
const unsubscribeTemperature = temperature.subscribe((value) => {
    if (initialTemperature) {
        initialTemperature = false;
        return;
    }
    adjustTemperature({
        tokenizer,
        logits: $modelData.logits,
        temperature: value,
        sampling: $sampling
    });
});
```

Replace with:

```typescript
let initialTemperature = true; // prevent initial redundant rendering
const unsubscribeTemperature = temperature.subscribe((value) => {
    if (initialTemperature) {
        initialTemperature = false;
        return;
    }
    if (!$modelData) return;
    adjustTemperature({
        tokenizer,
        logits: $modelData.logits,
        temperature: value,
        sampling: $sampling
    });
});
```

Find the sampling subscriber:

```typescript
let initialSampling = true; // prevent initial redundant rendering
const unsubscribeSmapling = sampling.subscribe((value) => {
    if (initialSampling) {
        initialSampling = false;
        return;
    }
    adjustTemperature({
        tokenizer,
        logits: $modelData.logits,
        temperature: $temperature,
        sampling: value
    });
});
```

Replace with:

```typescript
let initialSampling = true; // prevent initial redundant rendering
const unsubscribeSmapling = sampling.subscribe((value) => {
    if (initialSampling) {
        initialSampling = false;
        return;
    }
    if (!$modelData) return;
    adjustTemperature({
        tokenizer,
        logits: $modelData.logits,
        temperature: $temperature,
        sampling: value
    });
});
```

- [ ] **Step 5: Update chunk filename and count**

Replace (using the chunk count `<N>` from Task 3 Step 3):

```typescript
const chunkNum = 63; //TODO: move to model meta
const chunkUrls = Array(chunkNum)
    .fill(0)
    .map((d, i) => `${base}/model-v2/gpt2.onnx.part${i}`);
```

With:

```typescript
const chunkNum = <N>;
const chunkUrls = Array(chunkNum)
    .fill(0)
    .map((d, i) => `${base}/model-v2/gpt2-chinese.onnx.part${i}`);
```

- [ ] **Step 6: Run type check**

```bash
npm run check
```

Expected: no errors.

- [ ] **Step 7: Commit**

```bash
git add src/routes/+page.svelte
git commit -m "feat: update page.svelte for Chinese tokenizer, chunk URLs, null guards, remove cache fallback"
```

---

## Task 7: Update UI copy strings

**Files:**
- Modify: `src/utils/textbookPages.ts`
- Modify: `src/components/InputForm.svelte`
- Modify: `src/components/article/Article.svelte`
- Modify: `src/components/Popovers/LogitWeightPopover.svelte`

### `textbookPages.ts`

- [ ] **Step 1: Line 39 — update model description**

Replace:

```
Here we use GPT-2 (small), simpler than newer ones but perfect for learning the fundamentals.
```

With:

```
Here we use GPT-2 Chinese (small), simpler than newer ones but perfect for learning the fundamentals.
```

- [ ] **Step 2: Line 128 — update vocab size**

Replace:

```
GPT-2 (small) has 50,257 token vocabulary, each with a unique ID.
```

With:

```
GPT-2 Chinese has 21,128 token vocabulary, each with a unique ID.
```

- [ ] **Step 3: Line 190 — update model name**

Replace:

```
GPT-2 (small) has 12 of them.
```

With:

```
GPT-2 Chinese has 12 of them.
```

- [ ] **Step 4: Line 288 — update head count label**

Replace:

```
the model splits them into several <strong>heads</strong> (12 in GPT-2 small).
```

With:

```
the model splits them into several <strong>heads</strong> (12 in GPT-2 Chinese).
```

- [ ] **Step 5: Line 336 — update head count label**

Replace:

```
GPT-2 (small) has 12 such outputs
```

With:

```
GPT-2 Chinese has 12 such outputs
```

- [ ] **Step 6: Line 380 — update logit count**

Replace:

```
This produces <strong>logits</strong>, 50,257 numbers—one for each token in GPT-2's vocabulary—that indicate how likely each token is to come next.
```

With:

```
This produces <strong>logits</strong>, 21,128 numbers—one for each token in the vocabulary—that indicate how likely each token is to come next.
```

### `InputForm.svelte`

- [ ] **Step 7: Update prompt copy (~line 215)**

Replace:

```
Try the examples. Please use a desktop computer to input GPT-2 prompts directly.
```

With:

```
Try the examples. Please use a desktop computer to input prompts directly.
```

Replace:

```
Try the examples while GPT-2 model is being downloaded (600MB)
```

With:

```
Try the examples while the model is being downloaded (~400MB)
```

### `Article.svelte`

- [ ] **Step 8: Update vocab and model name references in Article.svelte**

Make these replacements (exact strings as they appear in the file):

| Replace | With |
|---|---|
| `GPT-2's vocabulary has <code>50,257</code> unique tokens.` | `GPT-2 Chinese's vocabulary has <code>21,128</code> unique tokens.` |
| `GPT-2 (small) represents each token in the vocabulary as a 768-dimensional vector` | `GPT-2 Chinese represents each token in the vocabulary as a 768-dimensional vector` |
| `of shape <code>(50,257, 768)</code>` | `of shape <code>(21,128, 768)</code>` |
| `The GPT-2 (small) model we are examining consists of` | `The GPT-2 Chinese model we are examining consists of` |
| `in GPT-2 (small)'s case, into` | `in GPT-2 Chinese's case, into` |
| `GPT-2 has <code>12</code> self-attention heads` | `GPT-2 Chinese has <code>12</code> self-attention heads` |
| `final representations into a <code>50,257</code>` | `final representations into a <code>21,128</code>` |
| `Transformer Explainer features a live GPT-2 (small) model running directly in the browser.` | `Transformer Explainer features a live GPT-2 Chinese model running directly in the browser.` |

### `LogitWeightPopover.svelte`

- [ ] **Step 9: Update dimension labels in LogitWeightPopover.svelte**

Replace all three occurrences of `50,257` in dimension size labels:

Line 363:
```
<div class="size">({$modelMeta.dimension}, 50,257)</div>
```
→
```
<div class="size">({$modelMeta.dimension}, 21,128)</div>
```

Line 383:
```
<div class="size">(50,257)</div>
```
→
```
<div class="size">(21,128)</div>
```

Line 408:
```
<div class="size">(1, 50,257)</div>
```
→
```
<div class="size">(1, 21,128)</div>
```

- [ ] **Step 10: Run type check**

```bash
npm run check
```

Expected: no errors.

- [ ] **Step 11: Commit**

```bash
git add src/utils/textbookPages.ts src/components/InputForm.svelte src/components/article/Article.svelte src/components/Popovers/LogitWeightPopover.svelte
git commit -m "fix: update all UI copy for Chinese GPT-2 vocab size and model name"
```

---

## Task 8: Smoke test the app

- [ ] **Step 1: Start the dev server**

```bash
npm run dev
```

- [ ] **Step 2: Open the app and verify**

Open `http://localhost:5173` (or the port shown). Check:

1. The example prompts show the 6 Chinese prompts (床前明月光… etc.)
2. The app loads without console errors
3. The model download begins (network tab shows `gpt2-chinese.onnx.part0` etc. being fetched)
4. While the model is loading, changing the temperature/sampling slider does not throw a JS error
5. Once the model loads, entering a Chinese prompt runs inference
6. The predicted tokens shown are Chinese characters (not English)
7. The attention visualization renders correctly

- [ ] **Step 3: Verify tokenization has no special tokens**

In the browser console, run:
```javascript
// tokenizer is available on the window if you expose it, or inspect the first inference run
// confirm the first token ID is NOT 101 ([CLS]) when encoding Chinese text
```

Confirm in the token display that the input `床前明月光` shows individual characters as tokens without a leading `[CLS]` token.
