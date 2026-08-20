# Layer Skipping and Linear Residual Patch

Layer skipping removes selected decoder blocks from the exported model to reduce
prefill and decode compute. The feature is useful when a model has enough depth
redundancy that some layers can be bypassed with limited quality loss. The risk
is that the hidden states entering the next surviving layer no longer match the
distribution that layer saw during pretraining.

The linear residual patch is a lightweight compensation mechanism for this
distribution shift. It calibrates a single linear map from a small set of
prompts and injects the resulting correction before the next surviving decoder
layer.

## Motivation

Transformer decoder layers are residual blocks. For a hidden state `h_l`, a
simplified layer update can be written as:

```text
h_{l+1} = h_l + F_l(h_l)
```

where `F_l` includes attention, MLP, normalization, routing, and any
model-specific logic inside the decoder block. If layer `l` is skipped, the
runtime path becomes:

```text
h_{l+1}^{skip} = h_l
```

The skipped model is faster because it avoids evaluating `F_l`, but the next
layers receive a state that is missing the residual update that the original
model expected. For several consecutive skipped layers, the mismatch compounds:

```text
h_j^{full} = L_{j-1}(...L_{s}(h_s)...)
h_j^{skip} = h_s
```

where layers `s ... j-1` are skipped and layer `j` is the next surviving layer.

The accuracy loss usually appears as:

- lower benchmark accuracy
- worse perplexity or next-token likelihood
- changed greedy decoding
- early divergence in generated token sequences
- invalid or repeated special tokens in generation

The important observation is that the skipped model has not necessarily lost all
useful information. The hidden state often remains close enough that a small
calibrated correction can move it back toward the full-model representation.

## Layer Skipping in QEfficient

QEfficient implements layer skipping by replacing selected decoder layers with
an export-safe pass-through wrapper. The wrapper preserves the decoder layer
slot and return structure while removing the skipped layer's compute from the
exported graph.

For a skipped layer, the effective behavior is:

```text
SkippedDecoderLayer(h, *args, **kwargs) -> h
```

For tuple-returning decoder layers, the wrapper returns a tuple whose first
element is the unchanged hidden state and whose optional cache or attention
slots remain structurally compatible with the rest of the model.

This design keeps important QEfficient contracts stable:

- decoder layer indexing remains unchanged
- cache slot layout remains compatible with export and runtime code
- ONNX and QPC tensor naming remain predictable
- the skip decision stays model-local and export-safe

## Why Accuracy Degrades

The next surviving layer was trained to consume `h_j^{full}`, not
`h_j^{skip}`. Even when the two tensors have the same shape and similar norm,
they can differ in directions that matter to attention heads, MLP gates, router
logits, or the language-model head.

Define the representation error at the injection layer:

```text
e_j = h_j^{full} - h_j^{skip}
```

The pruned model continues from:

```text
h_{j+1}^{skip} = L_j(h_j^{skip})
```

while the full model continues from:

```text
h_{j+1}^{full} = L_j(h_j^{full})
```

Because `L_j` is nonlinear, a small error in hidden-state space can produce a
larger error in logits:

```text
logits^{skip} = Head(L_{>j}(h_j^{skip}))
logits^{full} = Head(L_{>j}(h_j^{full}))
```

This is why a skipped model can remain fluent but drift away from the baseline
answer distribution or generate different stop tokens.

## Linear Residual Patch

The linear residual patch learns a correction for the hidden state entering the
next surviving layer. QEfficient applies it as:

```text
\hat h_j = h_j^{skip} + alpha * W h_j^{skip}
```

Then the patched hidden state is passed into the original surviving layer:

```text
h_{j+1}^{patch} = L_j(\hat h_j)
```

In code, this is implemented by wrapping the injection layer:

```text
PatchedDecoderLayer(
    patch=LinearResidualPatch(...),
    original_layer=L_j,
    injection_layer=j,
)
```

The patch is residual by construction. If `W = 0` or `alpha = 0`, the model
falls back to the raw skipped-layer behavior:

```text
\hat h_j = h_j^{skip}
```

This keeps the compensation narrow and easy to reason about.

## Relationship to LinearPatch

The broad LinearPatch idea is to learn a linear map that aligns the pruned
model's hidden state with the full model's hidden state:

```text
h_j^{full} approx B h_j^{skip}
```

or, for calibration matrices:

```text
F approx X B^T
```

QEfficient uses the residual form:

```text
F approx X + X W^T
```

This is equivalent to a constrained full LinearPatch where:

```text
B^T = I + W^T
```

The residual form is useful because the default skipped hidden state remains the
identity path, and the learned matrix only needs to model the missing correction.
This makes the patch easier to dampen with `alpha` and safer to disable without
changing the rest of the layer-skip graph.

## Calibration Data

Calibration uses a small set of prompts. For each prompt, run:

1. the full model, and capture the input hidden state to the injection layer
2. the skipped model, and capture the input hidden state to the same layer

Let:

```text
X = [h_1^{skip}; h_2^{skip}; ...; h_N^{skip}]      in R^{N x d}
F = [h_1^{full}; h_2^{full}; ...; h_N^{full}]      in R^{N x d}
Y = F - X                                          in R^{N x d}
```

where:

- `N` is the number of calibration tokens
- `d` is the hidden size
- `X` is the skipped-model activation matrix
- `F` is the full-model activation matrix
- `Y` is the missing residual correction

Padding tokens should be excluded when possible so the patch is fit on real
language tokens.

## Mathematics

QEfficient's current patch is a bias-free linear residual map:

```text
P(x) = x + alpha * W x
```

For a batch matrix `X`, PyTorch's `nn.Linear(d, d, bias=False)` applies:

```text
linear(X) = X W^T
```

So the calibrated objective is:

```text
min_W ||F - (X + X W^T)||_F^2 + lambda ||W||_F^2
```

Since `Y = F - X`, this becomes ridge regression:

```text
min_W ||Y - X W^T||_F^2 + lambda ||W||_F^2
```

Let:

```text
A = W^T
```

Then solve:

```text
min_A ||Y - X A||_F^2 + lambda ||A||_F^2
```

The closed-form ridge solution is:

```text
A* = (X^T X + lambda I)^{-1} X^T Y
```

The weight stored in the PyTorch module is:

```text
W* = (A*)^T
```

At runtime:

```text
\hat X = X + alpha * X A*
```

or per token:

```text
\hat h_j = h_j^{skip} + alpha * W* h_j^{skip}
```

The `lambda` term stabilizes the solve when calibration tokens are limited or
correlated. The `alpha` term lets users dampen or scale the correction without
re-solving the calibration problem.

## Why This Can Improve Accuracy

Layer skipping removes a nonlinear sequence of blocks, but the downstream model
does not always need an exact reconstruction of every intermediate computation.
It needs a hidden state at the next surviving layer that lies closer to the
full-model representation manifold.

The linear residual patch helps because it:

- directly minimizes the hidden-state error caused by skipping
- preserves the skipped model's original representation through the residual
  connection
- adds only one dense `d x d` projection at the injection point
- avoids retraining the full model
- can be calibrated from a small prompt set

The method is intentionally conservative. It does not introduce new attention,
MLP, routing, or cache behavior. It only transforms the hidden state before a
surviving layer that already exists in the model.

## Candidate Patch Families

The implementation currently available in QEfficient is the full
`LinearResidualPatch`:

```text
Delta_hat(h) = W h
h_corrected = h + alpha * W h
```

The following variants are useful next-step candidates. They should be treated
as design options until implemented and validated.

| Patch family | Formula | Parameters | What it approximates | Deployment notes |
|---|---|---:|---|---|
| Diagonal scale patch | `h + alpha * (s * h)` | `d` | Per-channel activation magnitude mismatch | Cheapest option; no channel mixing. |
| Hadamard + diagonal patch | `H D H h` or residual `h + alpha * (H D H - I)h` | `d` | Structured global mixing plus channel scaling | Fast if Hadamard is compiler-friendly; close to LinearPatch-style magnitude/outlier correction. |
| Low-rank residual patch | `h + alpha * B A h`, rank `r` | `2dr` | Low-dimensional residual subspace | LoRA-like shape, but applied to pruning-boundary activations instead of model weights. |
| Block-diagonal patch | `h + alpha * blockdiag(W_i)h` | about `d^2 / k` for `k` equal blocks | Local channel-group mixing | Middle ground between diagonal and full dense. |
| Orthogonal Procrustes patch | `R h`, `R^T R = I` | `d^2` or structured | Norm-preserving representation alignment | Closed-form SVD; stable, but full matrix can be large. |
| Polar patch | `R P h` | depends on `R`, `P` | Rotation plus stretch/scaling | Separates alignment from importance scaling. |
| PCA residual-subspace patch | `h + alpha * U_k A h` | `kd + kd` | Residuals restricted to top principal directions | Data-driven low-rank output basis. |
| Jacobian patch | `h + alpha * J(h - h_0)` or `h + alpha * J h` | varies | Local linearization of skipped-layer residual function | Full Jacobian is too expensive; diagonal/block/low-rank Jacobian is more realistic. |
| Gaussian-weighted linear patch | `h + alpha * W h`, fit with residual covariance weighting | `d^2` | Residual mean with uncertainty-aware weighting | Use Gaussian statistics for calibration, not stochastic inference. |
| Random-feature patch | `h + alpha * C phi(h)` | depends on features | Nonlinear residual correction | More expressive, but harder to compile and validate. |

### Diagonal Scale Patch

The diagonal scale patch learns one scalar per hidden channel:

```text
Delta_hat(h) = s * h
h_corrected = h + alpha * (s * h)
```

Equivalently:

```text
h_corrected[i] = (1 + alpha * s[i]) h[i]
```

Given skipped activations `X` and full-model activations `F`, a direct
non-residual scale `m` can be fit per channel with ridge stabilization:

```text
m_i = sum_n X_{n,i} F_{n,i} / (sum_n X_{n,i}^2 + lambda)
```

For residual form:

```text
s_i = m_i - 1
```

This is the strongest cheap baseline because it directly targets activation
magnitude mismatch. Its limitation is that it cannot move information between
channels.

### Hadamard + Diagonal Patch

A Hadamard patch adds structured channel mixing around a diagonal scale:

```text
T = H D H
h_corrected = T h
```

or residualized:

```text
h_corrected = h + alpha * (H D H - I) h
```

`H` is orthogonal up to normalization and has entries `+1` and `-1`, which can
spread out token/channel outliers before scaling. This family is especially
interesting because LinearPatch reports a Hadamard transform plus channel-wise
scaling for layer-pruned LLM recovery.

### Low-Rank Residual Patch

A low-rank patch factors the dense residual matrix:

```text
Delta_hat(h) = B A h
A in R^{r x d}
B in R^{d x r}
r << d
```

Runtime:

```text
h_corrected = h + alpha * B A h
```

This is similar in shape to LoRA, but the target is different. LoRA adapts
model weight matrices for a task or domain; this patch compensates hidden-state
mismatch caused by skipped layers. It is useful when the full dense patch works
but is too large for deployment.

### Block-Diagonal Patch

A block-diagonal patch partitions hidden channels into groups and learns one
matrix per group:

```text
W = blockdiag(W_1, W_2, ..., W_k)
h = [h_1, h_2, ..., h_k]
Delta_hat(h) = [W_1 h_1, W_2 h_2, ..., W_k h_k]
```

Runtime:

```text
h_corrected = h + alpha * Delta_hat(h)
```

For hidden size `d` and `k` equal blocks, parameter count drops from `d^2` to
roughly `d^2 / k`. This can align with attention-head groups, MLP channel
groups, or tensor-parallel shards. It provides local channel mixing without the
full cost of a dense matrix.

### Orthogonal and Polar Patches

An orthogonal patch solves:

```text
min_R ||F - X R^T||_F^2
subject to R^T R = I
```

The orthogonal constraint preserves norms and can reduce overfitting. A polar
patch extends this by separating alignment and scaling:

```text
T = R P
R^T R = I
P = P^T, P positive semidefinite
```

Then:

```text
h_corrected = T h
```

or:

```text
h_corrected = h + alpha * (T - I)h
```

This is a good direction for the proposed PolarResidualPatch idea because it
separates representation rotation from channel importance scaling.

### PCA Residual-Subspace Patch

Instead of predicting the residual in the full hidden space, collect residuals:

```text
Y = F - X
```

Find a top-`k` residual basis `U_k`, then predict only coefficients in that
subspace:

```text
Delta_hat(h) = U_k A h
h_corrected = h + alpha * U_k A h
```

This is another low-rank patch, but the output basis is tied directly to the
observed missing residuals.

### Jacobian Patch

The skipped block sequence can be viewed as a function:

```text
G(h) = h_after_skipped_blocks
Delta(h) = G(h) - h
```

A local approximation around calibration point `h_0` is:

```text
Delta(h) approx Delta(h_0) + J(h_0)(h - h_0)
```

A full Jacobian is usually too expensive. Practical variants are:

```text
J = diag(j)
J = blockdiag(J_i)
J = B A
```

This is useful when explaining the patch as a local linear approximation to the
skipped layers rather than as a generic alignment matrix.

### Gaussian and Random-Feature Patches

A Gaussian residual model treats the missing residual statistically:

```text
Delta | h ~ N(mu(h), Sigma)
```

A deterministic deployment form is preferable:

```text
Delta_hat(h) = mu + W h
```

The covariance can weight the calibration objective so high-variance residual
directions do not dominate the solve. Avoid stochastic sampling during
inference unless the generation stack is explicitly designed for nondeterminism.

Random-feature variants use a nonlinear feature map:

```text
phi(h) = cos(G h + b)
Delta_hat(h) = C phi(h)
```

A Hadamard/Fastfood-style `G` can approximate dense Gaussian projections with
structured matrices. This is more expressive than a linear patch, but it is a
second-stage research option because it complicates export, compile, and parity
validation.

## Example Interpretation

For Qwen3-VL-30B-A3B with layers `32 ... 36` skipped:

```text
full model:
    ... -> layer 31 -> layer 32 -> layer 33 -> layer 34 -> layer 35 -> layer 36 -> layer 37 -> ...

skipped model:
    ... -> layer 31 -> pass -> pass -> pass -> pass -> pass -> layer 37 -> ...

patched model:
    ... -> layer 31 -> pass -> pass -> pass -> pass -> pass -> LinearResidualPatch -> layer 37 -> ...
```

The calibration target is the hidden state entering layer `37` in the full
model. The patch learns to move the skipped hidden state toward that target
before layer `37` runs.

## Current Qwen3-VL-30B-A3B Smoke Result

The following result is a single-prompt QAIC/QPC smoke comparison for
Qwen3-VL-30B-A3B with language layers `32 ... 36` skipped. It is useful as
early evidence that the patch improves the observed generation, but it should
not be treated as benchmark accuracy. Use the validation flow below for a
reviewable accuracy claim.

| Variant | Skipped layers | Compensation | TTFT sec | Decode tok/sec | Total tok/sec | E2E sec | Relative result | Sample quality observation |
|---|---:|---|---:|---:|---:|---:|---|---|
| Full baseline | none | none | 0.46 | 64.52 | 50.15 | 1.97 | reference | Clean Qwen introduction and task list. |
| Layer skip | 32-36 | none | 0.41 | 71.40 | 55.62 | 1.78 | +10.66% decode, +10.91% total, -9.64% E2E time vs baseline | Fluent opening, but the sample ends with repeated special/chat tokens. |
| Layer skip + linear patch | 32-36 | LinearResidualPatch | 0.40 | 69.75 | 54.75 | 1.81 | +8.11% decode, +9.17% total, -8.12% E2E time vs baseline | Closer to the baseline style and avoids the repeated special-token failure in this sample. |

Compared with raw layer skipping, the linear patch gives back about 2.31%
decode throughput and 1.56% total throughput in this run, but preserves most of
the speedup while improving the visible generation quality. This is the desired
tradeoff: recover accuracy with a small compensation cost instead of restoring
the skipped decoder blocks.

## End-to-End Commands

The commands below show how to enable and explain the feature in the QEfficient
repository. They assume the repository root is the current working directory and
that the virtual environment is `../nas_env`. Set `HF_HUB_CACHE` to the model
cache location used by your environment before loading or calibrating the model.

```bash
cd /home/abhamidi/new_repo/efficient-transformers
source /home/abhamidi/new_repo/nas_env/bin/activate
export HF_HUB_ENABLE_HF_TRANSFER=1
export HF_HUB_CACHE=/path/to/hf_cache
```

### 1. Create calibration prompts

Use prompts that represent the traffic where layer skipping will be used. Keep
the first run small, then expand once the workflow is verified.

```bash
cat > /home/tmp/qwen3vl_patch_calibration_prompts.txt <<'EOF'
Tell me about yourself.
Explain why the sky appears blue in two sentences.
Give three examples of tasks a vision-language assistant can help with.
Solve step by step: if a box has 12 red balls and 8 blue balls, how many balls are there?
EOF
```

### 2. Calibrate the linear residual patch

This command runs the full model and the skipped model, captures hidden states
at the injection boundary, and solves the ridge-regression patch weights. For
skipped layers `32 ... 36`, the default injection layer is `37`.

```bash
python examples/image_text_to_text/models/qwen3_vl_moe/calibrate_linear_residual_patch.py \
  --model-id Qwen/Qwen3-VL-30B-A3B-Instruct \
  --skip-layers 32 33 34 35 36 \
  --injection-layer 37 \
  --prompts-file /home/tmp/qwen3vl_patch_calibration_prompts.txt \
  --max-length 256 \
  --max-tokens 8192 \
  --ridge-lambda 1e-3 \
  --alpha 1.0 \
  --dtype bfloat16 \
  --device-map auto \
  --trust-remote-code \
  --output /home/tmp/qwen3vl_skip_32_36_linear_patch.pt
```

The script prints calibration MSE before and after correction. A useful first
sanity check is that `mse_after` is lower than `mse_before`.

### 3. Compile and run the full baseline

Run the model without `qaic_config` to establish the quality and performance
reference. This mirrors the text-only language path used by the layer-skip
example and writes QPC artifacts under an explicit compile directory.

```bash
python - <<'PY'
import transformers
from transformers import AutoConfig, AutoProcessor

from QEfficient import QEFFAutoModelForImageTextToText

model_id = "Qwen/Qwen3-VL-30B-A3B-Instruct"
compile_dir = "/home/tmp/qpc_qwen3vl30b_baseline_no_subfunc_no_int8kv"

config = AutoConfig.from_pretrained(model_id, trust_remote_code=True)
qeff_model = QEFFAutoModelForImageTextToText.from_pretrained(
    model_id,
    attn_implementation="eager",
    kv_offload=True,
    config=config,
    trust_remote_code=True,
)

tokenizer = transformers.AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)

qeff_model.compile(
    batch_size=1,
    prefill_seq_len=128,
    ctx_len=4096,
    num_cores=16,
    num_devices=4,
    height=354,
    width=536,
    mxfp6_matmul=True,
    mxint8_kv_cache=False,
    aic_enable_depth_first=True,
    skip_vision=True,
    mos=1,
    use_onnx_subfunctions=False,
    compile_dir=compile_dir,
)

messages = [
    [
        {
            "role": "user",
            "content": [{"type": "text", "text": "Tell me about yourself."}],
        }
    ]
]
inputs = processor.apply_chat_template(
    messages,
    add_generation_prompt=True,
    tokenize=True,
    return_dict=True,
    return_tensors="pt",
)
inputs = qeff_model.model.prepare_inputs_for_generation(
    inputs=inputs,
    prefill_seq_len=128,
    batch_size=1,
)
output = qeff_model.generate(inputs=inputs, generation_len=100)

print("QPC_PATHS=", getattr(qeff_model, "qpc_paths", None))
print("GENERATED_IDS=", output.generated_ids)
print("DECODED=", tokenizer.batch_decode(output.generated_ids))
print(output)
PY
```

### 4. Compile and run layer skipping without compensation

This enables the QEfficient pruning transform and replaces layers `32 ... 36`
with export-safe pass-through layers.

```bash
python examples/image_text_to_text/models/qwen3_vl_moe/run_qwen3vl_layer_skip_linear_patch.py \
  --model-id Qwen/Qwen3-VL-30B-A3B-Instruct \
  --run-name qwen3vl30b_skip_32_36 \
  --skip-layers 32 33 34 35 36 \
  --layer-skip-config /home/tmp/qwen3vl_skip_32_36.json \
  --compile-dir /home/tmp/qpc_qwen3vl30b_skip_32_36_no_subfunc_no_int8kv \
  --generation-len 100 \
  --prefill-seq-len 128 \
  --ctx-len 4096 \
  --num-cores 16 \
  --num-devices 4 \
  --skip-vision
```

The generated QAIC config has this effective shape:

```json
{
  "enable_layer_skipping": true,
  "layer_skip_config": "/home/tmp/qwen3vl_skip_32_36.json"
}
```

The layer-skip JSON contains:

```json
{
  "skip_layers": [32, 33, 34, 35, 36]
}
```

### 5. Compile and run layer skipping with linear patch compensation

This uses the same skipped layers and injects the calibrated residual patch
before layer `37`.

```bash
python examples/image_text_to_text/models/qwen3_vl_moe/run_qwen3vl_layer_skip_linear_patch.py \
  --model-id Qwen/Qwen3-VL-30B-A3B-Instruct \
  --run-name qwen3vl30b_skip_32_36_linear_patch \
  --skip-layers 32 33 34 35 36 \
  --layer-skip-config /home/tmp/qwen3vl_skip_32_36_linear_patch.json \
  --linear-patch-weights /home/tmp/qwen3vl_skip_32_36_linear_patch.pt \
  --injection-layer 37 \
  --alpha 1.0 \
  --compile-dir /home/tmp/qpc_qwen3vl30b_skip_32_36_linear_patch \
  --generation-len 100 \
  --prefill-seq-len 128 \
  --ctx-len 4096 \
  --num-cores 16 \
  --num-devices 4 \
  --skip-vision
```

The effective QAIC config becomes:

```json
{
  "enable_layer_skipping": true,
  "layer_skip_config": "/home/tmp/qwen3vl_skip_32_36_linear_patch.json",
  "layer_skip_compensation": {
    "type": "linear_residual_patch",
    "patch_weights": "/home/tmp/qwen3vl_skip_32_36_linear_patch.pt",
    "injection_layer": 37,
    "alpha": 1.0
  }
}
```

### 6. Compare generated outputs and throughput

After the three runs print their `QPC_PATHS`, compare them with the native QPC
comparison script. Replace the QPC paths below with the paths printed by your
runs.

```bash
python scripts/eval/layer_skip_compensation_eval.py \
  --tokenizer-name Qwen/Qwen3-VL-30B-A3B-Instruct \
  --variant baseline=/home/tmp/qpc_qwen3vl30b_baseline_no_subfunc_no_int8kv/qpc-958e191f37ad58da/qpc \
  --variant skip_32_36=/home/tmp/qpc_qwen3vl30b_skip_32_36_no_subfunc_no_int8kv/qpc-1614ed1d1537ece2/qpc \
  --variant skip_32_36_linear_patch=/home/tmp/qpc_qwen3vl30b_skip_32_36_linear_patch/qpc-e8152a25b4f43f48/qpc \
  --baseline-variant baseline \
  --skip-variant skip_32_36 \
  --prompt "Tell me about yourself." \
  --generation-len 100 \
  --output-json /home/tmp/qwen3vl_layer_skip_eval.json \
  --output-csv /home/tmp/qwen3vl_layer_skip_eval.csv
```

For a prompt set with references, use JSONL:

```bash
cat > /home/tmp/qwen3vl_eval_prompts.jsonl <<'EOF'
{"prompt": "Tell me about yourself.", "reference": "Qwen"}
{"prompt": "Name three things a vision-language assistant can do.", "reference": "answer questions"}
EOF

python scripts/eval/layer_skip_compensation_eval.py \
  --tokenizer-name Qwen/Qwen3-VL-30B-A3B-Instruct \
  --variant baseline=/path/to/baseline/qpc \
  --variant skip_32_36=/path/to/skipped/qpc \
  --variant skip_32_36_linear_patch=/path/to/patched/qpc \
  --baseline-variant baseline \
  --skip-variant skip_32_36 \
  --prompts-file /home/tmp/qwen3vl_eval_prompts.jsonl \
  --generation-len 100 \
  --output-json /home/tmp/qwen3vl_layer_skip_eval_refs.json
```

### 7. Run lm_eval generation tasks

Install `lm_eval` in the active environment if it is not already available.
Do not edit `pyproject.toml` for this local experiment.

```bash
pip install lm-eval

python scripts/eval/lm_eval_qpc_accuracy.py \
  --tokenizer-name Qwen/Qwen3-VL-30B-A3B-Instruct \
  --variant baseline=/path/to/baseline/qpc \
  --variant skip_32_36=/path/to/skipped/qpc \
  --variant skip_32_36_linear_patch=/path/to/patched/qpc \
  --baseline-variant baseline \
  --skip-variant skip_32_36 \
  --tasks gsm8k \
  --limit 50 \
  --generation-len 256 \
  --output-json /home/tmp/qwen3vl_lm_eval.json
```

The lm_eval bridge currently supports generation-style tasks through
`generate_until`. It intentionally raises for loglikelihood-only tasks until a
logits-backed QPC adapter is added.

## Validation Strategy

Accuracy compensation should be validated quantitatively. A single generated
sample is useful for debugging, but it is not enough evidence.

Recommended validation order:

1. Compare calibration MSE before and after the patch.
2. Compare generated-token similarity to the full baseline.
3. Compare reference metrics on a prompt set with expected answers.
4. Run benchmark-style evaluation through `lm_eval` for generation tasks.
5. Measure throughput and TTFT to confirm the speed benefit remains.

QEfficient includes helper scripts for this workflow:

```bash
python scripts/eval/layer_skip_compensation_eval.py \
  --tokenizer-name Qwen/Qwen3-VL-30B-A3B-Instruct \
  --variant baseline=/path/to/baseline/qpc \
  --variant skip_32_36=/path/to/skipped/qpc \
  --variant skip_32_36_linear_patch=/path/to/patched/qpc \
  --baseline-variant baseline \
  --skip-variant skip_32_36 \
  --prompts-file prompts.jsonl \
  --generation-len 128 \
  --output-json /home/tmp/layer_skip_eval.json
```

For lm-evaluation-harness generation tasks:

```bash
python scripts/eval/lm_eval_qpc_accuracy.py \
  --tokenizer-name Qwen/Qwen3-VL-30B-A3B-Instruct \
  --variant baseline=/path/to/baseline/qpc \
  --variant skip_32_36=/path/to/skipped/qpc \
  --variant skip_32_36_linear_patch=/path/to/patched/qpc \
  --baseline-variant baseline \
  --skip-variant skip_32_36 \
  --tasks gsm8k \
  --limit 50 \
  --generation-len 256 \
  --output-json /home/tmp/lm_eval_layer_skip.json
```

The lm_eval bridge currently supports generation-style tasks through
`generate_until`. Loglikelihood-only tasks need a logits-backed QPC adapter
before they can be scored correctly.

## Related Work and Patent Search Notes

This section is a technical prior-art map as of 2026-08-20. It is not a
patentability, novelty, validity, or freedom-to-operate opinion. Treat it as
input for patent counsel and for a broader patent/non-patent literature search.

| Area | Resource | Similarity to this pitch | Key differences and claim-risk notes |
|---|---|---|---|
| Direct layer-pruning recovery | [LinearPatch: A Simple Linear Patch Revives Layer-Pruned Large Language Models](https://papers.neurips.cc/paper_files/paper/2025/hash/e323ebf7c91a4ca061612ba4a2f2164d-Abstract-Conference.html), NeurIPS 2025; [arXiv](https://arxiv.org/abs/2505.24680) | Very high. It targets layer-pruned LLM degradation by adding a lightweight linear patch at the pruning interface. | This is the closest non-patent literature. QEfficient's current residual form should not be pitched as broadly novel over linear activation patching by itself. Differentiation must come from specific residual objective, calibration target, QAIC export/runtime constraints, injection semantics, or future polar/orthogonal residual transport details. |
| Direct layer-pruning recovery | [Ghosted Layers: Unconstrained Activation Alignment for Recovering Layer-Pruned LLMs](https://arxiv.org/abs/2605.15491), 2026; [ICML page](https://icml.cc/virtual/2026/75231) | Very high. It frames layer pruning as boundary activation alignment and uses a closed-form linear operator from calibration activations. | High prior-art risk for any broad claim covering closed-form linear activation alignment at pruning boundaries. A patent story should emphasize what is not covered: residual-patch deployment on QEfficient/QAIC, cache-preserving skipped-layer wrappers, compilation-safe insertion, or a distinct polar/orthogonal residual-transport formulation if implemented. |
| Direct depth-pruning replacement | [ReplaceMe: Network Simplification via Depth Pruning and Transformer Block Linearization](https://arxiv.org/abs/2505.02819), NeurIPS 2025 | High. It is training-free depth pruning with calibration-estimated linear transformations replacing transformer blocks. | ReplaceMe approximates pruned blocks with a linear transformation and may merge the map into remaining blocks. QEfficient's current patch is a residual correction injected before a surviving layer and must preserve QPC/export interfaces. Broad claims about replacing skipped blocks with calibration-fitted linear maps are risky. |
| Layer pruning plus replacement network | [LLM-Streamline: Streamlining Redundant Layers to Compress Large Language Models](https://proceedings.iclr.cc/paper_files/paper/2025/hash/4b00a351b41358965613c118e87dc28c-Abstract-Conference.html), ICLR 2025; [OpenReview](https://openreview.net/forum?id=IC5RJvRoMp) | Medium. It prunes layers and trains a lightweight replacement module to reduce loss. | This is less direct because it uses a trained replacement network rather than a closed-form residual linear correction. It supports the general motivation that layer pruning needs compensation. |
| Layer cutting and stitching | [GPTailor: Large Language Model Pruning Through Layer Cutting and Stitching](https://proceedings.iclr.cc/paper_files/paper/2026/hash/5c99f2254833533c2a8ca0e0be04d77e-Abstract-Conference.html), ICLR 2026; [arXiv](https://arxiv.org/abs/2506.20480) | Medium. It works in the layer cutting/stitching family and searches over removal, selection, and merging operations. | GPTailor is broader model tailoring and stitching, not the same as a single calibrated residual patch at a pruning boundary. It is still relevant for claims around layer removal plus downstream recovery. |
| Latent-space alignment | [Model Stitching by Functional Latent Alignment](https://arxiv.org/abs/2505.20142), 2025 | Medium. Model stitching commonly uses affine transformations to align latent spaces between model parts. | This is conceptual prior art for latent alignment/stitching. The QEfficient idea is narrower: same model, skipped decoder layers, residual correction, QPC-compatible injection. |
| Model stitching background | [Model Stitching: Looking for Functional Similarity Between Representations](https://arxiv.org/abs/2303.11277), 2023 | Medium. It treats stitching layers as a way to test whether representations can be interchanged. | Relevant to the idea that lightweight mappings can connect representation spaces. It does not specifically address layer-pruned LLM recovery on an inference compiler/runtime path. |
| Parameter-efficient adaptation | [LoRA: Low-Rank Adaptation of Large Language Models](https://arxiv.org/abs/2106.09685), ICLR 2022; [OpenReview](https://openreview.net/forum?id=nZeVKeeFYf9) | Medium to low for the current full-rank patch; higher for a future low-rank patch. LoRA freezes base weights and injects trainable low-rank update matrices. | Similarity: residual/additive low-rank linear update with frozen base weights. Difference: LoRA adapts task/domain weights during fine-tuning, while this patch compensates skipped-layer hidden-state mismatch after pruning. Low-rank PRP should be carefully distinguished from LoRA-style adapters. |
| Patent landscape around LoRA | [US20220383126A1: Low-Rank Adaptation of Neural Network Models](https://patents.google.com/patent/US20220383126A1/en) | Medium to low. It covers adding trainable low-rank factorization matrices to model weight matrices. | Relevant if claims mention low-rank residual patching, frozen weights, or injected trainable matrices. Less direct for a training-free full-rank hidden-state correction calibrated from full-vs-pruned activations. |
| Adapter composition | [AdapterFusion: Non-Destructive Task Composition for Transfer Learning](https://arxiv.org/abs/2005.00247), EACL 2021 | Low to medium. It uses extra adapter/fusion modules while keeping much of the base model fixed. | Useful background for lightweight inserted modules, but it is task-transfer oriented rather than pruning-boundary activation recovery. |

Critical takeaways for a patent filing discussion:

- Broad claims around "use a linear map to recover layer-pruned LLM accuracy" are likely exposed by LinearPatch, Ghosted Layers, and ReplaceMe.
- Claims around "closed-form linear boundary activation alignment from calibration data" are likely exposed by Ghosted Layers and ReplaceMe.
- Claims around "low-rank additive modules in frozen transformers" must be distinguished from LoRA and related adapter patents.
- Stronger differentiation may come from a narrower system claim: QEfficient-specific layer skipping that preserves decoder-layer slots and cache contracts, plus compiler/export-safe residual patch injection into QAIC/QPC graphs.
- If the proposed polar/orthogonal residual transport is implemented, it should be searched separately against Procrustes alignment, orthogonal adapters, polar decomposition layers, model stitching, and activation transport literature.

Search strings worth repeating in Google Patents, Google Scholar, arXiv, Semantic
Scholar, and OpenReview:

```text
"layer-pruned" "linear patch" "large language model"
"boundary activation alignment" "layer-pruned" transformer
"depth pruning" transformer "linear transformation" calibration
"transformer block linearization" pruning calibration
"model stitching" affine transformation latent alignment transformer
"low-rank adaptation" transformer patent frozen weights injected matrices
"orthogonal" residual alignment transformer pruning
"polar decomposition" activation alignment neural network
```

## Limitations

The linear residual patch is not a proof that the skipped layers are fully
recovered. It is an approximation that should be judged by downstream metrics.

Known limitations:

- a full `d x d` matrix can be large for high hidden-size models
- one global linear map may not capture prompt-dependent or modality-dependent
  effects
- calibration quality depends on prompt coverage
- aggressive layer skipping can remove information that a linear patch cannot
  reconstruct
- benchmark parity still requires PyTorch, ONNXRuntime, and QAIC/QPC validation
  at the relevant boundary

## Future Extensions

The same framework can be extended without changing the layer-skip contract:

- low-rank residual patches
- per-layer or per-skip-group patches
- scalar-gated patches
- orthogonal or polar residual transport
- separate attention-residual and MLP-residual compensation

These should be added only when they improve measured accuracy, latency, memory,
or deployment complexity for real skipped-layer configurations.
