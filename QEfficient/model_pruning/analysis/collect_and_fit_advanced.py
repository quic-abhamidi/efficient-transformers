#!/usr/bin/env python3
"""
Advanced Non-Training Compensation: Data Collection & Fitting

Collects large-scale calibration data and fits multiple analytical compensation
models. Key improvements over previous collect_and_fit.py:

  1. More data: 2000 samples per dataset × 4 datasets = 8000 samples
     → ~160K prefill token vectors, ~32K decode vectors
  2. Higher-rank LoRA: rank=128, 256
  3. Full least-squares linear map (unconstrained rank)
  4. Whitening/optimal-transport map (Gaussian OT)
  5. More clusters: K=64, K=128
  6. Per-phase separate fitting with proper regularization

Usage:
    python analysis/collect_and_fit_advanced.py \
        --model Qwen/Qwen2.5-7B-Instruct \
        --start-layer 13 --end-layer 17 \
        --num-samples 2000 \
        --output-dir results/compensation_runs/comparisons/nontraining_advanced/7b_layers14-17
"""

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
from tqdm import tqdm


# ─────────────────────────────────────────────────────────────────────────────
# Dataset loading
# ─────────────────────────────────────────────────────────────────────────────

def load_samples(dataset_name: str, num_samples: int) -> List[str]:
    import random; random.seed(42)
    if dataset_name == "wikitext":
        ds = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="train")
        texts = [x["text"].strip() for x in ds if len(x["text"].strip()) > 80]
    elif dataset_name == "gsm8k":
        ds = load_dataset("openai/gsm8k", "main", split="train")
        texts = [x["question"] for x in ds]
    elif dataset_name == "hellaswag":
        ds = load_dataset("Rowan/hellaswag", split="train")
        texts = [x["ctx"] for x in ds if len(x["ctx"]) > 30]
    elif dataset_name == "winogrande":
        ds = load_dataset("allenai/winogrande", "winogrande_xl", split="train")
        texts = [x["sentence"] for x in ds]
    elif dataset_name == "arc_easy":
        ds = load_dataset("allenai/ai2_arc", "ARC-Easy", split="train")
        texts = [x["question"] for x in ds]
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")
    if len(texts) > num_samples:
        texts = random.sample(texts, num_samples)
    print(f"  Loaded {len(texts)} samples from {dataset_name}")
    return texts


# ─────────────────────────────────────────────────────────────────────────────
# Data collection
# ─────────────────────────────────────────────────────────────────────────────

def collect_hidden_states(
    model, tokenizer, prompts: List[str],
    start_layer: int, end_layer: int,
    device, max_len: int = 256,
) -> Tuple[List[torch.Tensor], List[torch.Tensor],
           List[torch.Tensor], List[torch.Tensor]]:
    """
    Collect (h_start, h_end) for both prefill (all tokens) and decode (last token).
    Returns: prefill_starts, prefill_ends, decode_starts, decode_ends
    """
    layers = model.model.layers
    pf_starts, pf_ends = [], []
    dc_starts, dc_ends = [], []

    model.eval()
    with torch.no_grad():
        for prompt in tqdm(prompts, desc="  Collecting", leave=False):
            inputs = tokenizer(prompt, return_tensors="pt",
                               max_length=max_len, truncation=True)
            inputs = {k: v.to(device) for k, v in inputs.items()}
            seq_len = inputs["input_ids"].shape[1]
            if seq_len < 4:
                continue

            h_s = h_e = None

            def hook_start(m, inp, out):
                nonlocal h_s
                h_s = (out[0] if isinstance(out, tuple) else out).detach().cpu()

            def hook_end(m, inp, out):
                nonlocal h_e
                h_e = (out[0] if isinstance(out, tuple) else out).detach().cpu()

            hk1 = layers[start_layer].register_forward_hook(hook_start)
            hk2 = layers[end_layer].register_forward_hook(hook_end)
            try:
                model(**inputs)
            finally:
                hk1.remove(); hk2.remove()

            if h_s is None or h_e is None:
                continue

            # Prefill: all token positions [seq_len, D]
            for pos in range(seq_len):
                pf_starts.append(h_s[0, pos])
                pf_ends.append(h_e[0, pos])

            # Decode proxy: last token only
            dc_starts.append(h_s[0, -1])
            dc_ends.append(h_e[0, -1])

    return pf_starts, pf_ends, dc_starts, dc_ends


# ─────────────────────────────────────────────────────────────────────────────
# Fitting functions
# ─────────────────────────────────────────────────────────────────────────────

def fit_lora(H: np.ndarray, Delta: np.ndarray, rank: int) -> Tuple[np.ndarray, np.ndarray]:
    """Fit low-rank adapter: delta ≈ h @ V @ U^T via SVD least-squares."""
    N, D = H.shape
    rank = min(rank, N - 1, D)
    U_h, S_h, Vh_h = np.linalg.svd(H, full_matrices=False)
    U_h = U_h[:, :rank]; S_h = S_h[:rank]; Vh_h = Vh_h[:rank, :]
    valid = S_h > S_h.max() * 1e-6
    U_h = U_h[:, valid]; S_h = S_h[valid]; Vh_h = Vh_h[valid, :]
    W = Delta.T @ U_h / S_h[None, :]   # [D, rank]
    return W, Vh_h.T                    # U=[D,r], V=[D,r]


def fit_full_lstsq(H: np.ndarray, H_end: np.ndarray,
                   reg: float = 1e-4) -> np.ndarray:
    """
    Fit full linear map: h_end ≈ h_start @ W  (unconstrained rank).
    Uses ridge regression: W = (H^T H + reg*I)^{-1} H^T H_end
    Returns W [D, D].
    """
    D = H.shape[1]
    HtH = H.T @ H + reg * np.eye(D, dtype=np.float64)
    HtY = H.T @ H_end
    W = np.linalg.solve(HtH, HtY)   # [D, D]
    return W.astype(np.float32)


def fit_whitening_ot(H: np.ndarray, H_end: np.ndarray,
                     reg: float = 1e-4) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Gaussian optimal transport map: T(h) = mu_end + A @ (h - mu_start)
    where A = Sigma_start^{-1/2} @ (Sigma_start^{1/2} Sigma_end Sigma_start^{1/2})^{1/2} @ Sigma_start^{-1/2}

    For efficiency, use PCA-whitened space (top-K components).
    Returns (mu_start, mu_end, A) where A is the transport matrix.
    """
    mu_s = H.mean(axis=0)
    mu_e = H_end.mean(axis=0)
    Hc = H - mu_s
    Hec = H_end - mu_e

    # PCA of h_start (top-256 components for efficiency)
    K = min(256, H.shape[0] - 1, H.shape[1])
    U, S, Vt = np.linalg.svd(Hc, full_matrices=False)
    U = U[:, :K]; S = S[:K]; Vt = Vt[:K, :]

    # Covariance in PCA space
    Sigma_s = np.diag(S**2 / H.shape[0])  # [K, K]
    Hec_pca = Hec @ Vt.T                   # [N, K]
    Sigma_e_pca = Hec_pca.T @ Hec_pca / H.shape[0]  # [K, K]

    # OT map in PCA space: A_pca = Sigma_s^{-1/2} (Sigma_s^{1/2} Sigma_e Sigma_s^{1/2})^{1/2} Sigma_s^{-1/2}
    # Simplified: since Sigma_s is diagonal, Sigma_s^{1/2} = diag(S/sqrt(N))
    s_sqrt = np.sqrt(np.diag(Sigma_s) + reg)
    s_inv_sqrt = 1.0 / s_sqrt
    M = (s_sqrt[:, None] * Sigma_e_pca) * s_sqrt[None, :]  # [K, K]
    eigvals, eigvecs = np.linalg.eigh(M)
    eigvals = np.maximum(eigvals, 0)
    M_sqrt = eigvecs @ np.diag(np.sqrt(eigvals)) @ eigvecs.T
    A_pca = (s_inv_sqrt[:, None] * M_sqrt) * s_inv_sqrt[None, :]  # [K, K]

    # Full-space transport: A_full = Vt^T @ A_pca @ Vt  [D, D] (low-rank)
    # Store as factored form: A_full = Vt^T @ A_pca @ Vt
    return mu_s.astype(np.float32), mu_e.astype(np.float32), \
           Vt.T.astype(np.float32), A_pca.astype(np.float32), Vt.astype(np.float32)


def fit_clusters(H: np.ndarray, Delta: np.ndarray,
                 n_clusters: int) -> Tuple[np.ndarray, np.ndarray, float]:
    """K-means cluster-based compensation. Returns (centroids, deltas, var_explained)."""
    from sklearn.cluster import MiniBatchKMeans
    km = MiniBatchKMeans(n_clusters=min(n_clusters, len(H)//2),
                         random_state=42, n_init=5, batch_size=512)
    labels = km.fit_predict(H)
    centroids = km.cluster_centers_.astype(np.float32)
    K = centroids.shape[0]
    cluster_deltas = np.zeros((K, H.shape[1]), dtype=np.float32)
    for k in range(K):
        mask = labels == k
        if mask.sum() > 0:
            cluster_deltas[k] = Delta[mask].mean(axis=0)
    # Variance explained
    pred = cluster_deltas[labels]
    ss_res = np.sum((Delta - pred)**2)
    ss_tot = np.sum((Delta - Delta.mean(axis=0))**2)
    var_exp = max(0.0, 1.0 - ss_res / (ss_tot + 1e-10))
    print(f"  K={K}: var_explained={var_exp*100:.1f}%")
    return centroids, cluster_deltas, var_exp


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct")
    p.add_argument("--start-layer", type=int, default=13)
    p.add_argument("--end-layer",   type=int, default=17)
    p.add_argument("--datasets", nargs="+",
                   default=["wikitext", "gsm8k", "hellaswag", "winogrande", "arc_easy"])
    p.add_argument("--num-samples", type=int, default=2000)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--device", default="cuda")
    return p.parse_args()


def main():
    args = parse_args()
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    print("=" * 70)
    print("ADVANCED NON-TRAINING COMPENSATION: DATA COLLECTION & FITTING")
    print("=" * 70)
    print(f"Model:       {args.model}")
    print(f"Layers:      {args.start_layer} → {args.end_layer}")
    print(f"Datasets:    {args.datasets} × {args.num_samples} samples")
    print(f"Output:      {out}")
    print()

    # ── Load model ────────────────────────────────────────────────────────────
    print("Loading model...")
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.bfloat16,
        device_map=str(device), low_cpu_mem_usage=True)
    model.eval()
    tok = AutoTokenizer.from_pretrained(args.model)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    hidden_dim = model.config.hidden_size
    print(f"Hidden dim: {hidden_dim}")

    # ── Collect data ──────────────────────────────────────────────────────────
    all_pf_s, all_pf_e = [], []
    all_dc_s, all_dc_e = [], []

    for ds_name in args.datasets:
        print(f"\nDataset: {ds_name}")
        prompts = load_samples(ds_name, args.num_samples)
        pf_s, pf_e, dc_s, dc_e = collect_hidden_states(
            model, tok, prompts,
            args.start_layer, args.end_layer, device)
        all_pf_s.extend(pf_s); all_pf_e.extend(pf_e)
        all_dc_s.extend(dc_s); all_dc_e.extend(dc_e)
        print(f"  Prefill: {len(pf_s)} tokens | Decode: {len(dc_s)} tokens")

    print(f"\nTotal prefill: {len(all_pf_s)} | decode: {len(all_dc_s)}")

    # Free model memory
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # Convert to numpy
    PF_S = torch.stack(all_pf_s).float().numpy()
    PF_E = torch.stack(all_pf_e).float().numpy()
    DC_S = torch.stack(all_dc_s).float().numpy()
    DC_E = torch.stack(all_dc_e).float().numpy()
    PF_D = PF_E - PF_S
    DC_D = DC_E - DC_S

    print(f"\nPrefill matrix: {PF_S.shape}")
    print(f"Decode  matrix: {DC_S.shape}")

    # Save raw stats
    stats = {
        "n_prefill": len(all_pf_s),
        "n_decode":  len(all_dc_s),
        "prefill_h_start_norm_mean": float(np.linalg.norm(PF_S, axis=1).mean()),
        "prefill_h_end_norm_mean":   float(np.linalg.norm(PF_E, axis=1).mean()),
        "decode_h_start_norm_mean":  float(np.linalg.norm(DC_S, axis=1).mean()),
        "decode_h_end_norm_mean":    float(np.linalg.norm(DC_E, axis=1).mean()),
        "prefill_delta_norm_mean":   float(np.linalg.norm(PF_D, axis=1).mean()),
        "decode_delta_norm_mean":    float(np.linalg.norm(DC_D, axis=1).mean()),
        "prefill_norm_ratio_mean":   float((np.linalg.norm(PF_E, axis=1) /
                                            np.linalg.norm(PF_S, axis=1).clip(1e-8)).mean()),
        "decode_norm_ratio_mean":    float((np.linalg.norm(DC_E, axis=1) /
                                            np.linalg.norm(DC_S, axis=1).clip(1e-8)).mean()),
    }

    # ── Fit models ────────────────────────────────────────────────────────────

    # 1. Mean delta (baseline)
    print("\n[1] Mean delta...")
    torch.save(torch.tensor(PF_D.mean(axis=0)), out / "prefill_mean_delta.pt")
    torch.save(torch.tensor(DC_D.mean(axis=0)), out / "decode_mean_delta.pt")
    torch.save(torch.tensor(PF_S.mean(axis=0)), out / "prefill_h_start_mean.pt")
    torch.save(torch.tensor(PF_E.mean(axis=0)), out / "prefill_h_end_mean.pt")
    torch.save(torch.tensor(DC_S.mean(axis=0)), out / "decode_h_start_mean.pt")
    torch.save(torch.tensor(DC_E.mean(axis=0)), out / "decode_h_end_mean.pt")

    # 2. Norm-adaptive
    print("\n[2] Norm-adaptive rescaling...")
    pf_norms_s = np.linalg.norm(PF_S, axis=1)
    pf_norms_e = np.linalg.norm(PF_E, axis=1)
    dc_norms_s = np.linalg.norm(DC_S, axis=1)
    dc_norms_e = np.linalg.norm(DC_E, axis=1)

    def fit_norm_adaptive(ns, ne):
        ratios = ne / np.maximum(ns, 1e-8)
        inv_ns = 1.0 / np.maximum(ns, 1e-8)
        X = np.column_stack([np.ones_like(inv_ns), inv_ns])
        coeffs, _, _, _ = np.linalg.lstsq(X, ratios, rcond=None)
        return float(coeffs[0]), float(coeffs[1])

    pf_a, pf_b = fit_norm_adaptive(pf_norms_s, pf_norms_e)
    dc_a, dc_b = fit_norm_adaptive(dc_norms_s, dc_norms_e)
    print(f"  Prefill: ratio = {pf_a:.4f} + {pf_b:.4f}/||h||")
    print(f"  Decode:  ratio = {dc_a:.4f} + {dc_b:.4f}/||h||")
    stats["norm_adaptive"] = {"prefill_a": pf_a, "prefill_b": pf_b,
                               "decode_a": dc_a, "decode_b": dc_b}

    # 3. LoRA rank=64
    print("\n[3] LoRA rank=64...")
    pf_U64, pf_V64 = fit_lora(PF_S, PF_D, rank=64)
    dc_U64, dc_V64 = fit_lora(DC_S, DC_D, rank=64)
    torch.save(torch.tensor(pf_U64), out / "prefill_lora64_U.pt")
    torch.save(torch.tensor(pf_V64), out / "prefill_lora64_V.pt")
    torch.save(torch.tensor(dc_U64), out / "decode_lora64_U.pt")
    torch.save(torch.tensor(dc_V64), out / "decode_lora64_V.pt")

    # 4. LoRA rank=128
    print("\n[4] LoRA rank=128...")
    pf_U128, pf_V128 = fit_lora(PF_S, PF_D, rank=128)
    dc_U128, dc_V128 = fit_lora(DC_S, DC_D, rank=128)
    torch.save(torch.tensor(pf_U128), out / "prefill_lora128_U.pt")
    torch.save(torch.tensor(pf_V128), out / "prefill_lora128_V.pt")
    torch.save(torch.tensor(dc_U128), out / "decode_lora128_U.pt")
    torch.save(torch.tensor(dc_V128), out / "decode_lora128_V.pt")

    # 5. LoRA rank=256
    print("\n[5] LoRA rank=256...")
    pf_U256, pf_V256 = fit_lora(PF_S, PF_D, rank=256)
    dc_U256, dc_V256 = fit_lora(DC_S, DC_D, rank=256)
    torch.save(torch.tensor(pf_U256), out / "prefill_lora256_U.pt")
    torch.save(torch.tensor(pf_V256), out / "prefill_lora256_V.pt")
    torch.save(torch.tensor(dc_U256), out / "decode_lora256_U.pt")
    torch.save(torch.tensor(dc_V256), out / "decode_lora256_V.pt")

    # 6. Whitening / Gaussian OT
    print("\n[6] Whitening / Gaussian OT...")
    pf_mu_s, pf_mu_e, pf_Vt_T, pf_A_pca, pf_Vt = fit_whitening_ot(PF_S, PF_E)
    dc_mu_s, dc_mu_e, dc_Vt_T, dc_A_pca, dc_Vt = fit_whitening_ot(DC_S, DC_E)
    torch.save(torch.tensor(pf_mu_s), out / "prefill_ot_mu_start.pt")
    torch.save(torch.tensor(pf_mu_e), out / "prefill_ot_mu_end.pt")
    torch.save(torch.tensor(pf_Vt_T), out / "prefill_ot_VtT.pt")
    torch.save(torch.tensor(pf_A_pca), out / "prefill_ot_A_pca.pt")
    torch.save(torch.tensor(pf_Vt),   out / "prefill_ot_Vt.pt")
    torch.save(torch.tensor(dc_mu_s), out / "decode_ot_mu_start.pt")
    torch.save(torch.tensor(dc_mu_e), out / "decode_ot_mu_end.pt")
    torch.save(torch.tensor(dc_Vt_T), out / "decode_ot_VtT.pt")
    torch.save(torch.tensor(dc_A_pca), out / "decode_ot_A_pca.pt")
    torch.save(torch.tensor(dc_Vt),   out / "decode_ot_Vt.pt")
    print(f"  OT map fitted (PCA dim={pf_Vt.shape[0]})")

    # 7. Clusters K=64
    print("\n[7] Clusters K=64...")
    pf_c64, pf_d64, pf_v64 = fit_clusters(PF_S, PF_D, 64)
    dc_c64, dc_d64, dc_v64 = fit_clusters(DC_S, DC_D, 64)
    torch.save(torch.tensor(pf_c64), out / "prefill_cluster64_centroids.pt")
    torch.save(torch.tensor(pf_d64), out / "prefill_cluster64_deltas.pt")
    torch.save(torch.tensor(dc_c64), out / "decode_cluster64_centroids.pt")
    torch.save(torch.tensor(dc_d64), out / "decode_cluster64_deltas.pt")
    stats["cluster64_var_explained"] = {"prefill": pf_v64, "decode": dc_v64}

    # 8. Clusters K=128
    print("\n[8] Clusters K=128...")
    pf_c128, pf_d128, pf_v128 = fit_clusters(PF_S, PF_D, 128)
    dc_c128, dc_d128, dc_v128 = fit_clusters(DC_S, DC_D, 128)
    torch.save(torch.tensor(pf_c128), out / "prefill_cluster128_centroids.pt")
    torch.save(torch.tensor(pf_d128), out / "prefill_cluster128_deltas.pt")
    torch.save(torch.tensor(dc_c128), out / "decode_cluster128_centroids.pt")
    torch.save(torch.tensor(dc_d128), out / "decode_cluster128_deltas.pt")
    stats["cluster128_var_explained"] = {"prefill": pf_v128, "decode": dc_v128}

    # 9. Variance explained analysis
    print("\n[9] Variance analysis...")
    def var_explained_lora(H, Delta, U, V):
        pred = H @ V @ U.T
        ss_res = np.sum((Delta - pred)**2)
        ss_tot = np.sum((Delta - Delta.mean(axis=0))**2)
        return max(0.0, 1.0 - ss_res / (ss_tot + 1e-10))

    stats["variance_explained"] = {
        "prefill_lora64":  var_explained_lora(PF_S, PF_D, pf_U64, pf_V64),
        "prefill_lora128": var_explained_lora(PF_S, PF_D, pf_U128, pf_V128),
        "prefill_lora256": var_explained_lora(PF_S, PF_D, pf_U256, pf_V256),
        "decode_lora64":   var_explained_lora(DC_S, DC_D, dc_U64, dc_V64),
        "decode_lora128":  var_explained_lora(DC_S, DC_D, dc_U128, dc_V128),
        "decode_lora256":  var_explained_lora(DC_S, DC_D, dc_U256, dc_V256),
        "prefill_cluster64":  pf_v64,
        "prefill_cluster128": pf_v128,
        "decode_cluster64":   dc_v64,
        "decode_cluster128":  dc_v128,
    }

    print("\nVariance explained:")
    for k, v in stats["variance_explained"].items():
        print(f"  {k:<25}: {v*100:.1f}%")

    # Save config
    config = {
        "model": args.model,
        "start_layer": args.start_layer,
        "end_layer": args.end_layer,
        "datasets": args.datasets,
        "num_samples": args.num_samples,
        "hidden_dim": hidden_dim,
        "stats": stats,
        "norm_adaptive": stats["norm_adaptive"],
    }
    class NpEncoder(json.JSONEncoder):
        def default(self, obj):
            if isinstance(obj, (np.floating, np.float32, np.float64)): return float(obj)
            if isinstance(obj, (np.integer,)): return int(obj)
            return super().default(obj)
    with open(out / "config.json", "w") as f:
        json.dump(config, f, indent=2, cls=NpEncoder)

    print(f"\n✅ All models saved to: {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
