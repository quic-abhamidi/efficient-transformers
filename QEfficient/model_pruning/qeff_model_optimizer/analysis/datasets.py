"""Dataset sample loaders shared by the NAS analysis API and legacy CLI.

Each entry in ``SUPPORTED_DATASETS`` maps to a loader that returns a list of
prompt strings. Loaders use the Hugging Face ``datasets`` library and select a
prefix of size ``num_samples`` from a well-known source.

If a dataset fails to load because of network, cache, or schema issues, the
public loader falls back to deterministic built-in prompts so layer contribution
analysis can still run as an infrastructure smoke test. Benchmark/evaluation
quality should still be measured with real datasets.
"""

from __future__ import annotations

from typing import Callable

from QEfficient.model_pruning.logging_utils import get_logger

logger = get_logger(__name__)


class DatasetLoadError(RuntimeError):
    """Raised when a dataset cannot be loaded and no fallback is available."""


def _load_gsm8k(num_samples: int) -> list[str]:
    from datasets import load_dataset

    ds = load_dataset("openai/gsm8k", "main", split="train")
    out: list[str] = []
    for i, item in enumerate(ds):
        if i >= num_samples:
            break
        out.append(item["question"])
    return out


def _load_mbpp(num_samples: int) -> list[str]:
    from datasets import load_dataset

    ds = load_dataset("google-research-datasets/mbpp", "sanitized", split="train")
    out: list[str] = []
    for i, item in enumerate(ds):
        if i >= num_samples:
            break
        prompt = item.get("prompt") or item.get("text") or ""
        if prompt:
            out.append(prompt)
    return out


def _load_wikitext(num_samples: int) -> list[str]:
    from datasets import load_dataset

    ds = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="train")
    out: list[str] = []
    for item in ds:
        text = item["text"].strip()
        if len(text) > 50 and not text.startswith("="):
            out.append(text)
            if len(out) >= num_samples:
                break
    return out


def _load_hellaswag(num_samples: int) -> list[str]:
    from datasets import load_dataset

    ds = load_dataset("Rowan/hellaswag", split="train")
    out: list[str] = []
    for i, item in enumerate(ds):
        if i >= num_samples:
            break
        out.append(f"{item['ctx']} {item['activity_label']}")
    return out


def _load_winogrande(num_samples: int) -> list[str]:
    from datasets import load_dataset

    ds = load_dataset("allenai/winogrande", "winogrande_xl", split="train")
    out: list[str] = []
    for i, item in enumerate(ds):
        if i >= num_samples:
            break
        out.append(item["sentence"])
    return out


def _load_arc_challenge(num_samples: int) -> list[str]:
    from datasets import load_dataset

    ds = load_dataset("allenai/ai2_arc", "ARC-Challenge", split="train")
    out: list[str] = []
    for i, item in enumerate(ds):
        if i >= num_samples:
            break
        choices = "  ".join(
            f"{label}) {text}"
            for label, text in zip(item["choices"]["label"], item["choices"]["text"])
        )
        out.append(f"{item['question']}\n{choices}")
    return out


def _load_arc_easy(num_samples: int) -> list[str]:
    from datasets import load_dataset

    ds = load_dataset("allenai/ai2_arc", "ARC-Easy", split="train")
    out: list[str] = []
    for i, item in enumerate(ds):
        if i >= num_samples:
            break
        choices = "  ".join(
            f"{label}) {text}"
            for label, text in zip(item["choices"]["label"], item["choices"]["text"])
        )
        out.append(f"{item['question']}\n{choices}")
    return out


def _load_openbookqa(num_samples: int) -> list[str]:
    from datasets import load_dataset

    ds = load_dataset("allenai/openbookqa", "main", split="train")
    out: list[str] = []
    for i, item in enumerate(ds):
        if i >= num_samples:
            break
        choices = "  ".join(
            f"{label}) {text}"
            for label, text in zip(item["choices"]["label"], item["choices"]["text"])
        )
        out.append(f"{item['question_stem']}\n{choices}")
    return out


def _load_piqa(num_samples: int) -> list[str]:
    from datasets import load_dataset

    ds = load_dataset("baber/piqa", split="train")
    out: list[str] = []
    for i, item in enumerate(ds):
        if i >= num_samples:
            break
        out.append(f"{item['goal']}  A) {item['sol1']}  B) {item['sol2']}")
    return out


def _load_mmlu(num_samples: int) -> list[str]:
    from datasets import load_dataset

    ds = load_dataset("cais/mmlu", "all", split="test")
    out: list[str] = []
    for i, item in enumerate(ds):
        if i >= num_samples:
            break
        letters = ["A", "B", "C", "D"]
        choices = "  ".join(f"{letters[j]}) {c}" for j, c in enumerate(item["choices"]))
        out.append(f"[{item['subject']}] {item['question']}\n{choices}")
    return out


def _load_boolq(num_samples: int) -> list[str]:
    from datasets import load_dataset

    ds = load_dataset("google/boolq", split="train")
    out: list[str] = []
    for i, item in enumerate(ds):
        if i >= num_samples:
            break
        passage_snippet = item["passage"][:300]
        out.append(f"Question: {item['question']}?\nContext: {passage_snippet}")
    return out


def _load_truthfulqa(num_samples: int) -> list[str]:
    from datasets import load_dataset

    ds = load_dataset("truthfulqa/truthful_qa", "multiple_choice", split="validation")
    out: list[str] = []
    for i, item in enumerate(ds):
        if i >= num_samples:
            break
        out.append(item["question"])
    return out


def _load_lambada(num_samples: int) -> list[str]:
    from datasets import load_dataset

    ds = load_dataset("EleutherAI/lambada_openai", split="test")
    out: list[str] = []
    for i, item in enumerate(ds):
        if i >= num_samples:
            break
        out.append(item["text"])
    return out


def _load_mmlu_pro(num_samples: int) -> list[str]:
    from datasets import load_dataset

    ds = load_dataset("TIGER-Lab/MMLU-Pro", split="test")
    out: list[str] = []
    for i, item in enumerate(ds):
        if i >= num_samples:
            break
        letters = [chr(ord("A") + j) for j in range(len(item["options"]))]
        choices = "  ".join(f"{letter}) {opt}" for letter, opt in zip(letters, item["options"]))
        out.append(f"{item['question']}\n{choices}")
    return out


def _load_bbh_causal(num_samples: int) -> list[str]:
    from datasets import load_dataset

    ds = load_dataset("lukaemon/bbh", "causal_judgement", split="test")
    out: list[str] = []
    for i, item in enumerate(ds):
        if i >= num_samples:
            break
        out.append(item["input"])
    return out


def _load_bbh_logical_deduction(num_samples: int) -> list[str]:
    from datasets import load_dataset

    ds = load_dataset("lukaemon/bbh", "logical_deduction_five_objects", split="test")
    out: list[str] = []
    for i, item in enumerate(ds):
        if i >= num_samples:
            break
        out.append(item["input"])
    return out


def _load_ifeval(num_samples: int) -> list[str]:
    from datasets import load_dataset

    ds = load_dataset("HuggingFaceH4/ifeval", split="train")
    out: list[str] = []
    for i, item in enumerate(ds):
        if i >= num_samples:
            break
        out.append(item["prompt"])
    return out


def _load_helpsteer2(num_samples: int) -> list[str]:
    from datasets import load_dataset

    ds = load_dataset("nvidia/HelpSteer2", split="train")
    out: list[str] = []
    for i, item in enumerate(ds):
        if i >= num_samples:
            break
        out.append(item["prompt"])
    return out


def _load_gsm_hard(num_samples: int) -> list[str]:
    from datasets import load_dataset

    ds = load_dataset("reasoning-machines/gsm-hard", split="train")
    out: list[str] = []
    for i, item in enumerate(ds):
        if i >= num_samples:
            break
        out.append(item["input"])
    return out


def _load_orca_math(num_samples: int) -> list[str]:
    from datasets import load_dataset

    ds = load_dataset("microsoft/orca-math-word-problems-200k", split="train")
    out: list[str] = []
    for i, item in enumerate(ds):
        if i >= num_samples:
            break
        out.append(item["question"])
    return out


def _load_humanevalpack(num_samples: int) -> list[str]:
    from datasets import load_dataset

    ds = load_dataset("bigcode/humanevalpack", "python", split="test")
    out: list[str] = []
    for i, item in enumerate(ds):
        if i >= num_samples:
            break
        out.append(item["prompt"])
    return out


def _load_metamathqa(num_samples: int) -> list[str]:
    from datasets import load_dataset

    ds = load_dataset("meta-math/MetaMathQA", split="train")
    out: list[str] = []
    for i, item in enumerate(ds):
        if i >= num_samples:
            break
        out.append(item["query"])
    return out


SUPPORTED_DATASETS: dict[str, Callable[[int], list[str]]] = {
    "gsm8k": _load_gsm8k,
    "mbpp": _load_mbpp,
    "wikitext": _load_wikitext,
    "hellaswag": _load_hellaswag,
    "winogrande": _load_winogrande,
    "arc_challenge": _load_arc_challenge,
    "arc_easy": _load_arc_easy,
    "openbookqa": _load_openbookqa,
    "piqa": _load_piqa,
    "mmlu": _load_mmlu,
    "boolq": _load_boolq,
    "truthfulqa": _load_truthfulqa,
    "lambada": _load_lambada,
    "mmlu_pro": _load_mmlu_pro,
    "bbh_causal": _load_bbh_causal,
    "bbh_logical_deduction": _load_bbh_logical_deduction,
    "ifeval": _load_ifeval,
    "helpsteer2": _load_helpsteer2,
    "gsm_hard": _load_gsm_hard,
    "orca_math": _load_orca_math,
    "humanevalpack": _load_humanevalpack,
    "metamathqa": _load_metamathqa,
}

MODERN_DATASETS = {
    "mmlu_pro", "bbh_causal", "bbh_logical_deduction",
    "ifeval", "helpsteer2", "gsm_hard", "orca_math",
    "humanevalpack", "metamathqa",
}

DEFAULT_ANALYSIS_DATASETS = [
    "mmlu_pro", "bbh_causal", "bbh_logical_deduction",
    "ifeval", "gsm_hard", "humanevalpack", "orca_math",
]

_FALLBACK_PROMPTS: dict[str, list[str]] = {
    "gsm8k": [
        "Janet has 12 apples and gives 5 away. How many apples does she have left?",
        "A train travels 60 miles in 2 hours. What is its average speed?",
    ],
    "hellaswag": [
        "A person opens the refrigerator and takes out a carton of milk. They pour it into a glass.",
        "The athlete walks to the starting line, waits for the signal, and begins running.",
    ],
    "wikitext": [
        "The history of computing includes mechanical calculators, early electronic machines, and modern programmable systems.",
        "Large language models process sequences of tokens and generate text by predicting likely continuations.",
    ],
    "gsm_hard": [
        "A store sells notebooks in packs of 6. If a teacher needs 42 notebooks, how many packs are required?",
        "If 3 workers finish a task in 8 hours, how long would 6 equally fast workers take?",
    ],
    "mbpp": [
        "Write a function that returns the largest number in a list.",
        "Write a Python function to check whether a string is a palindrome.",
    ],
}

_GENERIC_FALLBACK_PROMPTS = [
    "Explain the main idea of a short article about renewable energy.",
    "Compare two possible solutions to a scheduling problem and choose the better one.",
    "Write a concise answer to a factual question using complete sentences.",
    "Describe the steps needed to solve a simple arithmetic word problem.",
]


def _fallback_samples(dataset_name: str, num_samples: int) -> list[str]:
    base = _FALLBACK_PROMPTS.get(dataset_name, _GENERIC_FALLBACK_PROMPTS)
    return [base[i % len(base)] for i in range(num_samples)]


def load_dataset_samples(dataset_name: str, num_samples: int, *, allow_fallback: bool = True) -> list[str]:
    """Return ``num_samples`` prompt strings drawn from the named dataset.

    ``allow_fallback`` is intended for contribution analysis, where the goal is
    to exercise model layers with representative text. Evaluation/benchmarking
    code should use its own benchmark loader and real datasets.
    """
    if num_samples <= 0:
        raise ValueError("num_samples must be positive")
    loader = SUPPORTED_DATASETS.get(dataset_name)
    if loader is None:
        raise ValueError(
            f"Unsupported dataset {dataset_name!r}. "
            f"Supported: {sorted(SUPPORTED_DATASETS)}"
        )
    try:
        samples = loader(num_samples)
        if not samples:
            raise DatasetLoadError(f"Dataset {dataset_name!r} returned no samples")
        return samples[:num_samples]
    except Exception as exc:
        if allow_fallback:
            logger.warning(
                "Failed to load dataset %r (%s: %s); using %d built-in fallback prompts for analysis.",
                dataset_name,
                type(exc).__name__,
                exc,
                num_samples,
            )
            return _fallback_samples(dataset_name, num_samples)
        raise DatasetLoadError(
            f"Failed to load dataset {dataset_name!r} ({type(exc).__name__}: {exc})"
        ) from exc
