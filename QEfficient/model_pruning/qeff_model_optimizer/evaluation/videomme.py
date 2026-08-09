"""Video-MME dataset loading, preprocessing, and evaluation helpers."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable
from urllib.parse import parse_qs, urlparse

import torch

from QEfficient.model_pruning.logging_utils import get_logger

logger = get_logger(__name__)

_ANSWER_RE = re.compile(r"(?:^|\b)([A-D])(?:\b|[\).:])", re.IGNORECASE)
_OPTION_LABELS = ("A", "B", "C", "D")
_VIDEO_SUFFIXES = (".mp4", ".mkv", ".webm", ".avi", ".mov")


@dataclass(eq=True)
class VideoMMEExample:
    """Single Video-MME multiple-choice sample."""

    sample_id: str
    video_id: str
    question: str
    options: list[str]
    answer: str
    duration: str = "unknown"
    video_path: str | None = None
    subtitle: str | None = None
    audio_path: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def prompt(self, *, use_subtitles: bool = False) -> str:
        lines = []
        if use_subtitles and self.subtitle:
            lines.append(f"Subtitle:\n{self.subtitle.strip()}")
        lines.append(self.question.strip())
        lines.extend(f"{label}. {option}" for label, option in zip(_OPTION_LABELS, self.options))
        lines.append("Answer with the option letter only.")
        return "\n".join(lines)


@dataclass(eq=True)
class VideoMMESampleResult:
    """Prediction result for one Video-MME sample."""

    sample_id: str
    video_id: str
    duration: str
    answer: str
    prediction: str
    output_text: str
    correct: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "sample_id": self.sample_id,
            "video_id": self.video_id,
            "duration": self.duration,
            "answer": self.answer,
            "prediction": self.prediction,
            "output_text": self.output_text,
            "correct": self.correct,
        }


@dataclass(eq=True)
class VideoMMEReport:
    """Aggregate Video-MME accuracy report."""

    dataset_path: str | None
    video_root: str | None
    use_subtitles: bool
    num_samples: int
    correct: int
    overall_accuracy: float
    per_duration_accuracy: dict[str, float]
    results: list[VideoMMESampleResult]

    def to_dict(self) -> dict[str, Any]:
        return {
            "dataset_path": self.dataset_path,
            "video_root": self.video_root,
            "use_subtitles": self.use_subtitles,
            "num_samples": self.num_samples,
            "correct": self.correct,
            "overall_accuracy": self.overall_accuracy,
            "per_duration_accuracy": dict(self.per_duration_accuracy),
            "with_subtitle_accuracy": self.overall_accuracy if self.use_subtitles else None,
            "without_subtitle_accuracy": None if self.use_subtitles else self.overall_accuracy,
            "results": [result.to_dict() for result in self.results],
        }


def load_videomme_examples(
    *,
    dataset_path: str | None = None,
    video_root: str | None = None,
    split: str = "test",
    num_samples: int | None = None,
    use_subtitles: bool = False,
) -> list[VideoMMEExample]:
    """Load Video-MME examples from a local JSON/JSONL file or Hugging Face."""

    rows = _load_local_rows(dataset_path) if dataset_path else _load_hf_rows(split)
    examples = [_row_to_example(row, video_root=video_root, use_subtitles=use_subtitles) for row in rows]
    examples = [example for example in examples if example.question and example.options and example.answer]
    if num_samples is not None:
        examples = examples[:num_samples]
    if not examples:
        raise ValueError("No valid Video-MME examples were loaded")
    return examples


def evaluate_videomme(
    model,
    processor,
    *,
    dataset_path: str | None = None,
    video_root: str | None = None,
    split: str = "test",
    num_samples: int = 50,
    generation_len: int = 16,
    num_frames: int = 8,
    fps: float | None = None,
    use_subtitles: bool = False,
) -> VideoMMEReport:
    """Evaluate a VLM on Video-MME multiple-choice accuracy."""

    examples = load_videomme_examples(
        dataset_path=dataset_path,
        video_root=video_root,
        split=split,
        num_samples=num_samples,
        use_subtitles=use_subtitles,
    )
    results: list[VideoMMESampleResult] = []
    for example in examples:
        output_text = generate_videomme_answer(
            model,
            processor,
            example,
            generation_len=generation_len,
            num_frames=num_frames,
            fps=fps,
            use_subtitles=use_subtitles,
        )
        prediction = extract_choice(output_text)
        answer = normalize_answer(example.answer)
        results.append(
            VideoMMESampleResult(
                sample_id=example.sample_id,
                video_id=example.video_id,
                duration=example.duration,
                answer=answer,
                prediction=prediction,
                output_text=output_text,
                correct=prediction == answer,
            )
        )
    correct = sum(1 for result in results if result.correct)
    return VideoMMEReport(
        dataset_path=dataset_path,
        video_root=video_root,
        use_subtitles=use_subtitles,
        num_samples=len(results),
        correct=correct,
        overall_accuracy=correct / len(results) if results else 0.0,
        per_duration_accuracy=_per_duration_accuracy(results),
        results=results,
    )


def generate_videomme_answer(
    model,
    processor,
    example: VideoMMEExample,
    *,
    generation_len: int,
    num_frames: int,
    fps: float | None,
    use_subtitles: bool,
) -> str:
    """Build multimodal model inputs for one sample and decode the answer."""

    prompt = example.prompt(use_subtitles=use_subtitles)
    inputs = build_videomme_inputs(
        processor,
        example,
        prompt=prompt,
        num_frames=num_frames,
        fps=fps,
    )
    inputs = _move_to_model_device(inputs, model)
    with torch.no_grad():
        output_ids = model.generate(**inputs, max_new_tokens=generation_len, do_sample=False)
    return decode_generation(processor, output_ids, inputs)


def build_videomme_inputs(
    processor,
    example: VideoMMEExample,
    *,
    prompt: str,
    num_frames: int,
    fps: float | None,
) -> Any:
    """Create processor inputs for Qwen/Gemma-style video or image-text models."""

    messages = _build_messages(example, prompt, num_frames=num_frames, fps=fps)
    qwen_inputs = _build_qwen_vl_inputs(processor, messages)
    if qwen_inputs is not None:
        return qwen_inputs

    text = _apply_chat_template(processor, messages, prompt)
    videos = None
    if example.video_path:
        videos = [_video_input(example.video_path, num_frames=num_frames, fps=fps)]
    kwargs: dict[str, Any] = {
        "text": [text],
        "padding": True,
        "return_tensors": "pt",
    }
    if videos is not None:
        kwargs["videos"] = videos
    try:
        return processor(**kwargs)
    except TypeError:
        kwargs.pop("padding", None)
        return processor(**kwargs)


def _video_input(video_path: str, *, num_frames: int, fps: float | None) -> Any:
    if video_path.startswith(("http://", "https://")):
        return video_path
    if not Path(video_path).exists():
        logger.warning("Video file %s does not exist; passing path through to processor", video_path)
        return video_path
    return sample_video_frames(video_path, num_frames=num_frames, fps=fps)


def sample_video_frames(video_path: str, *, num_frames: int, fps: float | None = None) -> list[Any]:
    """Decode and sample frames from a video path."""

    try:
        import imageio.v3 as iio
    except Exception as exc:
        raise ImportError("Video-MME evaluation requires imageio for video frame decoding") from exc

    path = Path(video_path)
    if not path.exists():
        raise FileNotFoundError(f"Video file not found: {video_path}")

    frames = list(iio.imiter(path))
    if not frames:
        raise ValueError(f"No frames decoded from {video_path}")
    if fps is not None and fps > 0:
        meta = iio.immeta(path)
        source_fps = float(meta.get("fps") or fps)
        stride = max(int(round(source_fps / fps)), 1)
        frames = frames[::stride]
    if len(frames) <= num_frames:
        return frames
    indices = torch.linspace(0, len(frames) - 1, steps=num_frames).round().to(torch.int64).tolist()
    return [frames[idx] for idx in indices]


def extract_choice(text: str) -> str:
    """Extract an A/B/C/D multiple-choice prediction from generated text."""

    match = _ANSWER_RE.search(text.strip())
    return normalize_answer(match.group(1)) if match else ""


def normalize_answer(answer: Any) -> str:
    """Normalize answer labels or indices to A/B/C/D."""

    if isinstance(answer, int):
        return _OPTION_LABELS[answer] if 0 <= answer < len(_OPTION_LABELS) else ""
    value = str(answer).strip()
    if value.isdigit():
        idx = int(value)
        return _OPTION_LABELS[idx] if 0 <= idx < len(_OPTION_LABELS) else ""
    return value[:1].upper() if value[:1].upper() in _OPTION_LABELS else value.upper()


def decode_generation(processor, output_ids: Any, inputs: Any) -> str:
    """Decode generated ids with processor or tokenizer fallbacks."""

    if hasattr(processor, "batch_decode"):
        decoded = processor.batch_decode(output_ids, skip_special_tokens=True)
        return str(decoded[0]).strip()
    tokenizer = getattr(processor, "tokenizer", processor)
    if hasattr(tokenizer, "batch_decode"):
        decoded = tokenizer.batch_decode(output_ids, skip_special_tokens=True)
        return str(decoded[0]).strip()
    if hasattr(tokenizer, "decode"):
        first = output_ids[0] if isinstance(output_ids, (list, tuple)) else output_ids
        return str(tokenizer.decode(first, skip_special_tokens=True)).strip()
    return str(output_ids)


def _load_local_rows(dataset_path: str | None) -> list[dict[str, Any]]:
    path = Path(str(dataset_path))
    if path.is_dir():
        candidates = sorted(path.glob("*.jsonl")) + sorted(path.glob("*.json"))
        if not candidates:
            raise FileNotFoundError(f"No JSON/JSONL files found under {path}")
        path = candidates[0]
    if path.suffix == ".jsonl":
        return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    payload = json.loads(path.read_text())
    if isinstance(payload, dict):
        for key in ("data", "examples", "samples", "questions"):
            if isinstance(payload.get(key), list):
                return list(payload[key])
        return [payload]
    if isinstance(payload, list):
        return payload
    raise ValueError(f"Unsupported Video-MME payload in {path}")


def _load_hf_rows(split: str) -> list[dict[str, Any]]:
    try:
        from datasets import load_dataset
    except Exception as exc:
        raise ImportError("Install datasets or pass --videomme-dataset-path") from exc
    ds = load_dataset("lmms-lab/Video-MME", split=split)
    return [dict(row) for row in ds]


def _row_to_example(row: dict[str, Any], *, video_root: str | None, use_subtitles: bool) -> VideoMMEExample:
    options = _extract_options(row)
    video_path = _resolve_video_path(row, video_root)
    sample_id = str(_first(row, "sample_id", "question_id", "id", "qid", default=row.get("video_id", "")))
    video_id = str(_first(row, "video_id", "videoID", "video", "youtube_id", "url", default=sample_id))
    return VideoMMEExample(
        sample_id=sample_id,
        video_id=video_id,
        question=str(_first(row, "question", "query", "prompt", default="")),
        options=options,
        answer=normalize_answer(_first(row, "answer", "gt_answer", "label", "answer_idx", default="")),
        duration=str(_first(row, "duration", "duration_category", "duration_type", "video_duration_type", default="unknown")).lower(),
        video_path=video_path,
        subtitle=str(_first(row, "subtitle", "subtitles", "caption", default="")) if use_subtitles else None,
        audio_path=_optional_str(_first(row, "audio", "audio_path", default=None)),
        metadata={key: value for key, value in row.items() if key not in {"question", "options", "answer"}},
    )


def _extract_options(row: dict[str, Any]) -> list[str]:
    raw = _first(row, "options", "choices", "candidates", default=None)
    if isinstance(raw, dict):
        return [str(raw[label]) for label in _OPTION_LABELS if label in raw]
    if isinstance(raw, list):
        return [str(option) for option in raw[:4]]
    options = []
    for label in _OPTION_LABELS:
        value = _first(row, label, label.lower(), f"option_{label.lower()}", f"option_{label}", default=None)
        if value is not None:
            options.append(str(value))
    return options


def _resolve_video_path(row: dict[str, Any], video_root: str | None) -> str | None:
    raw = _first(row, "video_path", "path", default=None)
    if raw is None and video_root is not None:
        resolved = _resolve_video_from_root(row, video_root)
        if resolved is not None:
            return resolved
    if raw is None:
        raw = _first(row, "url", "video_url", "videoUrl", "video", "videoID", "video_id", "youtube_id", default=None)
    if raw is None:
        return None
    value = str(raw)
    if value.startswith(("http://", "https://")):
        return value
    path = Path(value)
    if path.is_absolute() or video_root is None:
        return str(path)
    root = Path(video_root)
    candidate = root / path
    if candidate.suffix:
        return str(candidate)
    for suffix in _VIDEO_SUFFIXES:
        resolved = candidate.with_suffix(suffix)
        if resolved.exists():
            return str(resolved)
    return str(candidate)


def _resolve_video_from_root(row: dict[str, Any], video_root: str) -> str | None:
    root = Path(video_root)
    for value in _local_video_candidate_values(row):
        candidate = root / value
        if candidate.suffix and candidate.exists():
            return str(candidate)
        for suffix in _VIDEO_SUFFIXES:
            resolved = candidate.with_suffix(suffix)
            if resolved.exists():
                return str(resolved)
    return None


def _local_video_candidate_values(row: dict[str, Any]) -> list[str]:
    values: list[str] = []
    for key in ("video", "videoID", "video_id", "youtube_id"):
        value = _optional_str(row.get(key))
        if value:
            values.append(value)
    for key in ("url", "video_url", "videoUrl"):
        value = _optional_str(row.get(key))
        if not value:
            continue
        parsed = _video_id_from_url(value)
        if parsed:
            values.append(parsed)
    deduped: list[str] = []
    seen = set()
    for value in values:
        if value.startswith(("http://", "https://")) or value in seen:
            continue
        seen.add(value)
        deduped.append(value)
    return deduped


def _video_id_from_url(value: str) -> str | None:
    parsed = urlparse(value)
    if not parsed.netloc:
        return None
    query = parse_qs(parsed.query)
    if query.get("v"):
        return query["v"][0]
    parts = [part for part in parsed.path.split("/") if part]
    if not parts:
        return None
    if parts[0] in {"shorts", "embed", "v"} and len(parts) > 1:
        return parts[1]
    return parts[-1]


def _build_messages(
    example: VideoMMEExample,
    prompt: str,
    *,
    num_frames: int | None = None,
    fps: float | None = None,
) -> list[dict[str, Any]]:
    content = []
    if example.video_path:
        video_content: dict[str, Any] = {"type": "video", "video": example.video_path}
        if num_frames is not None and num_frames > 0:
            video_content["nframes"] = num_frames
        if fps is not None and fps > 0:
            video_content["fps"] = fps
        content.append(video_content)
    content.append({"type": "text", "text": prompt})
    return [{"role": "user", "content": content}]


def _build_qwen_vl_inputs(processor, messages: list[dict[str, Any]]) -> Any | None:
    if not hasattr(processor, "apply_chat_template"):
        return None
    try:
        from qwen_vl_utils import process_vision_info
    except Exception:
        return None

    try:
        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        image_inputs, video_inputs = process_vision_info([messages])
        return processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )
    except TypeError:
        return None
    except KeyError as exc:
        if exc.args == ("video_fps",):
            logger.warning("qwen_vl_utils did not return video_fps metadata; falling back to local Video-MME sampler")
            return None
        raise


def _apply_chat_template(processor, messages: list[dict[str, Any]], prompt: str) -> str:
    if not hasattr(processor, "apply_chat_template"):
        return prompt
    try:
        return processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    except TypeError:
        return processor.apply_chat_template(messages, add_generation_prompt=True)


def _move_to_model_device(inputs: Any, model) -> Any:
    try:
        device = next(model.parameters()).device
    except Exception:
        return inputs
    if hasattr(inputs, "to"):
        return inputs.to(device)
    if isinstance(inputs, dict):
        return {key: value.to(device) if hasattr(value, "to") else value for key, value in inputs.items()}
    return inputs


def _per_duration_accuracy(results: Iterable[VideoMMESampleResult]) -> dict[str, float]:
    totals: dict[str, int] = {}
    correct: dict[str, int] = {}
    for result in results:
        duration = result.duration or "unknown"
        totals[duration] = totals.get(duration, 0) + 1
        correct[duration] = correct.get(duration, 0) + int(result.correct)
    return {duration: correct.get(duration, 0) / total for duration, total in totals.items()}


def _first(row: dict[str, Any], *keys: str, default: Any = None) -> Any:
    for key in keys:
        if key in row and row[key] not in (None, ""):
            return row[key]
    return default


def _optional_str(value: Any) -> str | None:
    return None if value in (None, "") else str(value)


__all__ = [
    "VideoMMEExample",
    "VideoMMEReport",
    "VideoMMESampleResult",
    "build_videomme_inputs",
    "evaluate_videomme",
    "extract_choice",
    "generate_videomme_answer",
    "load_videomme_examples",
    "normalize_answer",
    "sample_video_frames",
]
