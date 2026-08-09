
import json
import os
import sys
import types
from pathlib import Path

import pandas as pd
import pytest


pytestmark = pytest.mark.model_pruning


def test_dataset_loader_falls_back_when_hf_load_fails(monkeypatch):
    from QEfficient.model_pruning.qeff_model_optimizer.analysis import datasets as ds

    def fail_loader(num_samples):
        raise TypeError("must be called with a dataclass type or instance")

    monkeypatch.setitem(ds.SUPPORTED_DATASETS, "hellaswag", fail_loader)

    prompts = ds.load_dataset_samples("hellaswag", 3)

    assert len(prompts) == 3
    assert all(isinstance(prompt, str) and prompt for prompt in prompts)


def test_public_package_imports():
    import QEfficient.model_pruning.analysis
    import QEfficient.model_pruning.core
    import QEfficient.model_pruning.optimization.layer_skipping
    import QEfficient.model_pruning.qeff_model_optimizer.analysis
    import QEfficient.model_pruning.qeff_model_optimizer.api
    import QEfficient.model_pruning.qeff_model_optimizer.config
    import QEfficient.model_pruning.qeff_model_optimizer.evaluation
    import QEfficient.model_pruning.qeff_model_optimizer.search
    import QEfficient.model_pruning.qeff_model_optimizer.transforms



def test_dataset_downloader_covers_supported_aliases():
    from QEfficient.model_pruning.datasets import download_datasets
    from QEfficient.model_pruning.qeff_model_optimizer.analysis.datasets import SUPPORTED_DATASETS

    assert set(SUPPORTED_DATASETS).issubset(download_datasets.DATASET_SPECS)
    assert "videomme" in download_datasets.DATASET_SPECS
    assert "videomme" in download_datasets.resolve_aliases(["all"])
    assert "videomme" not in download_datasets.resolve_aliases(["text"])
    assert download_datasets.resolve_aliases(["gsm8k", "gsm8k", "videomme"]) == ["gsm8k", "videomme"]

    with pytest.raises(ValueError, match="Unsupported dataset aliases"):
        download_datasets.resolve_aliases(["not_a_dataset"])


def test_dataset_downloader_writes_dataset_artifacts(monkeypatch, tmp_path):
    from QEfficient.model_pruning.datasets import download_datasets

    class FakeDataset:
        column_names = ["question", "answer"]

        def __init__(self, rows):
            self.rows = rows

        def __len__(self):
            return len(self.rows)

        def __iter__(self):
            return iter(self.rows)

        def select(self, indices):
            return FakeDataset([self.rows[index] for index in indices])

        def save_to_disk(self, path):
            target = Path(path)
            target.mkdir(parents=True, exist_ok=True)
            (target / "state.json").write_text(json.dumps({"rows": len(self.rows)}))

    rows = [
        {"question": "q1", "answer": "a1"},
        {"question": "q2", "answer": "a2"},
    ]
    monkeypatch.setattr(
        download_datasets,
        "load_hf_dataset",
        lambda spec, cache_dir: FakeDataset(rows),
    )

    result = download_datasets.download_dataset(
        "gsm8k",
        output_dir=tmp_path,
        cache_dir=None,
        limit=1,
        export_jsonl=True,
        save_to_disk=True,
        force=False,
    )

    alias_dir = tmp_path / "gsm8k"
    manifest = json.loads((alias_dir / "manifest.json").read_text())
    jsonl_lines = (alias_dir / "gsm8k.jsonl").read_text().splitlines()

    assert len(result) == 1
    assert (alias_dir / "dataset" / "state.json").exists()
    assert manifest["alias"] == "gsm8k"
    assert manifest["num_rows"] == 1
    assert manifest["artifacts"]["dataset_dir"] == str(alias_dir / "dataset")
    assert manifest["artifacts"]["jsonl"] == str(alias_dir / "gsm8k.jsonl")
    assert json.loads(jsonl_lines[0]) == rows[0]


def test_dataset_downloader_writes_videomme_url_file(tmp_path):
    from QEfficient.model_pruning.datasets import download_datasets

    videomme_jsonl = tmp_path / "videomme.jsonl"
    url_file = tmp_path / "video_urls.txt"
    videomme_jsonl.write_text(
        "\n".join(
            [
                json.dumps({"video_id": "v1", "url": "https://example.com/v1"}),
                json.dumps({"video_id": "v1", "url": "https://example.com/v1"}),
                json.dumps({"videoID": "v2", "video_url": "https://example.com/v2"}),
                json.dumps({"video_id": "v3"}),
            ]
        )
        + "\n"
    )

    count = download_datasets.write_videomme_url_file(videomme_jsonl, url_file)

    assert count == 2
    assert url_file.read_text().splitlines() == [
        "https://example.com/v1",
        "https://example.com/v2",
    ]


def test_videomme_prefers_downloaded_video_over_remote_url(tmp_path):
    from QEfficient.model_pruning.qeff_model_optimizer.evaluation.videomme import load_videomme_examples

    video_root = tmp_path / "videos"
    video_root.mkdir()
    local_video = video_root / "abc123.mp4"
    local_video.write_bytes(b"placeholder")
    dataset_path = tmp_path / "videomme.jsonl"
    dataset_path.write_text(
        json.dumps({
            "sample_id": "sample-1",
            "video_id": "abc123",
            "url": "https://www.youtube.com/watch?v=abc123",
            "question": "What is shown?",
            "options": ["one", "two", "three", "four"],
            "answer": "A",
        })
        + "\n"
    )

    examples = load_videomme_examples(dataset_path=str(dataset_path), video_root=str(video_root))

    assert examples[0].video_path == str(local_video)


def test_videomme_qwen_message_includes_sampling_controls():
    from QEfficient.model_pruning.qeff_model_optimizer.evaluation import videomme

    example = videomme.VideoMMEExample(
        sample_id="sample-1",
        video_id="abc123",
        question="What is shown?",
        options=["one", "two", "three", "four"],
        answer="A",
        video_path="/tmp/abc123.mp4",
    )

    messages = videomme._build_messages(example, "prompt", num_frames=8, fps=2.0)

    video_content = messages[0]["content"][0]
    assert video_content == {
        "type": "video",
        "video": "/tmp/abc123.mp4",
        "nframes": 8,
        "fps": 2.0,
    }


def test_videomme_falls_back_when_qwen_video_fps_metadata_missing(monkeypatch):
    from QEfficient.model_pruning.qeff_model_optimizer.evaluation import videomme

    class FakeProcessor:
        def apply_chat_template(self, messages, **kwargs):
            return "templated prompt"

        def __call__(self, **kwargs):
            return kwargs

    def fail_process_vision_info(messages):
        raise KeyError("video_fps")

    monkeypatch.setitem(
        sys.modules,
        "qwen_vl_utils",
        types.SimpleNamespace(process_vision_info=fail_process_vision_info),
    )
    example = videomme.VideoMMEExample(
        sample_id="sample-1",
        video_id="abc123",
        question="What is shown?",
        options=["one", "two", "three", "four"],
        answer="A",
        video_path="/tmp/missing-video.mp4",
    )

    inputs = videomme.build_videomme_inputs(
        FakeProcessor(),
        example,
        prompt="prompt",
        num_frames=8,
        fps=None,
    )

    assert inputs["text"] == ["templated prompt"]
    assert inputs["videos"] == ["/tmp/missing-video.mp4"]


def test_weak_layer_analysis_uses_text_tokenizer_from_vlm_processor():
    from QEfficient.model_pruning.qeff_model_optimizer.analysis import weak_layers

    class FakeInputs(dict):
        def to(self, device):
            self["moved_to"] = device
            return self

    class FakeTextTokenizer:
        eos_token = "<eos>"
        pad_token = None

        def __init__(self):
            self.calls = []

        def __call__(self, prompts, **kwargs):
            self.calls.append((prompts, kwargs))
            return FakeInputs({
                "input_ids": [[1, 2, 3]],
                "attention_mask": [[1, 1, 1]],
            })

    class FakeProcessor:
        image_processor = object()

        def __init__(self):
            self.tokenizer = FakeTextTokenizer()

        def __call__(self, *args, **kwargs):
            raise AssertionError("weak-layer text calibration must not call the VLM processor directly")

    class FakeModel:
        def __call__(self, **kwargs):
            assert kwargs["input_ids"] == [[1, 2, 3]]
            assert kwargs["output_hidden_states"] is True
            return types.SimpleNamespace(hidden_states=("embedding", "layer0"))

    processor = FakeProcessor()

    hidden_states, attention_mask = weak_layers._run_forward_pass_batch(
        FakeModel(), processor, ["text prompt"], "cpu", 32
    )

    assert hidden_states == ("embedding", "layer0")
    assert attention_mask == [[1, 1, 1]]
    assert processor.tokenizer.pad_token == "<eos>"
    assert processor.tokenizer.calls[0][0] == ["text prompt"]



class DummyReport:
    def __init__(self):
        self.ranked_layers = [types.SimpleNamespace(layer=1), types.SimpleNamespace(layer=2)]

    def to_dict(self):
        return {
            "model": {"model_id": "dummy"},
            "datasets": ["toy"],
            "ranked_layers": [{"layer": 1}, {"layer": 2}],
        }

    def weakest(self, n):
        return self.ranked_layers[:n]


class DummyPlan:
    def __init__(self, name="baseline", layers=None):
        self.name = name
        self.plan = types.SimpleNamespace(to_dict=lambda: {"transforms": []})
        self.metadata = {"kind": name, "layers": layers or []}

    def to_dict(self):
        return {"metadata": self.metadata, "plan": {"transforms": []}}

    @classmethod
    def from_dict(cls, payload):
        metadata = payload.get("metadata", {})
        return cls(metadata.get("kind", "candidate"), metadata.get("layers", []))


def test_measure_layer_contributions_dataset_mode(monkeypatch, tmp_path):
    from QEfficient.model_pruning.analysis import measure_layer_contributions as mlc

    calls = {}
    monkeypatch.setattr(mlc, "setup_device_and_dtype", lambda device, dtype: ("cpu", "float32"))
    monkeypatch.setattr(mlc, "load_model_and_tokenizer", lambda model, device, dtype: (object(), object()))
    monkeypatch.setattr(mlc, "create_output_directory", lambda model, parent_dir=None: tmp_path / "layer_contributions")

    def fake_single_dataset(model, tokenizer, dataset, metrics, num_samples, device, save_per_sample, model_id, output_dir, verbose, batch_size):
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / "layer_contributions_toy_cosine.csv").write_text("layer_index,avg_delta\n0,0.1\n1,0.2\n")
        calls.update(dataset=dataset, metrics=metrics, num_samples=num_samples, batch_size=batch_size)

    monkeypatch.setattr(mlc, "run_single_dataset_mode", fake_single_dataset)

    output_dir = mlc.generate_layer_analysis(
        model="dummy-model",
        dataset="toy",
        num_samples=2,
        metric="cosine",
        device="cpu",
        dtype="float32",
        batch_size=1,
        output_dir=tmp_path,
    )

    assert output_dir == tmp_path / "layer_contributions"
    assert calls == {"dataset": "toy", "metrics": ["cosine"], "num_samples": 2, "batch_size": 1}
    assert (output_dir / "layer_contributions_toy_cosine.csv").exists()


def test_generate_skip_configurations_from_layer_contributions(tmp_path):
    from QEfficient.model_pruning.optimization.layer_skipping.generate_config import generate_configurations

    data = "layer_index,avg_delta\n0,0.90\n1,0.05\n2,0.10\n3,0.80\n"
    (tmp_path / "layer_contributions_gsm8k_cosine.csv").write_text(data)
    (tmp_path / "layer_contributions_hellaswag_cosine.csv").write_text(data)

    config_data = generate_configurations(
        contribution_dir=str(tmp_path),
        metric="cosine",
        threshold_percentile=50.0,
        max_skip_layers=2,
    )

    assert config_data["metadata"]["datasets_analyzed"] == ["gsm8k", "hellaswag"]
    assert config_data["metadata"]["num_configurations"] == len(config_data["configurations"])
    assert any(config["skip_layers"] for config in config_data["configurations"])


def test_candidate_generation_stamps_metric_metadata():
    from QEfficient.model_pruning.qeff_model_optimizer.analysis.reports import RankedLayer, WeakLayerReport
    from QEfficient.model_pruning.qeff_model_optimizer.config.models import ModelSpec
    from QEfficient.model_pruning.qeff_model_optimizer.search import generate_candidate_plans

    report = WeakLayerReport(
        model_spec=ModelSpec(model_id="dummy"),
        datasets=["toy"],
        ranked_layers=[
            RankedLayer(layer=1, aggregate_score=0.1, rank=1, per_dataset_scores={"toy": 0.1}),
            RankedLayer(layer=2, aggregate_score=0.2, rank=2, per_dataset_scores={"toy": 0.2}),
        ],
        metadata={"metric": "cosine"},
    )

    candidates = generate_candidate_plans(report, max_skip_layers=1, top_k=1, include_baseline=True)

    assert candidates[0].metadata["metric"] == "cosine"
    assert candidates[1].metadata["metric"] == "cosine"


def test_nas_pipeline_analyze_both_writes_metric_families(monkeypatch, tmp_path):
    from QEfficient.model_pruning import nas_pipeline
    from QEfficient.model_pruning.qeff_model_optimizer.analysis.reports import RankedLayer, WeakLayerReport
    from QEfficient.model_pruning.qeff_model_optimizer.config.models import ModelSpec

    def make_report(metric):
        if metric == "cosine":
            scores = [0.1, 0.4, 0.9]
        else:
            scores = [0.8, 0.2, 0.5]
        ranked = sorted(enumerate(scores), key=lambda item: item[1])
        return WeakLayerReport(
            model_spec=ModelSpec(model_id="dummy"),
            datasets=["toy"],
            ranked_layers=[
                RankedLayer(
                    layer=layer,
                    aggregate_score=score,
                    rank=rank,
                    per_dataset_scores={"toy": score},
                )
                for rank, (layer, score) in enumerate(ranked, start=1)
            ],
            metadata={"metric": metric},
        )

    analysis_mod = types.ModuleType("analysis")
    analysis_mod.WeakLayerReport = WeakLayerReport
    analysis_mod.RankedLayer = RankedLayer
    analysis_mod.analyze_weak_layers = lambda *args, **kwargs: make_report(kwargs["metric"])
    config_models_mod = types.ModuleType("config.models")
    config_models_mod.ModelSpec = lambda **kwargs: types.SimpleNamespace(**kwargs)

    monkeypatch.setitem(sys.modules, "QEfficient.model_pruning.qeff_model_optimizer.analysis", analysis_mod)
    monkeypatch.setitem(sys.modules, "QEfficient.model_pruning.qeff_model_optimizer.config.models", config_models_mod)

    nas_pipeline.main([
        "analyze",
        "--model",
        "dummy",
        "--datasets",
        "toy",
        "--num-samples",
        "1",
        "--batch-size",
        "1",
        "--metric",
        "both",
        "--max-skip-layers",
        "1",
        "--top-k",
        "1",
        "--output-dir",
        str(tmp_path),
    ])

    assert (tmp_path / "weak_layer_report_cosine.json").exists()
    assert (tmp_path / "weak_layer_report_l2.json").exists()
    combined = json.loads((tmp_path / "weak_layer_report.json").read_text())
    assert combined["metadata"]["metric"] == "both"
    candidates = json.loads((tmp_path / "candidate_plans.json").read_text())
    metrics = [candidate["metadata"].get("metric") for candidate in candidates]
    assert "both" in metrics
    assert "cosine" in metrics
    assert "l2" in metrics


def test_run_benchmark_uses_lm_eval_adapter(monkeypatch):
    from QEfficient.model_pruning.benchmarking.run_benchmark import run_lm_eval

    captured = {}

    class FakeHFLM:
        def __init__(self, **kwargs):
            captured["hflm"] = kwargs

    def fake_simple_evaluate(**kwargs):
        captured["eval"] = kwargs
        return {"results": {"gsm8k": {"acc,none": 0.5}}}

    lm_eval = types.ModuleType("lm_eval")
    lm_eval.simple_evaluate = fake_simple_evaluate
    hf_mod = types.ModuleType("lm_eval.models.huggingface")
    hf_mod.HFLM = FakeHFLM
    models_mod = types.ModuleType("lm_eval.models")
    monkeypatch.setitem(sys.modules, "lm_eval", lm_eval)
    monkeypatch.setitem(sys.modules, "lm_eval.models", models_mod)
    monkeypatch.setitem(sys.modules, "lm_eval.models.huggingface", hf_mod)

    result = run_lm_eval(object(), object(), tasks=["gsm8k"], batch_size=2, device="cpu", limit=3, max_gen_toks=16)

    assert result["results"]["gsm8k"]["acc,none"] == 0.5
    assert captured["hflm"]["batch_size"] == 2
    assert captured["eval"]["tasks"] == ["gsm8k"]
    assert captured["eval"]["limit"] == 3
    assert captured["eval"]["gen_kwargs"] == {"max_gen_toks": 16, "do_sample": False}




def test_run_benchmark_retries_lm_eval_tasks_individually(monkeypatch):
    from QEfficient.model_pruning.benchmarking.run_benchmark import run_lm_eval

    calls = []

    class FakeHFLM:
        def __init__(self, **kwargs):
            pass

    def fake_simple_evaluate(**kwargs):
        tasks = kwargs["tasks"]
        calls.append(list(tasks))
        if len(tasks) > 1:
            raise TypeError("must be called with a dataclass type or instance")
        if tasks == ["hellaswag"]:
            raise TypeError("must be called with a dataclass type or instance")
        return {"results": {"gsm8k": {"acc,none": 0.5}}}

    lm_eval = types.ModuleType("lm_eval")
    lm_eval.simple_evaluate = fake_simple_evaluate
    hf_mod = types.ModuleType("lm_eval.models.huggingface")
    hf_mod.HFLM = FakeHFLM
    models_mod = types.ModuleType("lm_eval.models")
    monkeypatch.setitem(sys.modules, "lm_eval", lm_eval)
    monkeypatch.setitem(sys.modules, "lm_eval.models", models_mod)
    monkeypatch.setitem(sys.modules, "lm_eval.models.huggingface", hf_mod)

    result = run_lm_eval(object(), object(), tasks=["gsm8k", "hellaswag"], batch_size=1, device="cpu", limit=2)

    assert calls == [["gsm8k", "hellaswag"], ["gsm8k"], ["hellaswag"]]
    assert result["results"]["gsm8k"]["acc,none"] == 0.5
    assert "hellaswag" in result["task_errors"]


def test_skip_layers_preserves_qwen2_tensor_contract():
    from QEfficient.model_pruning.qeff_model_optimizer.config.artifacts import ModelArtifact
    from QEfficient.model_pruning.qeff_model_optimizer.config.models import ModelSpec
    from QEfficient.model_pruning.qeff_model_optimizer.config.transforms import SkipLayersSpec, TransformationPlan
    from QEfficient.model_pruning.qeff_model_optimizer.transforms.skip_layers import SkipLayersTransform
    from QEfficient.model_pruning.qeff_model_optimizer.utils.cleanup import run_model_cleanup

    class FakeLayer:
        def forward(self, hidden_states, *args, **kwargs):
            return f"original:{hidden_states}"

    class FakeModel:
        config = types.SimpleNamespace(model_type="qwen2")

        def __init__(self):
            self.layers = [FakeLayer()]

    model = FakeModel()
    original_forward = model.layers[0].forward
    artifact = ModelArtifact(
        artifact_id="fake",
        model=model,
        tokenizer=object(),
        model_spec=ModelSpec(model_id="fake", dtype="float32", device_map="cpu"),
        plan=TransformationPlan(),
    )

    SkipLayersTransform().apply(artifact, SkipLayersSpec(layers=[0]))

    assert model.layers[0].forward("hidden") == "hidden"
    assert model.layers[0].forward("hidden", attention_mask="mask", use_cache=True) == "hidden"

    run_model_cleanup(model)
    assert model.layers[0].forward("hidden") == "original:hidden"
    assert model.layers[0].forward.__func__ is original_forward.__func__


def test_skip_layers_preserves_gemma3_tuple_contract():
    from QEfficient.model_pruning.qeff_model_optimizer.config.artifacts import ModelArtifact
    from QEfficient.model_pruning.qeff_model_optimizer.config.models import ModelSpec
    from QEfficient.model_pruning.qeff_model_optimizer.config.transforms import SkipLayersSpec, TransformationPlan
    from QEfficient.model_pruning.qeff_model_optimizer.transforms.skip_layers import SkipLayersTransform

    class FakeLayer:
        def forward(self, hidden_states, *args, **kwargs):
            return (f"original:{hidden_states}", None)

    class FakeModel:
        config = types.SimpleNamespace(model_type="gemma3")

        def __init__(self):
            self.layers = [FakeLayer()]

    model = FakeModel()
    artifact = ModelArtifact(
        artifact_id="fake",
        model=model,
        tokenizer=object(),
        model_spec=ModelSpec(model_id="fake", dtype="float32", device_map="cpu"),
        plan=TransformationPlan(),
    )

    SkipLayersTransform().apply(artifact, SkipLayersSpec(layers=[0]))

    assert model.layers[0].forward("hidden") == ("hidden",)
    assert model.layers[0].forward("hidden", output_attentions=True) == ("hidden", None)


def test_compare_benchmark_results_dataframe(tmp_path):
    from QEfficient.model_pruning.benchmarking.generate_report import BenchmarkReportGenerator

    baseline = tmp_path / "baseline.json"
    target = tmp_path / "target.json"
    baseline.write_text(json.dumps({"results": {"gsm8k": {"acc,none": 0.8}}}))
    target.write_text(json.dumps({"results": {"gsm8k": {"acc,none": 0.7}}}))

    report = BenchmarkReportGenerator(
        baseline_results_path=baseline,
        target_results_path=target,
        baseline_name="baseline",
        target_name="skip",
    )
    report.baseline_data = report.load_results_from_file(baseline)
    report.target_data = report.load_results_from_file(target)

    df = report.generate_comparison_dataframe()

    gsm8k = df[df["Dataset"] == "gsm8k"].iloc[0]
    assert gsm8k["Baseline Score"] == pytest.approx(0.8)
    assert gsm8k["Target Score"] == pytest.approx(0.7)
    assert gsm8k["Percentage Change (%)"] == pytest.approx(-12.5)


@pytest.mark.model_pruning_gpu
def test_gpu_pipeline_analyze_stage_writes_artifacts(monkeypatch, tmp_path):
    from QEfficient.model_pruning import nas_pipeline

    analysis_mod = types.ModuleType("analysis")
    analysis_mod.WeakLayerReport = DummyReport
    analysis_mod.analyze_weak_layers = lambda *args, **kwargs: DummyReport()
    config_models_mod = types.ModuleType("config.models")
    config_models_mod.ModelSpec = lambda **kwargs: types.SimpleNamespace(**kwargs)
    search_mod = types.ModuleType("search")
    search_mod.generate_candidate_plans = lambda *args, **kwargs: [DummyPlan("baseline"), DummyPlan("skip", [1])]

    monkeypatch.setitem(sys.modules, "QEfficient.model_pruning.qeff_model_optimizer.analysis", analysis_mod)
    monkeypatch.setitem(sys.modules, "QEfficient.model_pruning.qeff_model_optimizer.config.models", config_models_mod)
    monkeypatch.setitem(sys.modules, "QEfficient.model_pruning.qeff_model_optimizer.search", search_mod)

    nas_pipeline.main([
        "analyze",
        "--model",
        "dummy",
        "--datasets",
        "toy",
        "--num-samples",
        "1",
        "--batch-size",
        "1",
        "--output-dir",
        str(tmp_path),
    ])

    assert (tmp_path / "weak_layer_report.json").exists()
    candidates = json.loads((tmp_path / "candidate_plans.json").read_text())
    assert candidates[1]["metadata"]["layers"] == [1]


@pytest.mark.model_pruning_qaic
def test_qaic_cache_scope_uses_run_local_cache(monkeypatch, tmp_path):
    from QEfficient.model_pruning.qeff_model_optimizer.evaluation.qaic_benchmark import QAICBenchmarkRunner
    import QEfficient.utils.cache as cache_mod
    import QEfficient.utils.constants as constants_mod

    original_cache_home = cache_mod.QEFF_HOME
    original_models_dir = constants_mod.QEFF_MODELS_DIR
    global_cache = tmp_path / "global_qeff_home"
    scoped_cache = tmp_path / "compile" / "baseline" / ".qeff_cache"
    monkeypatch.setenv("QEFF_HOME", str(global_cache))

    with QAICBenchmarkRunner._qeff_cache_scope(scoped_cache):
        assert os.environ["QEFF_HOME"] == str(scoped_cache.resolve())
        assert cache_mod.QEFF_HOME == scoped_cache.resolve()
        assert constants_mod.QEFF_MODELS_DIR == str(scoped_cache.resolve() / "qeff_models")

    assert os.environ["QEFF_HOME"] == str(global_cache)
    assert cache_mod.QEFF_HOME == original_cache_home
    assert constants_mod.QEFF_MODELS_DIR == original_models_dir


@pytest.mark.model_pruning_qaic
def test_qaic_runner_returns_structured_compile_error(monkeypatch, tmp_path):
    from QEfficient.model_pruning.qeff_model_optimizer.config.transforms import TransformationPlan
    from QEfficient.model_pruning.qeff_model_optimizer.evaluation import qaic_benchmark as qb

    class FakeQEffModel:
        def __init__(self):
            self.model = types.SimpleNamespace()

        def compile(self, **kwargs):
            raise RuntimeError("compile boom")

    runner = qb.QAICBenchmarkRunner(model_id="dummy", compile_dir_base=str(tmp_path / "compile"))
    monkeypatch.setattr(
        runner,
        "_load_fresh_qeff_model",
        lambda: (FakeQEffModel(), object(), types.SimpleNamespace(model_id="dummy")),
    )

    result = runner.run("baseline", TransformationPlan(), device_group=[0], batch_size=1)

    assert result.error.startswith("Compile failed: compile boom")
    assert result.plan_name == "baseline"


@pytest.mark.model_pruning_qaic
def test_qaic_runner_returns_structured_generation_error(monkeypatch, tmp_path):
    from QEfficient.model_pruning.qeff_model_optimizer.config.transforms import TransformationPlan
    from QEfficient.model_pruning.qeff_model_optimizer.evaluation import qaic_benchmark as qb

    class FakeQEffModel:
        def __init__(self):
            self.model = types.SimpleNamespace()

        def compile(self, **kwargs):
            return str(tmp_path / "compiled.qpc")

        def generate(self, **kwargs):
            raise RuntimeError("generate boom")

    runner = qb.QAICBenchmarkRunner(model_id="dummy", compile_dir_base=str(tmp_path / "compile"))
    monkeypatch.setattr(
        runner,
        "_load_fresh_qeff_model",
        lambda: (FakeQEffModel(), object(), types.SimpleNamespace(model_id="dummy")),
    )

    result = runner.run("baseline", TransformationPlan(), device_group=[0], batch_size=1)

    assert result.error.startswith("Generation failed: RuntimeError: generate boom")
    assert result.qpc_path == str(tmp_path / "compiled.qpc")


@pytest.mark.model_pruning_qaic
def test_qaic_pipeline_stage_accepts_manual_skip_layers(monkeypatch, tmp_path):
    from QEfficient.model_pruning import nas_pipeline

    transforms_mod = types.ModuleType("config.transforms")
    transforms_mod.TransformationPlan = lambda: types.SimpleNamespace(transforms=[])
    transforms_mod.plan_from_dict = lambda payload: types.SimpleNamespace(transforms=payload.get("transforms", []))

    class FakeRunResult:
        def __init__(self, name):
            self.error = None
            self.qpc_path = f"{tmp_path}/{name}.qpc"
            self.name = name
            self.avg_stats = {"ttft": 2.0, "decode_tps": 4.0, "e2e": 5.0}
            self.accuracy_score = None
            self.accuracy_metric = None
            self.videomme_report = None

        def to_dict(self):
            return {"plan_name": self.name, "avg_stats": self.avg_stats, "error": None}

    class FakeQAICBenchmarkRunner:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

        def run(self, name, plan, device_group, batch_size):
            return FakeRunResult(name)

        def compute_speedups(self, results, baseline_name="baseline"):
            return {"manual_skip_layers_1_3": {"decode_pct": 10.0}}

    evaluation_mod = types.ModuleType("evaluation")
    evaluation_mod.QAICBenchmarkRunner = FakeQAICBenchmarkRunner

    monkeypatch.setitem(sys.modules, "QEfficient.model_pruning.qeff_model_optimizer.config.transforms", transforms_mod)
    monkeypatch.setitem(sys.modules, "QEfficient.model_pruning.qeff_model_optimizer.evaluation", evaluation_mod)

    output_dir = tmp_path / "qaic_manual"
    nas_pipeline.main([
        "qaic",
        "--model",
        "dummy",
        "--skip-layers",
        "3",
        "1",
        "--device-group",
        "0",
        "--output-dir",
        str(output_dir),
    ])

    manual_plan = json.loads((output_dir / "manual_best_plan.json").read_text())
    assert manual_plan["skip_layers"] == [1, 3]
    assert manual_plan["plan_name"] == "manual_skip_layers_1_3"

    comparison = json.loads((output_dir / "benchmark_comparison.json").read_text())
    assert comparison["skip_layers"] == [1, 3]
    assert comparison["optimized_plan_name"] == "manual_skip_layers_1_3"

    all_results = json.loads((output_dir / "all_results.json").read_text())
    assert all_results["plan_file"] is None
    assert all_results["selection_mode"] == "manual_skip_layers_qaic_only"


def test_qaic_pipeline_stage_writes_results(monkeypatch, tmp_path):
    from QEfficient.model_pruning import nas_pipeline

    plan_file = tmp_path / "best_plan.json"
    plan_file.write_text(json.dumps({"plan_name": "skip_1", "plan": {"transforms": []}}))

    transforms_mod = types.ModuleType("config.transforms")
    transforms_mod.TransformationPlan = lambda: types.SimpleNamespace(transforms=[])
    transforms_mod.plan_from_dict = lambda payload: types.SimpleNamespace(transforms=payload.get("transforms", []))

    class FakeRunResult:
        def __init__(self, name):
            self.error = None
            self.qpc_path = f"{tmp_path}/{name}.qpc"
            self.name = name

        def to_dict(self):
            return {"plan_name": self.name, "avg_stats": {"decode_tps": 1.0}, "error": None}

    class FakeQAICBenchmarkRunner:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

        def run(self, name, plan, device_group, batch_size):
            return FakeRunResult(name)

        def compute_speedups(self, results, baseline_name="baseline"):
            return {"skip_1": {"decode_tps_speedup": 1.0}}

    evaluation_mod = types.ModuleType("evaluation")
    evaluation_mod.QAICBenchmarkRunner = FakeQAICBenchmarkRunner

    monkeypatch.setitem(sys.modules, "QEfficient.model_pruning.qeff_model_optimizer.config.transforms", transforms_mod)
    monkeypatch.setitem(sys.modules, "QEfficient.model_pruning.qeff_model_optimizer.evaluation", evaluation_mod)

    output_dir = tmp_path / "qaic"
    nas_pipeline.main([
        "qaic",
        "--model",
        "dummy",
        "--plan",
        str(plan_file),
        "--device-group",
        "0,1",
        "--batch-size",
        "2",
        "--output-dir",
        str(output_dir),
    ])

    all_results = json.loads((output_dir / "all_results.json").read_text())
    assert all_results["device_group"] == [0, 1]
    assert all_results["batch_size"] == 2
    assert all_results["speedups"]["skip_1"]["decode_tps_speedup"] == 1.0



def _write_candidate_plans(path: Path) -> None:
    candidates = [
        {
            "plan": {"transforms": [], "compatibility_mode": "strict", "metadata": {}},
            "priority": 0.0,
            "rationale": "baseline",
            "metadata": {"kind": "baseline"},
        },
        {
            "plan": {
                "transforms": [{"kind": "skip_layers", "layers": [1]}],
                "compatibility_mode": "strict",
                "metadata": {},
            },
            "priority": 0.1,
            "rationale": "skip layer 1",
            "metadata": {"kind": "single", "layers": [1]},
        },
    ]
    path.write_text(json.dumps(candidates))


@pytest.mark.model_pruning_gpu
def test_evaluate_stage_lm_eval_writes_best_plan_and_uses_resolved_tasks(monkeypatch, tmp_path):
    from QEfficient.model_pruning import nas_pipeline
    from QEfficient.model_pruning.qeff_model_optimizer.evaluation import plan_evaluator as pe
    from QEfficient.model_pruning.qeff_model_optimizer.api import loaders as loader_mod

    candidate_file = tmp_path / "candidate_plans.json"
    _write_candidate_plans(candidate_file)
    captured = {"tasks": []}

    class FakeParam:
        device = types.SimpleNamespace(type="cpu", index=None)

    class FakeModel:
        def parameters(self):
            return iter([FakeParam()])

        def generate(self, **kwargs):
            return [[1, 2, 3]]

    class FakeTokenizer:
        def __call__(self, prompt, return_tensors="pt"):
            class Inputs(dict):
                def to(self, device):
                    return self
            return Inputs({"input_ids": [1]})

        def decode(self, tokens, skip_special_tokens=True):
            return "completion"

    class FakeLoader:
        def load(self, spec):
            return FakeModel(), FakeTokenizer()

    def fake_run_lm_eval(model, tokenizer, tasks, **kwargs):
        captured["tasks"].append(list(tasks))
        plan_layers = getattr(model, "_active_skip_layers", [])
        score = 0.80 if not plan_layers else 0.78
        return {"results": {task: {"acc,none": score} for task in tasks}}

    def fake_apply_plan(self, artifact, plan):
        layers = []
        for spec in plan.transforms:
            layers.extend(getattr(spec, "layers", []))
        artifact.model._active_skip_layers = layers
        artifact.plan = plan
        return artifact

    monkeypatch.setattr(loader_mod, "TransformersModelLoader", FakeLoader)
    monkeypatch.setattr(pe, "TransformersModelLoader", FakeLoader)
    monkeypatch.setattr(pe, "run_lm_eval", fake_run_lm_eval)
    monkeypatch.setattr(pe.NASSession, "apply_plan", fake_apply_plan)
    monkeypatch.setattr(pe.NASSession, "close", lambda self: None)

    nas_pipeline.main([
        "evaluate",
        "--model", "dummy",
        "--candidate-plans", str(candidate_file),
        "--datasets", "gsm8k", "hellaswag",
        "--num-samples", "4",
        "--max-candidates", "2",
        "--accuracy-threshold", "5",
        "--eval-method", "lm_eval",
        "--accuracy-metric", "acc",
        "--device-map", "cpu",
        "--output-dir", str(tmp_path),
    ])

    best = json.loads((tmp_path / "best_plan.json").read_text())
    summary = json.loads((tmp_path / "evaluation_summary.json").read_text())

    assert best["plan_name"] == "01_single_1"
    assert best["skip_layers"] == [1]
    assert best["accuracy_score"] == pytest.approx(0.78)
    assert summary["eval_method"] == "lm_eval"
    assert captured["tasks"] == [["gsm8k", "hellaswag"], ["gsm8k", "hellaswag"]]



@pytest.mark.model_pruning_gpu
def test_evaluate_stage_max_candidates_counts_only_non_baseline(monkeypatch, tmp_path):
    from QEfficient.model_pruning import nas_pipeline
    from QEfficient.model_pruning.qeff_model_optimizer.evaluation import plan_evaluator as pe
    from QEfficient.model_pruning.qeff_model_optimizer.api import loaders as loader_mod

    candidates = [
        {
            "plan": {"transforms": [], "compatibility_mode": "strict", "metadata": {}},
            "priority": 0.0,
            "rationale": "baseline",
            "metadata": {"kind": "baseline"},
        },
        {
            "plan": {
                "transforms": [{"kind": "skip_layers", "layers": [1]}],
                "compatibility_mode": "strict",
                "metadata": {},
            },
            "priority": 0.1,
            "rationale": "skip layer 1",
            "metadata": {"kind": "single", "layers": [1]},
        },
        {
            "plan": {
                "transforms": [{"kind": "skip_layers", "layers": [2]}],
                "compatibility_mode": "strict",
                "metadata": {},
            },
            "priority": 0.2,
            "rationale": "skip layer 2",
            "metadata": {"kind": "single", "layers": [2]},
        },
        {
            "plan": {
                "transforms": [{"kind": "skip_layers", "layers": [3]}],
                "compatibility_mode": "strict",
                "metadata": {},
            },
            "priority": 0.3,
            "rationale": "skip layer 3",
            "metadata": {"kind": "single", "layers": [3]},
        },
    ]
    candidate_file = tmp_path / "candidate_plans.json"
    candidate_file.write_text(json.dumps(candidates))

    evaluated_layers = []

    class FakeParam:
        device = types.SimpleNamespace(type="cpu", index=None)

    class FakeModel:
        def parameters(self):
            return iter([FakeParam()])

        def generate(self, **kwargs):
            return [[1, 2, 3]]

    class FakeTokenizer:
        def __call__(self, prompt, return_tensors="pt"):
            class Inputs(dict):
                def to(self, device):
                    return self
            return Inputs({"input_ids": [1]})

        def decode(self, tokens, skip_special_tokens=True):
            return "completion"

    class FakeLoader:
        def load(self, spec):
            return FakeModel(), FakeTokenizer()

    def fake_run_lm_eval(model, tokenizer, tasks, **kwargs):
        plan_layers = tuple(getattr(model, "_active_skip_layers", []))
        evaluated_layers.append(plan_layers)
        score = {(): 0.80, (1,): 0.79, (2,): 0.78}.get(plan_layers, 0.10)
        return {"results": {task: {"acc,none": score} for task in tasks}}

    def fake_apply_plan(self, artifact, plan):
        layers = []
        for spec in plan.transforms:
            layers.extend(getattr(spec, "layers", []))
        artifact.model._active_skip_layers = layers
        artifact.plan = plan
        return artifact

    monkeypatch.setattr(loader_mod, "TransformersModelLoader", FakeLoader)
    monkeypatch.setattr(pe, "TransformersModelLoader", FakeLoader)
    monkeypatch.setattr(pe, "run_lm_eval", fake_run_lm_eval)
    monkeypatch.setattr(pe.NASSession, "apply_plan", fake_apply_plan)
    monkeypatch.setattr(pe.NASSession, "close", lambda self: None)

    nas_pipeline.main([
        "evaluate",
        "--model", "dummy",
        "--candidate-plans", str(candidate_file),
        "--datasets", "gsm8k",
        "--num-samples", "4",
        "--max-candidates", "2",
        "--accuracy-threshold", "5",
        "--eval-method", "lm_eval",
        "--accuracy-metric", "acc",
        "--device-map", "cpu",
        "--output-dir", str(tmp_path),
    ])

    result_names = [
        item["plan_name"]
        for item in json.loads((tmp_path / "plan_results.json").read_text())
    ]
    assert set(result_names) == {"baseline", "01_single_1", "02_single_2"}
    assert (3,) not in evaluated_layers



@pytest.mark.model_pruning_gpu
def test_evaluate_stage_skip_layers_generates_manual_plan_and_comparison(monkeypatch, tmp_path):
    from QEfficient.model_pruning import nas_pipeline
    from QEfficient.model_pruning.qeff_model_optimizer.evaluation import plan_evaluator as pe
    from QEfficient.model_pruning.qeff_model_optimizer.api import loaders as loader_mod

    captured = {"tasks": []}

    class FakeParam:
        device = types.SimpleNamespace(type="cpu", index=None)

    class FakeModel:
        def parameters(self):
            return iter([FakeParam()])

        def generate(self, **kwargs):
            return [[1, 2, 3]]

    class FakeTokenizer:
        def __call__(self, prompt, return_tensors="pt"):
            class Inputs(dict):
                def to(self, device):
                    return self
            return Inputs({"input_ids": [1]})

        def decode(self, tokens, skip_special_tokens=True):
            return "completion"

    class FakeLoader:
        def load(self, spec):
            return FakeModel(), FakeTokenizer()

    def fake_run_lm_eval(model, tokenizer, tasks, **kwargs):
        captured["tasks"].append(list(tasks))
        plan_layers = getattr(model, "_active_skip_layers", [])
        score = 0.80 if not plan_layers else 0.10
        return {"results": {task: {"acc,none": score} for task in tasks}}

    def fake_apply_plan(self, artifact, plan):
        layers = []
        for spec in plan.transforms:
            layers.extend(getattr(spec, "layers", []))
        artifact.model._active_skip_layers = layers
        artifact.plan = plan
        return artifact

    monkeypatch.setattr(loader_mod, "TransformersModelLoader", FakeLoader)
    monkeypatch.setattr(pe, "TransformersModelLoader", FakeLoader)
    monkeypatch.setattr(pe, "run_lm_eval", fake_run_lm_eval)
    monkeypatch.setattr(pe.NASSession, "apply_plan", fake_apply_plan)
    monkeypatch.setattr(pe.NASSession, "close", lambda self: None)

    nas_pipeline.main([
        "evaluate",
        "--model", "dummy",
        "--skip-layers", "5", "1", "5",
        "--datasets", "gsm8k", "hellaswag",
        "--num-samples", "4",
        "--eval-method", "lm_eval",
        "--accuracy-metric", "acc",
        "--device-map", "cpu",
        "--output-dir", str(tmp_path),
    ])

    manual = json.loads((tmp_path / "manual_candidate_plans.json").read_text())
    best = json.loads((tmp_path / "best_plan.json").read_text())
    comparison = json.loads((tmp_path / "comparison_report.json").read_text())
    regression = json.loads((tmp_path / "accuracy_regression_report.json").read_text())

    assert regression == comparison
    assert manual[1]["metadata"]["layers"] == [1, 5]
    assert best["plan_name"] == "01_manual_1_5"
    assert best["skip_layers"] == [1, 5]
    assert comparison["baseline_accuracy_score"] == pytest.approx(0.80)
    assert best["accuracy_threshold"] is None
    assert best["selection_mode"] == "manual_skip_layers"
    assert comparison["selected_accuracy_score"] == pytest.approx(0.10)
    assert comparison["accuracy_regression_pct"] == pytest.approx(87.5)
    assert captured["tasks"] == [["gsm8k", "hellaswag"], ["gsm8k", "hellaswag"]]




@pytest.mark.model_pruning_gpu
def test_evaluate_stage_skip_layers_error_reports_plan_failure(monkeypatch, tmp_path):
    from QEfficient.model_pruning import nas_pipeline
    from QEfficient.model_pruning.qeff_model_optimizer.evaluation import plan_evaluator as pe
    from QEfficient.model_pruning.qeff_model_optimizer.api import loaders as loader_mod

    class FakeParam:
        device = types.SimpleNamespace(type="cpu", index=None)

    class FakeModel:
        def parameters(self):
            return iter([FakeParam()])

        def generate(self, **kwargs):
            return [[1]]

    class FakeTokenizer:
        def __call__(self, prompt, return_tensors="pt"):
            class Inputs(dict):
                def to(self, device):
                    return self
            return Inputs({"input_ids": [1]})

        def decode(self, tokens, skip_special_tokens=True):
            return "completion"

    class FakeLoader:
        def load(self, spec):
            return FakeModel(), FakeTokenizer()

    def fake_run_lm_eval(model, tokenizer, tasks, **kwargs):
        raise ImportError("Failed to import lm_eval: No module named 'lm_eval'")

    def fake_apply_plan(self, artifact, plan):
        artifact.plan = plan
        return artifact

    monkeypatch.setattr(loader_mod, "TransformersModelLoader", FakeLoader)
    monkeypatch.setattr(pe, "TransformersModelLoader", FakeLoader)
    monkeypatch.setattr(pe, "run_lm_eval", fake_run_lm_eval)
    monkeypatch.setattr(pe.NASSession, "apply_plan", fake_apply_plan)
    monkeypatch.setattr(pe.NASSession, "close", lambda self: None)

    with pytest.raises(RuntimeError) as exc_info:
        nas_pipeline.main([
            "evaluate",
            "--model", "dummy",
            "--skip-layers", "1", "2",
            "--datasets", "gsm8k",
            "--num-samples", "4",
            "--eval-method", "lm_eval",
            "--device-map", "cpu",
            "--output-dir", str(tmp_path),
        ])

    message = str(exc_info.value)
    assert "Manual skip-layer evaluation failed" in message
    assert "Plan errors:" in message
    assert "No module named 'lm_eval'" in message
    assert (tmp_path / "plan_results.json").exists()
    assert not (tmp_path / "best_plan.json").exists()


@pytest.mark.model_pruning_gpu
def test_evaluate_stage_raises_when_no_candidate_meets_threshold(monkeypatch, tmp_path):
    from QEfficient.model_pruning import nas_pipeline
    from QEfficient.model_pruning.qeff_model_optimizer.evaluation import plan_evaluator as pe
    from QEfficient.model_pruning.qeff_model_optimizer.api import loaders as loader_mod

    candidate_file = tmp_path / "candidate_plans.json"
    _write_candidate_plans(candidate_file)

    class FakeParam:
        device = types.SimpleNamespace(type="cpu", index=None)

    class FakeModel:
        def parameters(self):
            return iter([FakeParam()])

        def generate(self, **kwargs):
            return [[1]]

    class FakeTokenizer:
        def __call__(self, prompt, return_tensors="pt"):
            class Inputs(dict):
                def to(self, device):
                    return self
            return Inputs({"input_ids": [1]})

        def decode(self, tokens, skip_special_tokens=True):
            return "completion"

    class FakeLoader:
        def load(self, spec):
            return FakeModel(), FakeTokenizer()

    def fake_run_lm_eval(model, tokenizer, tasks, **kwargs):
        plan_layers = getattr(model, "_active_skip_layers", [])
        score = 0.80 if not plan_layers else 0.10
        return {"results": {task: {"acc,none": score} for task in tasks}}

    def fake_apply_plan(self, artifact, plan):
        layers = []
        for spec in plan.transforms:
            layers.extend(getattr(spec, "layers", []))
        artifact.model._active_skip_layers = layers
        artifact.plan = plan
        return artifact

    monkeypatch.setattr(loader_mod, "TransformersModelLoader", FakeLoader)
    monkeypatch.setattr(pe, "TransformersModelLoader", FakeLoader)
    monkeypatch.setattr(pe, "run_lm_eval", fake_run_lm_eval)
    monkeypatch.setattr(pe.NASSession, "apply_plan", fake_apply_plan)
    monkeypatch.setattr(pe.NASSession, "close", lambda self: None)

    with pytest.raises(RuntimeError, match="best_plan.json was not written"):
        nas_pipeline.main([
            "evaluate",
            "--model", "dummy",
            "--candidate-plans", str(candidate_file),
            "--datasets", "gsm8k",
            "--num-samples", "4",
            "--max-candidates", "2",
            "--accuracy-threshold", "5",
            "--eval-method", "lm_eval",
            "--accuracy-metric", "acc",
            "--device-map", "cpu",
            "--output-dir", str(tmp_path),
        ])

    assert not (tmp_path / "best_plan.json").exists()
    assert (tmp_path / "plan_results.json").exists()



def test_videomme_loader_normalizes_local_jsonl(tmp_path):
    from QEfficient.model_pruning.qeff_model_optimizer.evaluation.videomme import load_videomme_examples

    data_path = tmp_path / "videomme.jsonl"
    data_path.write_text(
        json.dumps(
            {
                "video_id": "sample_001",
                "question": "What happens first?",
                "options": ["A cat jumps", "A dog sleeps", "A car stops", "A door opens"],
                "answer": "B",
                "duration": "short",
                "subtitle": "A dog is resting indoors.",
            }
        )
        + "\n"
    )

    examples = load_videomme_examples(dataset_path=str(data_path), num_samples=1, use_subtitles=True)

    assert len(examples) == 1
    assert examples[0].video_id == "sample_001"
    assert examples[0].answer == "B"
    assert examples[0].duration == "short"
    assert "Subtitle:" in examples[0].prompt(use_subtitles=True)


def test_videomme_build_inputs_uses_qwen_vl_utils(monkeypatch, tmp_path):
    from QEfficient.model_pruning.qeff_model_optimizer.evaluation import videomme

    video_path = tmp_path / "sample.mp4"
    video_path.write_bytes(b"not-a-real-video")
    example = videomme.VideoMMEExample(
        sample_id="q1",
        video_id="v1",
        question="What is shown?",
        options=["a", "b", "c", "d"],
        answer="A",
        video_path=str(video_path),
    )

    calls = {}

    def fake_process_vision_info(messages):
        calls["messages"] = messages
        return ["image_inputs"], ["video_inputs"]

    qwen_mod = types.ModuleType("qwen_vl_utils")
    qwen_mod.process_vision_info = fake_process_vision_info
    monkeypatch.setitem(sys.modules, "qwen_vl_utils", qwen_mod)

    class FakeProcessor:
        def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=True):
            calls["template_messages"] = messages
            return "templated prompt"

        def __call__(self, **kwargs):
            return kwargs

    inputs = videomme.build_videomme_inputs(
        FakeProcessor(),
        example,
        prompt=example.prompt(),
        num_frames=8,
        fps=None,
    )

    assert calls["messages"][0][0]["content"][0]["type"] == "video"
    assert inputs["text"] == ["templated prompt"]
    assert inputs["images"] == ["image_inputs"]
    assert inputs["videos"] == ["video_inputs"]
    assert inputs["return_tensors"] == "pt"


def test_qwen_vl_skip_layer_adapter_returns_tuple_output():
    from QEfficient.model_pruning.qeff_model_optimizer.config.artifacts import ModelArtifact
    from QEfficient.model_pruning.qeff_model_optimizer.config.models import ModelSpec
    from QEfficient.model_pruning.qeff_model_optimizer.config.transforms import SkipLayersSpec, TransformationPlan
    from QEfficient.model_pruning.qeff_model_optimizer.transforms.skip_layers import SkipLayersTransform

    class FakeLayer:
        def forward(self, hidden_states, *args, **kwargs):
            return ("original", hidden_states)

    class FakeLanguageModel:
        def __init__(self):
            self.layers = [FakeLayer()]

    class FakeModel:
        def __init__(self):
            self.config = types.SimpleNamespace(model_type="qwen3_vl")
            self.language_model = FakeLanguageModel()

    model = FakeModel()
    artifact = ModelArtifact(
        artifact_id="artifact",
        model=model,
        tokenizer=None,
        model_spec=ModelSpec(model_id="dummy-vl"),
        plan=TransformationPlan(),
    )

    SkipLayersTransform().apply(artifact, SkipLayersSpec(layers=[0]))

    output = model.language_model.layers[0].forward("hidden", output_attentions=True)
    assert output == ("hidden", None)


def test_videomme_evaluator_reports_accuracy_by_duration(tmp_path):
    from QEfficient.model_pruning.qeff_model_optimizer.evaluation.videomme import evaluate_videomme

    data_path = tmp_path / "videomme.json"
    data_path.write_text(
        json.dumps(
            [
                {
                    "id": "short_ok",
                    "video_id": "v1",
                    "question": "Pick the answer.",
                    "A": "wrong",
                    "B": "right",
                    "C": "wrong",
                    "D": "wrong",
                    "answer": "B",
                    "duration": "short",
                },
                {
                    "id": "long_bad",
                    "video_id": "v2",
                    "question": "Pick the answer.",
                    "A": "wrong",
                    "B": "wrong",
                    "C": "right",
                    "D": "wrong",
                    "answer": "C",
                    "duration": "long",
                },
            ]
        )
    )

    class FakeParam:
        device = types.SimpleNamespace(type="cpu", index=None)

    class FakeModel:
        def parameters(self):
            return iter([FakeParam()])

        def generate(self, **kwargs):
            return [[1]]

    class FakeProcessor:
        def __call__(self, **kwargs):
            class Inputs(dict):
                def to(self, device):
                    return self
            return Inputs({"input_ids": [1]})

        def batch_decode(self, output_ids, skip_special_tokens=True):
            return ["Answer: B"]

    report = evaluate_videomme(
        FakeModel(),
        FakeProcessor(),
        dataset_path=str(data_path),
        num_samples=2,
        generation_len=4,
    )

    assert report.overall_accuracy == pytest.approx(0.5)
    assert report.per_duration_accuracy["short"] == pytest.approx(1.0)
    assert report.per_duration_accuracy["long"] == pytest.approx(0.0)


@pytest.mark.model_pruning_gpu
def test_evaluate_stage_videomme_writes_baseline_and_optimized_accuracy(monkeypatch, tmp_path):
    from QEfficient.model_pruning import nas_pipeline
    from QEfficient.model_pruning.qeff_model_optimizer.api import loaders as loader_mod
    from QEfficient.model_pruning.qeff_model_optimizer.evaluation import plan_evaluator as pe

    data_path = tmp_path / "videomme.jsonl"
    data_path.write_text(
        json.dumps(
            {
                "id": "q1",
                "video_id": "v1",
                "question": "Which option is correct?",
                "options": ["baseline", "optimized", "other", "none"],
                "answer": "B",
                "duration": "short",
            }
        )
        + "\n"
    )

    class FakeParam:
        device = types.SimpleNamespace(type="cpu", index=None)

    class FakeModel:
        def parameters(self):
            return iter([FakeParam()])

        def generate(self, **kwargs):
            return [[1]]

    class FakeProcessor:
        def __call__(self, **kwargs):
            class Inputs(dict):
                def to(self, device):
                    return self
            return Inputs({"input_ids": [1]})

        def batch_decode(self, output_ids, skip_special_tokens=True):
            return ["Answer: B" if getattr(self, "active_skip_layers", []) else "Answer: A"]

    fake_processor = FakeProcessor()

    class FakeLoader:
        def load(self, spec):
            return FakeModel(), fake_processor

    def fake_apply_plan(self, artifact, plan):
        layers = []
        for spec in plan.transforms:
            layers.extend(getattr(spec, "layers", []))
        artifact.model._active_skip_layers = layers
        artifact.tokenizer.active_skip_layers = layers
        artifact.plan = plan
        return artifact

    monkeypatch.setattr(loader_mod, "TransformersModelLoader", FakeLoader)
    monkeypatch.setattr(pe, "TransformersModelLoader", FakeLoader)
    monkeypatch.setattr(pe.NASSession, "apply_plan", fake_apply_plan)
    monkeypatch.setattr(pe.NASSession, "close", lambda self: None)

    nas_pipeline.main([
        "evaluate",
        "--model", "dummy-vl",
        "--skip-layers", "1",
        "--datasets", "videomme",
        "--num-samples", "1",
        "--eval-method", "videomme",
        "--videomme-dataset-path", str(data_path),
        "--generation-len", "4",
        "--device-map", "cpu",
        "--output-dir", str(tmp_path / "out"),
    ])

    summary = json.loads((tmp_path / "out" / "evaluation_summary.json").read_text())
    plan_results = json.loads((tmp_path / "out" / "plan_results.json").read_text())

    assert summary["eval_method"] == "videomme"
    assert summary["comparison"]["baseline_accuracy_score"] == pytest.approx(0.0)
    assert summary["comparison"]["selected_accuracy_score"] == pytest.approx(1.0)
    assert plan_results[0]["videomme_report"]["overall_accuracy"] == pytest.approx(1.0)
    assert plan_results[1]["videomme_report"]["overall_accuracy"] == pytest.approx(0.0)



def test_qaic_runner_videomme_branch_reports_accuracy(tmp_path):
    from QEfficient.model_pruning.qeff_model_optimizer.evaluation.qaic_benchmark import QAICBenchmarkRunner

    data_path = tmp_path / "videomme.jsonl"
    data_path.write_text(
        json.dumps(
            {
                "id": "q1",
                "video_id": "v1",
                "question": "Which option is correct?",
                "options": ["wrong", "right", "other", "none"],
                "answer": "B",
                "duration": "short",
            }
        )
        + "\n"
    )

    class FakePrepared:
        def generate(self, **kwargs):
            print("Completion : Answer: B\n=")

    class FakeProcessor:
        def __call__(self, **kwargs):
            return {"input_ids": [1]}

    runner = QAICBenchmarkRunner(
        model_id="dummy-vl",
        videomme_dataset_path=str(data_path),
        videomme_num_samples=1,
        videomme_num_frames=1,
    )

    per_prompt, completions, report = runner._run_videomme(FakePrepared(), FakeProcessor())

    assert report["overall_accuracy"] == pytest.approx(1.0)
    assert report["per_duration_accuracy"]["short"] == pytest.approx(1.0)
    assert report["results"][0]["prediction"] == "B"
    assert completions[0]["completion"] == "Answer: B"
    assert isinstance(per_prompt, list)
