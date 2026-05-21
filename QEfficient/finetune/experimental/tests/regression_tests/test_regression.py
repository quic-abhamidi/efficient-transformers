import gc
import json
import os
from typing import List, Optional

import numpy as np
import pytest
import torch.distributed as dist

# ✅ Import source test (IMPORTANT)
import QEfficient.finetune.experimental.tests.test_ddp as ddp_test_module
import QEfficient.finetune.experimental.tests.test_pipeline_parallelism as pp_test_module
from QEfficient.finetune.experimental.core.logger import Logger
from QEfficient.finetune.experimental.tests.test_ddp import STORE_PATH

logger = Logger(__name__)

# ============================================================================
# Config
# ============================================================================

BASELINE_SDK_VERSION = "SDK_1.22.0.32"
UPDATE_GOLDEN = os.getenv("UPDATE_GOLDEN") == "1"

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
GOLDEN_DIR = os.path.join(BASE_DIR, "goldens")

LOSS_TOL = 1e-3  # Matches pipeline test tolerance
OUTPUT_DIR_DDP = "/tmp/test_regression_ddp"
OUTPUT_DIR_SINGLE = "/tmp/test_regression_single"
OUTPUT_DIR_PP2 = "/tmp/test_regression_pp2"

# ============================================================================
# Golden Utils
# ============================================================================


def _golden_path(name: str) -> str:
    return os.path.join(GOLDEN_DIR, f"{name}_{BASELINE_SDK_VERSION}.json")


def save_golden(name: str, data: dict):
    os.makedirs(GOLDEN_DIR, exist_ok=True)
    path = _golden_path(name)

    with open(path, "w") as f:
        json.dump(data, f, indent=2)

    logger.info(f"✅ Saved golden: {path}")


def load_golden(name: str) -> Optional[dict]:
    path = _golden_path(name)

    if not os.path.exists(path):
        logger.warning(f"Missing golden: {path}")
        return None

    with open(path) as f:
        return json.load(f)


# ============================================================================
# Helpers
# ============================================================================


def extract_losses(loss_list):
    return [loss for _, loss in loss_list]


def compute_diff(curr: List[float], base: List[float]):
    diffs = [abs(c - b) for c, b in zip(curr, base)]
    return max(diffs), float(np.mean(diffs))


def compare(name: str, curr: List[float], base: List[float], tol: float):
    assert len(curr) == len(base), f"{name} length mismatch"

    max_diff, avg_diff = compute_diff(curr, base)

    logger.info(f"{name}: max={max_diff:.6f}, avg={avg_diff:.6f}, tol={tol}")

    assert avg_diff < tol, f"{name} regression: {avg_diff} > {tol}"


def cleanup():
    """Clean up distributed process group."""
    if dist.is_initialized():
        dist.destroy_process_group()


# ============================================================================
# Regression Tests (DRIVEN BY PIPELINE PARITY)
# ============================================================================


@pytest.mark.skipif(os.getenv("RUN_REGRESSION") != "1", reason="Run only when RUN_REGRESSION=1")
class TestPipelineRegression:
    def _run_pipeline(self):
        """
        ✅ Run pipeline parity test ONLY ONCE
        ✅ Reuse returned losses for regression
        """
        # Clean file store
        if os.path.exists(STORE_PATH):
            os.remove(STORE_PATH)
        # Ensure unique port per run
        os.environ["MASTER_PORT"] = str(ddp_test_module.TestDDPPipelineParity._get_unique_port())
        logger.info("Running DDP pipeline...")
        ddp_loss, _ = ddp_test_module.TestDDPPipelineParity()._run_ddp(
            ddp_test_module.TestDDPPipelineParity()._build_config_dict(
                backend="qccl",
                output_dir=OUTPUT_DIR_DDP,
            ),
            port=int(os.environ["MASTER_PORT"]),
        )
        gc.collect()
        cleanup()
        logger.info("Running single-device pipeline...")
        single_loss, _ = ddp_test_module.TestDDPPipelineParity()._run_single(
            ddp_test_module.TestDDPPipelineParity()._build_config_dict(
                backend=None,
                output_dir=OUTPUT_DIR_SINGLE,
            )
        )
        gc.collect()
        cleanup()
        # Ensure unique port per run
        pp2_loss, _ = pp_test_module.TestPPPipelineParity()._run_pipeline(
            pp_test_module.TestPPPipelineParity()._build_config_manager(
                pp_degree=2,
                output_dir=OUTPUT_DIR_PP2,
            )
        )
        gc.collect()
        cleanup()
        return single_loss, ddp_loss, pp2_loss

    # ----------------------------------------------------------------------

    def test_regression_pipeline_losses(self):
        """
        ✅ Main regression entrypoint
        ✅ Uses pipeline parity test to generate results
        ✅ Saves OR compares against golden
        """

        single_loss, ddp_loss, pp2_loss = self._run_pipeline()

        data = {
            "single": single_loss,
            "single_golden": "finetuning_pipeline_single",
            "ddp": ddp_loss,
            "ddp_golden": "finetuning_pipeline_ddp",
            "pp": pp2_loss,
            "pp_golden": "finetuning_pipeline_pp2",
        }

        if UPDATE_GOLDEN:
            logger.info("UPDATE_GOLDEN=1 → saving new baseline")
            save_golden(data["single_golden"], data["single"])
            save_golden(data["ddp_golden"], data["ddp"])
            save_golden(data["pp_golden"], data["pp"])
            return

        golden_single = load_golden("finetuning_pipeline_single")
        golden_ddp = load_golden("finetuning_pipeline_ddp")
        golden_pp = load_golden("finetuning_pipeline_pp2")

        if golden_single is None or golden_ddp is None or golden_pp is None:
            logger.info("No golden found → creating new baseline")
            save_golden(data["single_golden"], data["single"])
            save_golden(data["ddp_golden"], data["ddp"])
            save_golden(data["pp_golden"], data["pp"])
            return

        # ------------------------------------------------------------------
        # ✅ Compare single-device
        # ------------------------------------------------------------------

        curr_single = extract_losses(single_loss)
        base_single = extract_losses(golden_single)
        compare("Single Loss", curr_single, base_single, LOSS_TOL)

        # ------------------------------------------------------------------
        # ✅ Compare DDP
        # ------------------------------------------------------------------
        curr_ddp = extract_losses(ddp_loss)
        base_ddp = extract_losses(golden_ddp)
        compare("DDP Loss", curr_ddp, base_ddp, LOSS_TOL)

        # ------------------------------------------------------------------
        # ✅ Compare PP
        # ------------------------------------------------------------------
        curr_pp = extract_losses(pp2_loss)
        base_pp = extract_losses(golden_pp)
        compare("PP Loss", curr_pp, base_pp, LOSS_TOL)
