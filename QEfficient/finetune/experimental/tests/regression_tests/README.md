# DDP Regression Testing

Regression tests for validating DDP training across different SDK versions.

## Overview

The regression test suite compares current DDP training results against **golden baseline** values stored in JSON files. This ensures that:

- ✅ Loss values don't regress when SDK is updated
- ✅ Training metrics remain stable across versions  
- ✅ Loss differences stay < 1e-2 from baseline
- ✅ DDP/Single-device parity is maintained

## Structure

```
regression_tests/
├── test_regression.py      # Regression test suite (optional, runs separately)
├── goldens/                # Directory for storing golden baseline values
│   ├── finetuning_pipeline_single_SDK_1.22.0.32.json
│   └── finetuning_pipeline_ddp_SDK_1.22.0.32.json
└── README.md               # This file
```

## Running Regression Tests

### 1. **Create Initial Baseline**

First run to establish baseline for current SDK:

```bash
UPDATE_GOLDEN=1 pytest regression_tests/test_regression.py -v
```

This will:
- Run both single-device and DDP training
- Save loss trajectories as golden baselines
- Store files in `regression_tests/goldens/`

### 2. **Validate Against Baseline**

Subsequent runs to validate against established baseline:

```bash
pytest regression_tests/test_regression.py -v
```

This will:
- Run training with current SDK
- Compare losses against golden baseline
- PASS if max loss difference < 1e-2
- FAIL if regression detected

### 3. **Run Specific Test**

Test only single-device or DDP:

```bash
# Test single-device regression
pytest regression_tests/test_regression.py::TestDDPRegressionBaseline::test_regression_single_device_losses -v

# Test DDP regression
pytest regression_tests/test_regression.py::TestDDPRegressionBaseline::test_regression_ddp_losses -v

# Test loss parity
pytest regression_tests/test_regression.py::TestDDPRegressionBaseline::test_regression_loss_parity -v
```

## Updating Baseline for New SDK

When upgrading to a new QAIC SDK version:

1. **Update SDK version constant** in `test_regression.py`:
   ```python
   BASELINE_SDK_VERSION = "SDK_1.23.0.00"  # New version
   ```

2. **Generate new golden files**:
   ```bash
   UPDATE_GOLDEN=1 pytest regression_tests/test_regression.py -v
   ```

3. **Verify baseline quality**:
   ```bash
   pytest regression_tests/test_regression.py -v
   ```

## Test Cases

### 1. `test_regression_single_device_losses`
- **Purpose**: Validate single-device training losses remain stable
- **Tolerance**: < 1e-2 max difference
- **Golden**: `finetuning_pipeline_single_{SDK_VERSION}.json`

### 2. `test_regression_ddp_losses`
- **Purpose**: Validate DDP training losses remain stable
- **Tolerance**: < 1e-2 max difference
- **Golden**: `finetuning_pipeline_ddp_{SDK_VERSION}.json`

### 3. `test_regression_loss_parity`
- **Purpose**: Ensure DDP vs single-device parity is maintained
- **Tolerance**: < 1e-2 max parity difference
- **Golden**: No golden (compares current single/DDP in same run)

## Understanding Golden Files

Golden baseline files are JSON with loss trajectories:

```json
{
  "loss": [
    [1, 3.245],      // [step, loss]
    [2, 3.123],
    [3, 2.987],
    ...
  ]
}
```

**Format**:
- `[step, loss]`: Tuple of training step and loss value
- Multiple steps across training trajectory
- Used to validate complete loss curve, not just final value

## Interpreting Results

### ✅ PASSED
```
Regression Test: Single-Device Training Losses
  Max difference: 1.23e-04
  Avg difference: 4.56e-05
  Tolerance: 1.00e-02
✅ Single-Device Training Losses regression test PASSED
```

**Meaning**: Current training matches baseline within tolerance.

### ❌ FAILED
```
AssertionError: Single-Device Training Losses regression detected! 
Max diff 1.25e-02 exceeds tolerance 1.00e-02
```

**Meaning**: Loss trajectory diverged beyond acceptable threshold.

## Troubleshooting

### Golden files not found
```
📌 Golden baseline not found: regression_tests/goldens/finetuning_pipeline_single_SDK_1.22.0.32.json

Solution: Run with UPDATE_GOLDEN=1 to create baseline
UPDATE_GOLDEN=1 pytest regression_tests/test_regression.py -v
```

### Regression detected
```
AssertionError: Loss parity regression! Max diff 1.50e-02 exceeds tolerance 1.00e-02

Solutions:
1. Review recent code changes for numerical differences
2. Check if tolerance needs adjustment (1e-2 may be too strict)
3. Verify SDK version compatibility
4. Re-establish baseline if intentional changes made
```

### Tests require QAIC devices
```
SKIPPED: Requires at least 2 QAIC devices

Solution: Run on machine with 2+ QAIC devices, or:
- Skip regression tests: pytest -m "not regression"
- Focus on main test suite
```

## Configuration Constants

| Constant | Value | Purpose |
|----------|-------|---------|
| `BASELINE_SDK_VERSION` | `"SDK_1.22.0.32"` | SDK version for golden files |
| `LOSS_REGRESSION_TOLERANCE` | `1e-2` | Max allowed loss difference |
| `UPDATE_GOLDEN` | `os.getenv("UPDATE_GOLDEN") == "1"` | Enable baseline creation |
| `WORLD_SIZE` | `2` | Number of DDP processes |
| `_MAX_STEPS` | `50` | Training steps per test |

## CI/CD Integration

### Run in CI pipeline
```bash
# Create baseline (first time)
UPDATE_GOLDEN=1 pytest regression_tests/test_regression.py -v

# Validate in subsequent runs
pytest regression_tests/test_regression.py -v -m regression
```

### Skip regression tests (faster CI)
```bash
pytest tests/ -v -m "not regression"
```

## Best Practices

1. ✅ **Create baseline on stable environment** - Consistent hardware, clean install
2. ✅ **Version golden files** - Include SDK version in filename
3. ✅ **Document changes** - Note why baseline was updated
4. ✅ **Review loss curves** - Validate baseline looks reasonable before committing
5. ✅ **Tolerance tuning** - Adjust if SDK changes cause expected divergence
6. ✅ **Regular validation** - Run regression tests after each SDK update

## Example Workflow

```bash
# 1. Upgrade to new SDK
# (install new SDK dependencies)

# 2. Create new baseline
UPDATE_GOLDEN=1 pytest regression_tests/test_regression.py -v
# ✅ Baseline created in regression_tests/goldens/

# 3. Commit golden files
git add regression_tests/goldens/
git commit -m "Add regression baseline for SDK_1.23.0.00"

# 4. Validate in CI
pytest regression_tests/test_regression.py -v
# ✅ All regression tests PASSED

# 5. Continue development
# Regression tests now validate against new SDK baseline
```

## References

- Main test suite: `../test_ddp.py`
- Logger: `QEfficient.finetune.experimental.core.logger.Logger`
- FineTuningPipeline: `QEfficient.cloud.finetune_experimental.FineTuningPipeline`
- ConfigManager: `QEfficient.finetune.experimental.core.config_manager.ConfigManager`
