import importlib.util
import json
import sys
from pathlib import Path
from unittest.mock import patch

import pytest

EXAMPLES_DIR = Path(__file__).resolve().parent.parent.parent / "examples"
# Make example_scripts a list of strings instead of Path objects so that pytest can use it in its
# automatic printouts. We convert the elements back to Paths in the test function.
example_scripts = [str(p) for p in sorted(EXAMPLES_DIR.rglob("*.py"))]


@pytest.mark.parametrize("example_script", [str(p) for p in example_scripts])
@pytest.mark.timeout(60)
@pytest.mark.integration
def test_example_main(example_script: str):
    """Dynamically import and execute the main function from an example script."""
    example_script = Path(example_script)
    if "RUN_IN_INTEGRATION_TEST = False" in example_script.read_text():
        pytest.skip("research entry point has its own bounded integration tests")

    # Add the examples directory to sys.path to resolve imports
    sys.path.insert(0, str(EXAMPLES_DIR))

    # Dynamically import the module
    spec = importlib.util.spec_from_file_location("example_module", example_script)
    example_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(example_module)

    # Ensure the script has a main function
    assert hasattr(example_module, "main"), f"{example_script.name} has no main() function."

    # Remove render function to enable headless testing
    with patch("crazyflow.sim.sim.Sim.render", return_value=None):
        example_module.main()

    # Clean up sys.path
    sys.path.remove(str(EXAMPLES_DIR))


@pytest.mark.integration
def test_mellinger_scaling_construction_diagnostics_avoid_simulation_and_jit(
    tmp_path: Path,
) -> None:
    script = EXAMPLES_DIR / "jax/mellinger_scaling_benchmark.py"
    spec = importlib.util.spec_from_file_location("mellinger_scaling_benchmark", script)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    def forbidden(*_args, **_kwargs):
        raise AssertionError("construction-only diagnostics reached simulation or jax.jit")

    output_dir = tmp_path / "diagnostics"
    arguments = module.parse_args(
        [
            "--config",
            "configs/research/mellinger/workstation.json",
            "--trajectory-diagnostics-only",
            "--diagnostic-horizons",
            "400",
            "--diagnostic-seed-count",
            "1",
            "--output-dir",
            str(output_dir),
        ]
    )
    with (
        patch.object(module, "build_split_pipeline", side_effect=forbidden),
        patch.object(module.jax, "jit", side_effect=forbidden),
    ):
        module.main(arguments)

    raw = json.loads((output_dir / "trajectory_diagnostics_raw.json").read_text())
    summary = json.loads((output_dir / "trajectory_diagnostics_summary.json").read_text())
    assert raw["construction_only_contract"] == {
        "jax_jit_called": False,
        "simulation_built": False,
        "test_manifest_opened": False,
        "test_split_built_or_evaluated": False,
        "training_run": False,
        "validation_run": False,
    }
    assert summary["horizons"][0]["success_count"] == 1


@pytest.mark.integration
@pytest.mark.parametrize(
    ("error", "expected_phase"),
    [
        (
            ValueError("no valid trajectory found in 16 deterministic attempts"),
            "trajectory_construction_before_jit",
        ),
        (RuntimeError("mass realization failed"), "episode_batch_realization_before_jit"),
    ],
)
def test_mellinger_scaling_benchmark_records_specific_pre_jit_failure_phases(
    tmp_path: Path, error: Exception, expected_phase: str
) -> None:
    script = EXAMPLES_DIR / "jax/mellinger_scaling_benchmark.py"
    spec = importlib.util.spec_from_file_location(
        f"mellinger_scaling_benchmark_{expected_phase}", script
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    output_dir = tmp_path / expected_phase
    arguments = module.parse_args(
        [
            "--config",
            "configs/research/mellinger/workstation.json",
            "--worlds",
            "1",
            "--horizon",
            "100",
            "--output-dir",
            str(output_dir),
        ]
    )

    with (
        patch.object(module, "build_split_pipeline", return_value=object()),
        patch.object(module, "build_episode_batch", side_effect=error),
        pytest.raises(type(error), match=str(error)),
    ):
        module.main(arguments)

    failure = json.loads((output_dir / "benchmark.json").read_text())
    assert failure["schema_version"] == "crazyflow.mellinger_scaling_benchmark.v3"
    assert failure["failure_phase"] == expected_phase
    assert failure["test_manifest_opened"] is False
