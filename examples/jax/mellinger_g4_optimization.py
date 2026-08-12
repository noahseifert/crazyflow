"""Run bounded technical G4 gates for exactly kp_xy plus kp_z."""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import os
import platform
import stat
import subprocess
import sys
from pathlib import Path
from typing import Any

from crazyflow.control.mellinger.research import g4_optimization as g4

RUN_IN_INTEGRATION_TEST = False

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]


class G4CliError(g4.G4ContractError):
    """Raised before or during a contract-invalid CLI target."""


def parse_args(argv: list[str] | tuple[str, ...] | None = None) -> argparse.Namespace:
    """Parse one and only one bounded G4 target."""
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_mutually_exclusive_group(required=True)
    modes.add_argument("--write-source-origin-record", type=Path)
    modes.add_argument("--write-backend-source-origin-record", type=Path)
    modes.add_argument("--dry-run", action="store_true")
    modes.add_argument("--theta-zero-parity", action="store_true")
    modes.add_argument("--resource-rebenchmark-only", action="store_true")
    modes.add_argument("--single-smokes", action="store_true")
    modes.add_argument("--joint-smoke", action="store_true")
    modes.add_argument("--longrun", action="store_true")
    modes.add_argument("--resume-checkpoint", type=Path)
    modes.add_argument("--m8-longrun", action="store_true")
    modes.add_argument("--m8-resume-checkpoint", type=Path)
    modes.add_argument("--backend-parity", action="store_true")
    modes.add_argument("--backend-throughput", action="store_true")
    modes.add_argument("--backend-checkpoint-run", action="store_true")
    modes.add_argument("--backend-checkpoint-resume", type=Path)
    modes.add_argument("--resume-reproduction", action="store_true")
    modes.add_argument("--resume-worker-checkpoint", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--resume-worker-output", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--source-origin-record", type=Path)
    parser.add_argument("--max-updates-this-process", type=int)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--backend", choices=("cpu", "gpu"))
    parser.add_argument(
        "--runtime-profile", choices=("local_cpu", "remote_pixi_cpu", "remote_pixi_gpu")
    )
    parser.add_argument("--runtime-contract", type=Path)
    parser.add_argument("--input-manifest", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--phase", choices=("warmup", "steady"))
    parser.add_argument("--sample-index", type=int, choices=(1, 2, 3))
    parser.add_argument("--effective-worlds", type=int, choices=(32,))
    parser.add_argument("--microbatches", type=int, choices=(8,))
    parser.add_argument("--worlds-per-microbatch", type=int, choices=(4,))
    args = parser.parse_args(argv)

    backend_values = (
        args.backend,
        args.runtime_profile,
        args.runtime_contract,
        args.input_manifest,
        args.output,
        args.phase,
        args.sample_index,
        args.effective_worlds,
        args.microbatches,
        args.worlds_per_microbatch,
    )
    if args.write_backend_source_origin_record is not None:
        if (
            args.backend != "cpu"
            or args.runtime_profile != "local_cpu"
            or args.input_manifest is None
            or args.runtime_contract is not None
            or any(
                value is not None
                for value in (
                    args.config,
                    args.source_origin_record,
                    args.max_updates_this_process,
                    args.output_dir,
                    args.output,
                    args.phase,
                    args.sample_index,
                    args.effective_worlds,
                    args.microbatches,
                    args.worlds_per_microbatch,
                    args.resume_worker_output,
                )
            )
        ):
            parser.error("backend source-origin arguments changed")
        return args

    backend_mode = (
        args.backend_parity
        or args.backend_throughput
        or args.backend_checkpoint_run
        or args.backend_checkpoint_resume is not None
    )
    if backend_mode:
        if any(
            value is None
            for value in (
                args.backend,
                args.runtime_profile,
                args.runtime_contract,
                args.input_manifest,
            )
        ):
            parser.error("backend evidence requires its complete runtime contract")
        if any(
            value is not None
            for value in (args.config, args.source_origin_record, args.resume_worker_output)
        ):
            parser.error("legacy arguments are invalid for backend evidence")
        if args.backend_parity:
            if args.output is None or any(
                value is not None
                for value in (
                    args.phase,
                    args.sample_index,
                    args.effective_worlds,
                    args.microbatches,
                    args.worlds_per_microbatch,
                    args.output_dir,
                    args.max_updates_this_process,
                )
            ):
                parser.error("backend parity arguments changed")
        elif args.backend_throughput:
            if (
                args.output is None
                or args.phase is None
                or args.sample_index is None
                or args.effective_worlds != 32
                or args.microbatches != 8
                or args.worlds_per_microbatch != 4
                or args.output_dir is not None
                or args.max_updates_this_process is not None
                or (args.phase == "warmup" and args.sample_index != 1)
            ):
                parser.error("backend throughput arguments changed")
        else:
            if (
                args.output_dir is None
                or args.output is not None
                or any(
                    value is not None
                    for value in (
                        args.phase,
                        args.sample_index,
                        args.effective_worlds,
                        args.microbatches,
                        args.worlds_per_microbatch,
                    )
                )
            ):
                parser.error("backend checkpoint arguments changed")
            if args.backend_checkpoint_run and args.max_updates_this_process not in {10, 20}:
                parser.error("backend checkpoint run requires 10 or 20 updates")
            if args.backend_checkpoint_resume is not None and (
                not args.backend_checkpoint_resume.is_absolute()
                or args.max_updates_this_process != 10
            ):
                parser.error("backend checkpoint resume requires absolute path and 10 updates")
        return args

    if any(value is not None for value in backend_values):
        parser.error("backend arguments require a backend mode")

    if args.write_source_origin_record is not None:
        if any(
            value is not None
            for value in (
                args.config,
                args.source_origin_record,
                args.max_updates_this_process,
                args.output_dir,
                args.resume_worker_output,
            )
        ):
            parser.error("source-origin generation accepts only its record path")
        return args

    required = (args.config, args.source_origin_record)
    if any(value is None for value in required):
        parser.error("every G4 target requires --config and --source-origin-record")

    if args.resume_worker_checkpoint is not None:
        if args.resume_worker_output is None or args.output_dir is not None:
            parser.error("resume worker requires only checkpoint and worker output paths")
        if args.max_updates_this_process is not None:
            parser.error("resume worker update count is fixed internally")
        return args

    if args.resume_worker_output is not None:
        parser.error("--resume-worker-output is private to the resume worker")
    if args.output_dir is None:
        parser.error("public G4 targets require --output-dir")
    if (
        args.longrun
        or args.resume_checkpoint is not None
        or args.m8_longrun
        or args.m8_resume_checkpoint is not None
    ):
        if (
            args.max_updates_this_process is None
            or not 1 <= args.max_updates_this_process <= g4.LONGRUN_MAX_UPDATES
        ):
            parser.error("longrun/resume requires --max-updates-this-process in 1..100")
        resume_path = args.resume_checkpoint or args.m8_resume_checkpoint
        if resume_path is not None and not resume_path.is_absolute():
            parser.error("resume checkpoint must be absolute")
    elif args.joint_smoke:
        if args.max_updates_this_process != 10:
            parser.error("joint smoke requires --max-updates-this-process 10")
    elif args.max_updates_this_process is not None:
        parser.error("only joint smoke accepts --max-updates-this-process")
    return args


def _validate_record_path(path: Path) -> Path:
    """Require an absent record under an existing fresh mode-0700 task root."""
    if not path.is_absolute() or path.name != "source-origin.json":
        raise G4CliError("source-origin record must be an absolute .../source-origin.json path")
    parent = path.parent
    if parent.parent != Path("/tmp") or not parent.name.startswith("gr-g4-004-"):
        raise G4CliError("source-origin record must be under /tmp/gr-g4-004-*")
    if parent.is_symlink() or stat.S_IMODE(parent.stat().st_mode) != 0o700:
        raise G4CliError("source-origin root must be a mode-0700 non-symlink directory")
    if path.exists() or path.is_symlink():
        raise G4CliError("source-origin record must be absent")
    return path


def _validate_worker_output(path: Path) -> Path:
    """Require the exact absent result file under a fresh worker task root."""
    if not path.is_absolute() or path.name != "resume-worker-result.json":
        raise G4CliError("resume worker output path changed")
    output = path.parent
    if output.name != "output" or not output.parent.name.startswith("gr-g4-004-"):
        raise G4CliError("resume worker output is outside the G4 task root")
    task_root = output.parent
    if (
        task_root.parent != Path("/tmp")
        or task_root.is_symlink()
        or stat.S_IMODE(task_root.stat().st_mode) != 0o700
    ):
        raise G4CliError("resume worker task root is invalid")
    if output.exists() or output.is_symlink() or path.exists() or path.is_symlink():
        raise G4CliError("resume worker output must be wholly absent")
    return path


def _common_preflight(args: argparse.Namespace) -> tuple[dict[str, Any], str]:
    """Run public-input, config, runtime, origin, protection, and source gates."""
    if REPOSITORY_ROOT != g4.EXPECTED_REPOSITORY_ROOT:
        raise G4CliError("CLI is not executing from the exact G4 worktree")
    g4.required_source_files_exist(REPOSITORY_ROOT)
    g4.validate_runtime_environment(REPOSITORY_ROOT)
    g4.validate_source_origin_record(REPOSITORY_ROOT, args.source_origin_record)
    public = g4.validate_public_inputs(REPOSITORY_ROOT)
    config, config_sha256 = g4.load_and_validate_config(args.config, REPOSITORY_ROOT)
    if g4.protected_contract()["contents_opened_or_hashed"]:
        raise G4CliError("protected sentinel contract changed")
    return {"public_inputs": public, "config": config}, config_sha256


def _backend_preflight(args: argparse.Namespace) -> tuple[dict[str, Any], dict[str, Any], str, str]:
    """Validate the relocatable manifest and observed Backend-Evidence runtime."""
    manifest = g4.load_backend_input_manifest(args.input_manifest, REPOSITORY_ROOT)
    runtime_path = args.runtime_contract
    if (
        runtime_path.is_symlink()
        or not runtime_path.is_file()
        or stat.S_IMODE(runtime_path.stat().st_mode) != 0o600
        or runtime_path.stat().st_uid != os.getuid()
    ):
        raise G4CliError("backend runtime contract type, owner, or mode changed")
    runtime = json.loads(runtime_path.read_bytes())
    observed = _observe_backend_runtime(args)
    runtime = g4.validate_backend_runtime_payload(
        runtime,
        expected_repository_root=REPOSITORY_ROOT,
        expected_backend=args.backend,
        expected_profile=args.runtime_profile,
        observed_payload=observed,
    )
    return (manifest, runtime, g4.sha256_file(args.input_manifest), g4.sha256_file(runtime_path))


def _git_observation(*args: str) -> str:
    result = subprocess.run(
        ("git", *args), cwd=REPOSITORY_ROOT, check=True, capture_output=True, text=True
    )
    return result.stdout.strip()


def _distribution_version(name: str, *, required: bool) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        if required:
            raise G4CliError(f"required runtime distribution is missing: {name}") from None
        return None


def _required_absolute_environment_path(name: str) -> str:
    value = os.environ.get(name)
    if value is None or not Path(value).is_absolute():
        raise G4CliError(f"backend runtime path is missing or non-absolute: {name}")
    resolved = Path(value).resolve(strict=True)
    if resolved.is_symlink() or resolved.stat().st_uid != os.getuid():
        raise G4CliError(f"backend runtime path owner or type changed: {name}")
    return str(resolved)


def _observe_backend_runtime(args: argparse.Namespace) -> dict[str, Any]:
    """Observe the running process independently from the loaded runtime contract."""
    root = REPOSITORY_ROOT.resolve(strict=True)
    if Path.cwd().resolve(strict=True) != root:
        raise G4CliError("backend runtime CWD differs from repository root")
    executable = Path(sys.executable)
    target = executable.resolve(strict=True)
    actual_backend = g4.jax.default_backend()
    devices = tuple(g4.jax.devices())
    device_records = [
        {"id": int(device.id), "platform": device.platform, "device_kind": device.device_kind}
        for device in devices
    ]
    pixi_name = os.environ.get("PIXI_ENVIRONMENT_NAME")
    if pixi_name is None:
        manager = "repository_venv"
        environment_name = None
        pixi_version = None
        observed_profile = "local_cpu"
    else:
        manager = "pixi"
        environment_name = pixi_name
        pixi = os.environ.get("PIXI_EXE")
        if pixi is None or not Path(pixi).is_absolute():
            raise G4CliError("Pixi runner path is not independently observable")
        version_result = subprocess.run(
            (pixi, "--version"), check=True, capture_output=True, text=True
        ).stdout.strip()
        pixi_version = version_result.removeprefix("pixi ")
        observed_profile = "remote_pixi_gpu" if actual_backend == "gpu" else "remote_pixi_cpu"
    flags = {
        "pythonpath": os.environ.get("PYTHONPATH"),
        "python_dont_write_bytecode": os.environ.get("PYTHONDONTWRITEBYTECODE"),
        "python_no_user_site": os.environ.get("PYTHONNOUSERSITE"),
        "jax_enable_x64": os.environ.get("JAX_ENABLE_X64"),
        "jax_platforms": os.environ.get("JAX_PLATFORMS"),
        "xla_flags": os.environ.get("XLA_FLAGS"),
    }
    if any(value is None for value in flags.values()):
        raise G4CliError("backend runtime environment flag is not independently observable")
    flags["xla_flags_sha256"] = g4.sha256_bytes(flags["xla_flags"].encode())
    status = _git_observation("status", "--porcelain=v1", "--untracked-files=all")
    plugin_required = actual_backend == "gpu"
    return {
        "schema_version": g4.BACKEND_RUNTIME_SCHEMA_VERSION,
        "profile": observed_profile,
        "backend": actual_backend,
        "repository": {
            "root": str(root),
            "head": _git_observation("rev-parse", "HEAD"),
            "tree": _git_observation("rev-parse", "HEAD^{tree}"),
            "clean": status == "",
        },
        "interpreter": {
            "path": str(executable),
            "target": str(target),
            "sha256": g4.sha256_file(target),
            "python_version": platform.python_version(),
        },
        "environment": {
            "manager": manager,
            "name": environment_name,
            "pixi_version": pixi_version,
            "pyproject_sha256": g4.sha256_file(root / "pyproject.toml"),
            "pixi_lock_sha256": g4.sha256_file(root / "pixi.lock"),
        },
        "jax": {
            "python_version": platform.python_version(),
            "jax_version": _distribution_version("jax", required=True),
            "jaxlib_version": _distribution_version("jaxlib", required=True),
            "numpy_version": _distribution_version("numpy", required=True),
            "jax_cuda12_plugin_version": _distribution_version(
                "jax-cuda12-plugin", required=plugin_required
            ),
            "jax_cuda12_pjrt_version": _distribution_version(
                "jax-cuda12-pjrt", required=plugin_required
            ),
            "float_dtype": "float32",
            "x64_enabled": bool(g4.jax.config.x64_enabled),
        },
        "device": {
            "observed_backend": actual_backend,
            "device_count": len(device_records),
            "devices": device_records,
        },
        "cache_roots": {
            "xdg": _required_absolute_environment_path("XDG_CACHE_HOME"),
            "tmp": _required_absolute_environment_path("TMPDIR"),
            "jax": _required_absolute_environment_path("JAX_COMPILATION_CACHE_DIR"),
            "cuda": _required_absolute_environment_path("CUDA_CACHE_PATH"),
        },
        "environment_flags": flags,
    }


def _create_output(path: Path) -> Path:
    output = g4.validate_task_output_path(path)
    output.mkdir(mode=0o700)
    if stat.S_IMODE(output.stat().st_mode) != 0o700:
        raise G4CliError("output directory mode is not 0700")
    return output


def _result_envelope(
    mode: str, result: dict[str, Any], preflight: dict[str, Any]
) -> dict[str, Any]:
    return {
        "schema_version": g4.SCHEMA_VERSION,
        "mode": mode,
        "status": result["status"],
        "claim_boundary": "M0-M6 technical simulation evidence only",
        "preflight": {
            "public_input_status": preflight["public_inputs"]["status"],
            "config_schema_version": preflight["config"]["schema_version"],
            "protected_contract": g4.protected_contract(),
        },
        "result": result,
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    """Execute one parsed G4 mode and write one canonical result."""
    if args.write_source_origin_record is not None:
        path = _validate_record_path(args.write_source_origin_record)
        payload = g4.write_source_origin_record(REPOSITORY_ROOT, path)
        return {
            "schema_version": g4.SOURCE_ORIGIN_SCHEMA_VERSION,
            "status": "PASS_SOURCE_ORIGIN_RECORD",
            "record": str(path),
            "record_sha256": payload["record_sha256"],
        }
    if args.write_backend_source_origin_record is not None:
        path = _validate_record_path(args.write_backend_source_origin_record)
        payload = g4.write_backend_source_origin_record(REPOSITORY_ROOT, args.input_manifest, path)
        return {
            "schema_version": g4.BACKEND_SOURCE_ORIGIN_SCHEMA_VERSION,
            "status": "PASS_BACKEND_SOURCE_ORIGIN_RECORD",
            "record": str(path),
            "record_sha256": payload["record_sha256"],
        }

    if (
        args.backend_parity
        or args.backend_throughput
        or args.backend_checkpoint_run
        or args.backend_checkpoint_resume is not None
    ):
        manifest, runtime, manifest_sha, runtime_sha = _backend_preflight(args)
        provenance = {"input_manifest_sha256": manifest_sha, "runtime_contract_sha256": runtime_sha}
        resource = {"before": g4.memory_snapshot(), "after": g4.memory_snapshot()}
        common = {
            "backend": args.backend,
            "provenance": provenance,
            "runtime": runtime,
            "resource": resource,
            "repository_root": REPOSITORY_ROOT,
            "manifest": manifest,
        }
        if args.backend_parity:
            result = g4.run_backend_parity(**common)
            g4.write_result(args.output, result)
            return result
        if args.backend_throughput:
            result = g4.run_backend_throughput(
                **common, phase=args.phase, sample_index=args.sample_index
            )
            g4.write_result(args.output, result)
            return result
        result = g4.run_backend_checkpoint_segment(
            **common,
            config_sha256=manifest["immutable_product"][0]["sha256"],
            output_dir=args.output_dir,
            max_updates_this_process=args.max_updates_this_process,
            resume_checkpoint=args.backend_checkpoint_resume,
        )
        segment = result["m8_segment"]
        coordinates = (segment["actual_start_count"], segment["actual_end_count"])
        if coordinates in {(0, 20), (10, 20)}:
            mode = "continuous" if coordinates == (0, 20) else "resume"
            payload = g4.backend_resume_payload_from_m8_segment(
                mode=mode,
                backend=args.backend,
                provenance=provenance,
                runtime=runtime,
                resource=resource,
                segment=segment,
            )
            record = args.output_dir / f"backend-resume-{mode}-v1.json"
            g4.write_result(record, payload)
        return result

    preflight, config_sha256 = _common_preflight(args)
    if args.resume_worker_checkpoint is not None:
        output_path = _validate_worker_output(args.resume_worker_output)
        return g4.resume_worker(
            REPOSITORY_ROOT, config_sha256, args.resume_worker_checkpoint, output_path
        )

    if args.longrun or args.resume_checkpoint is not None:
        return g4.run_bounded_longrun_segment(
            REPOSITORY_ROOT,
            config_sha256,
            args.output_dir,
            args.max_updates_this_process,
            resume_checkpoint=args.resume_checkpoint,
        )
    if args.m8_longrun or args.m8_resume_checkpoint is not None:
        return g4.run_bounded_m8_segment(
            REPOSITORY_ROOT,
            config_sha256,
            args.output_dir,
            args.max_updates_this_process,
            resume_checkpoint=args.m8_resume_checkpoint,
        )

    output = _create_output(args.output_dir)
    if args.dry_run:
        mode = "dry_run"
        result = g4.dry_run_contract(REPOSITORY_ROOT, args.config)
    elif args.theta_zero_parity:
        mode = "theta_zero_parity"
        result = g4.run_m2(REPOSITORY_ROOT)
    elif args.resource_rebenchmark_only:
        mode = "resource_rebenchmark_only"
        result = g4.run_m3()
    elif args.single_smokes:
        mode = "single_smokes"
        result = g4.run_single_smokes()
    elif args.joint_smoke:
        mode = "joint_smoke"
        result = g4.run_joint_smoke(REPOSITORY_ROOT, config_sha256, output)
    elif args.resume_reproduction:
        mode = "resume_reproduction"
        result = g4.run_resume_reproduction_parent(
            REPOSITORY_ROOT,
            args.config,
            config_sha256,
            args.source_origin_record,
            output,
            Path(__file__).resolve(strict=True),
        )
    else:
        raise G4CliError("unreachable G4 mode")
    envelope = _result_envelope(mode, result, preflight)
    g4.write_result(output / "result.json", envelope)
    return envelope


def main(argv: list[str] | tuple[str, ...] | None = None) -> None:
    """CLI entry point."""
    args = parse_args(argv)
    result = run(args)
    print(json.dumps({"mode": result.get("mode", "source_origin"), "status": result["status"]}))


if __name__ == "__main__":
    os.environ.setdefault("MPLBACKEND", "Agg")
    main()
