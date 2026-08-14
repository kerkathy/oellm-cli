import csv
import os
import subprocess
from pathlib import Path
from unittest.mock import patch

import pandas as pd

from oellm.main import collect_results, schedule_evals
from oellm.task_groups import _collect_dataset_specs, _expand_task_groups


def _fake_venv(tmp_path: Path) -> Path:
    venv = tmp_path / "inspect-venv"
    bin_dir = venv / "bin"
    bin_dir.mkdir(parents=True)
    python = bin_dir / "python"
    python.write_text(
        "#!/bin/bash\n"
        'printf "%s\n" "$@" > "$INSPECT_ARGS_FILE"\n'
        'exit "$FAKE_INSPECT_EXIT"\n'
    )
    python.chmod(0o755)
    (bin_dir / "activate").write_text(f'export PATH="{bin_dir}:$PATH"\n')
    return venv


def _run_generated_script(
    tmp_path: Path,
    model: str,
    *,
    suite: str = "inspect",
    limit: int = 10,
    inspect_exit: int = 0,
) -> tuple[subprocess.CompletedProcess, list[str]]:
    venv = _fake_venv(tmp_path)
    csv_path = tmp_path / "evals.csv"
    with csv_path.open("w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["model_path", "task_path", "n_shot", "eval_suite"])
        writer.writerow([model, "inspect_evals/custom_task", 0, suite])

    output_dir = tmp_path / "output"
    args_file = tmp_path / "inspect-args.txt"
    with patch.dict(
        os.environ,
        {
            "EVAL_OUTPUT_DIR": str(output_dir),
            "INSPECT_ARGS_FILE": str(args_file),
            "FAKE_INSPECT_EXIT": str(inspect_exit),
        },
        clear=True,
    ):
        schedule_evals(
            eval_csv_path=str(csv_path),
            skip_checks=True,
            dry_run=True,
            local=True,
            venv_path=str(venv),
            limit=limit,
        )
        sbatch_path = next(output_dir.glob("*/submit_evals.sbatch"))
        run_env = {
            **os.environ,
            "SLURM_ARRAY_TASK_ID": "0",
            "SLURM_ARRAY_JOB_ID": "1",
            "SLURM_JOB_ID": "1",
        }
        result = subprocess.run(
            ["bash", str(sbatch_path)],
            capture_output=True,
            text=True,
            env=run_env,
        )

    args = args_file.read_text().splitlines() if args_file.exists() else []
    return result, args


def test_bbq_task_group_metadata():
    results = _expand_task_groups(["safety"])
    assert [(result.task, result.n_shot, result.suite) for result in results] == [
        ("bbq", 0, "inspect")
    ]
    specs = _collect_dataset_specs(["safety"])
    assert [(spec.repo_id, spec.subset) for spec in specs] == [("heegyu/bbq", None)]


def test_inspect_dispatch_maps_hugging_face_model_and_limit(tmp_path):
    result, args = _run_generated_script(tmp_path, "HuggingFaceTB/SmolLM2-135M-Instruct")

    assert result.returncode == 0, result.stderr
    assert args[:4] == ["-m", "inspect_ai", "eval", "inspect_evals/custom_task"]
    assert args[args.index("--model") + 1] == "hf/HuggingFaceTB/SmolLM2-135M-Instruct"
    assert args[args.index("--limit") + 1] == "10"
    assert "--log-dir" in args


def test_inspect_dispatch_maps_local_model(tmp_path):
    model_dir = tmp_path / "checkpoint"
    model_dir.mkdir()
    (model_dir / "model.safetensors").touch()

    result, args = _run_generated_script(tmp_path, str(model_dir))

    assert result.returncode == 0, result.stderr
    assert args[args.index("--model") + 1] == "hf/local"
    assert args[args.index("-M") + 1] == f"model_path={model_dir}"


def test_inspect_command_failure_propagates(tmp_path):
    result, _ = _run_generated_script(
        tmp_path,
        "HuggingFaceTB/SmolLM2-135M-Instruct",
        inspect_exit=7,
    )
    assert result.returncode != 0


def test_unknown_suite_fails(tmp_path):
    result, _ = _run_generated_script(
        tmp_path,
        "HuggingFaceTB/SmolLM2-135M-Instruct",
        suite="unknown-suite",
    )
    assert result.returncode != 0
    assert "Unknown evaluation suite" in result.stdout


def _inspect_header(status: str = "success") -> dict:
    return {
        "status": status,
        "eval": {
            "model": "hf/HuggingFaceTB/SmolLM2-135M-Instruct",
            "model_args": {},
            "task": "bbq",
            "task_registry_name": "inspect_evals/bbq",
        },
        "results": {
            "scores": [
                {
                    "metrics": {
                        "accuracy": {"name": "accuracy", "value": 0.75},
                        "stderr": {"name": "stderr", "value": 0.01},
                    }
                }
            ]
        },
    }


def test_eval_result_discovery_recognizes_success(tmp_path):
    pd.DataFrame(
        [
            {
                "model_path": "HuggingFaceTB/SmolLM2-135M-Instruct",
                "task_path": "bbq",
                "n_shot": 0,
                "eval_suite": "inspect",
            }
        ]
    ).to_csv(tmp_path / "jobs.csv", index=False)
    eval_file = tmp_path / "results" / "bbq.eval"
    eval_file.parent.mkdir()
    eval_file.touch()
    output_csv = tmp_path / "results.csv"

    with patch(
        "oellm.main._read_inspect_eval_header",
        return_value=_inspect_header(),
    ):
        collect_results(str(tmp_path), output_csv=str(output_csv), check=True)

    results = pd.read_csv(output_csv)
    assert results.to_dict("records") == [
        {
            "model_name": "HuggingFaceTB/SmolLM2-135M-Instruct",
            "task": "bbq",
            "n_shot": 0,
            "performance": 0.75,
            "metric_name": "accuracy",
        }
    ]
    assert not (tmp_path / "results_missing.csv").exists()


def test_eval_result_discovery_ignores_failed_log(tmp_path):
    pd.DataFrame(
        [
            {
                "model_path": "HuggingFaceTB/SmolLM2-135M-Instruct",
                "task_path": "bbq",
                "n_shot": 0,
                "eval_suite": "inspect",
            }
        ]
    ).to_csv(tmp_path / "jobs.csv", index=False)
    eval_file = tmp_path / "results" / "bbq.eval"
    eval_file.parent.mkdir()
    eval_file.touch()
    output_csv = tmp_path / "results.csv"

    with patch(
        "oellm.main._read_inspect_eval_header",
        return_value=_inspect_header(status="error"),
    ):
        collect_results(str(tmp_path), output_csv=str(output_csv), check=True)

    missing = pd.read_csv(tmp_path / "results_missing.csv")
    assert len(missing) == 1
