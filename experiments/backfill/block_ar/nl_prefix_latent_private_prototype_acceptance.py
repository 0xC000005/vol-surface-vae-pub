#!/usr/bin/env python
"""Run local private-prototype acceptance for the narrative demo."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Callable

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.backfill.block_ar.nl_prefix_latent_browser_qa import run_browser_qa
from experiments.backfill.block_ar.nl_prefix_latent_demo_run_registry import (
    DEFAULT_AUDIT_MANIFEST,
    DEFAULT_QA_PACKET,
    build_registry,
)
from experiments.backfill.block_ar.nl_prefix_latent_gradio_api_smoke import (
    run_gradio_api_smoke,
)
from experiments.backfill.block_ar.nl_prefix_latent_run_store import build_store


DEFAULT_OUTPUT_DIR = (
    "experiments/backfill/block_ar/nl_scenario_demo_outputs/"
    "prefix_latent_private_acceptance_844a"
)
APP_SCRIPT = "experiments/backfill/block_ar/nl_risk_manager_story_gradio_app.py"


def _write_json(path: str | Path, payload: dict[str, Any]) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def wait_for_http(url: str, *, timeout_seconds: float = 30.0) -> bool:
    from urllib.request import urlopen

    deadline = time.monotonic() + timeout_seconds
    while time.monotonic() < deadline:
        try:
            with urlopen(url, timeout=2) as response:
                return int(response.status) == 200
        except Exception:
            time.sleep(0.5)
    return False


def launch_app(
    *,
    port: int,
    server_name: str,
    require_auth: bool,
    popen: Callable[..., subprocess.Popen] = subprocess.Popen,
) -> subprocess.Popen:
    cmd = [
        sys.executable,
        APP_SCRIPT,
        "--server-name",
        str(server_name),
        "--server-port",
        str(port),
    ]
    if require_auth:
        cmd.append("--require-auth")
    return popen(
        cmd,
        cwd=".",
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )


def stop_process(proc: subprocess.Popen) -> None:
    if proc.poll() is not None:
        return
    proc.terminate()
    try:
        proc.wait(timeout=10)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait(timeout=10)


def _status_from_steps(steps: list[dict[str, Any]]) -> str:
    return "pass" if steps and all(row.get("status") in {"pass", "ok"} for row in steps) else "fail"


def run_acceptance(args: argparse.Namespace) -> dict[str, Any]:
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    url = f"http://{args.server_name}:{int(args.port)}"
    proc = launch_app(
        port=int(args.port),
        server_name=str(args.server_name),
        require_auth=bool(args.require_auth),
    )
    steps: list[dict[str, Any]] = []
    try:
        http_ready = wait_for_http(url, timeout_seconds=float(args.timeout_seconds))
        steps.append({"name": "app_http_ready", "status": "pass" if http_ready else "fail"})
        if not http_ready:
            raise RuntimeError(f"app did not become ready at {url}")

        smoke = run_gradio_api_smoke(
            SimpleNamespace(
                url=url,
                output_dir=str(output_dir / "cached_smoke"),
                mode="cached_casebook",
                casebook_choice=str(args.casebook_choice),
                story="",
                expected_start_index=int(args.expected_start_index),
                samples=int(args.samples),
                fan_market=str(args.fan_market),
                redraw_market=str(args.redraw_market),
                auth_user_env=str(args.auth_user_env),
                auth_password_env=str(args.auth_password_env),
                require_auth=bool(args.require_auth),
            )
        )
        steps.append({"name": "cached_smoke", "status": smoke["status"]})

        browser = run_browser_qa(
            SimpleNamespace(
                url=url,
                output_dir=str(output_dir / "browser_qa"),
                chrome_path="",
                viewport=["desktop:1440x1200", "mobile:390x900"],
                timeout_seconds=float(args.timeout_seconds),
                virtual_time_budget_ms=10000,
            )
        )
        steps.append({"name": "browser_qa", "status": browser["status"]})

        registry = build_registry(
            SimpleNamespace(
                repo_root=".",
                output_dir=str(output_dir / "run_registry"),
                evidence=[
                    f"qa_packet:qa_packet:{DEFAULT_QA_PACKET}",
                    f"audit_manifest:audit_manifest:{DEFAULT_AUDIT_MANIFEST}",
                    f"browser_qa:browser_qa:{browser['artifact_paths']['json']}",
                    f"prefix_run_record:prefix_run_record:{smoke['prefix_run_record_path']}",
                ],
            )
        )
        steps.append({"name": "run_registry", "status": registry["status"]})

        store = build_store(
            SimpleNamespace(
                sqlite=str(output_dir / "run_store" / "demo_run_store.sqlite"),
                summary_json=str(output_dir / "run_store" / "demo_run_store_summary.json"),
                registry=[registry["artifact_paths"]["json"]],
                run_record=[smoke["prefix_run_record_path"]],
            )
        )
        store_status = "pass" if store["run_record_count"] >= 1 else "fail"
        steps.append({"name": "run_store", "status": store_status})
    finally:
        stop_process(proc)

    summary = {
        "status": _status_from_steps(steps),
        "url": url,
        "steps": steps,
        "artifact_paths": {
            "summary": str(output_dir / "private_acceptance_summary.json"),
            "cached_smoke": str(output_dir / "cached_smoke" / "gradio_api_smoke_summary.json"),
            "browser_qa": str(output_dir / "browser_qa" / "browser_qa_report.json"),
            "run_registry": str(output_dir / "run_registry" / "demo_run_registry.json"),
            "run_store": str(output_dir / "run_store" / "demo_run_store_summary.json"),
        },
    }
    _write_json(summary["artifact_paths"]["summary"], summary)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--server-name", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=7867)
    parser.add_argument("--timeout-seconds", type=float, default=30.0)
    parser.add_argument("--casebook-choice", default="safe_haven_gold_bid:18")
    parser.add_argument("--expected-start-index", type=int, default=18)
    parser.add_argument("--samples", type=int, default=2)
    parser.add_argument("--fan-market", default="SPX")
    parser.add_argument("--redraw-market", default="IV_ATM_3M")
    parser.add_argument("--auth-user-env", default="NARRATIVE_DEMO_AUTH_USER")
    parser.add_argument("--auth-password-env", default="NARRATIVE_DEMO_AUTH_PASSWORD")
    parser.add_argument("--require-auth", action="store_true")
    args = parser.parse_args()
    summary = run_acceptance(args)
    print(
        json.dumps(
            {
                "status": summary["status"],
                "summary": summary["artifact_paths"]["summary"],
                "steps": summary["steps"],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
