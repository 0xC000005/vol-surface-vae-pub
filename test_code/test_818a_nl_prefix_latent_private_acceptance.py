import json
import sys
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, ".")

from experiments.backfill.block_ar import (
    nl_prefix_latent_private_prototype_acceptance as acceptance,
)


class FakeProcess:
    def __init__(self) -> None:
        self.terminated = False
        self.killed = False

    def poll(self):
        return None if not self.terminated else 0

    def terminate(self) -> None:
        self.terminated = True

    def wait(self, timeout=None):
        return 0

    def kill(self) -> None:
        self.killed = True
        self.terminated = True


def test_run_acceptance_orchestrates_smoke_browser_registry_and_store(
    monkeypatch, tmp_path: Path
) -> None:
    calls = []
    fake_proc = FakeProcess()

    monkeypatch.setattr(
        acceptance,
        "launch_app",
        lambda **kwargs: calls.append(("launch", kwargs)) or fake_proc,
    )
    monkeypatch.setattr(acceptance, "wait_for_http", lambda url, timeout_seconds: True)

    def fake_smoke(args: SimpleNamespace) -> dict:
        calls.append(("smoke", args.output_dir))
        return {
            "status": "ok",
            "prefix_run_record_path": str(tmp_path / "run_record.json"),
        }

    def fake_browser(args: SimpleNamespace) -> dict:
        calls.append(("browser", args.output_dir))
        return {
            "status": "pass",
            "artifact_paths": {"json": str(tmp_path / "browser.json")},
        }

    def fake_registry(args: SimpleNamespace) -> dict:
        calls.append(("registry", args.output_dir))
        return {
            "status": "pass",
            "artifact_paths": {"json": str(tmp_path / "registry.json")},
        }

    def fake_store(args: SimpleNamespace) -> dict:
        calls.append(("store", args.sqlite, args.run_record))
        return {"run_record_count": 1}

    monkeypatch.setattr(acceptance, "run_gradio_api_smoke", fake_smoke)
    monkeypatch.setattr(acceptance, "run_browser_qa", fake_browser)
    monkeypatch.setattr(acceptance, "build_registry", fake_registry)
    monkeypatch.setattr(acceptance, "build_store", fake_store)

    summary = acceptance.run_acceptance(
        SimpleNamespace(
            output_dir=str(tmp_path / "acceptance"),
            server_name="127.0.0.1",
            port=7867,
            timeout_seconds=1.0,
            casebook_choice="safe_haven_gold_bid:18",
            expected_start_index=18,
            samples=2,
            fan_market="SPX",
            redraw_market="IV_ATM_3M",
            auth_user_env="USER",
            auth_password_env="PASSWORD",
            require_auth=False,
        )
    )

    assert summary["status"] == "pass"
    assert fake_proc.terminated is True
    assert [call[0] for call in calls] == [
        "launch",
        "smoke",
        "browser",
        "registry",
        "store",
    ]
    saved = json.loads(
        (tmp_path / "acceptance" / "private_acceptance_summary.json").read_text()
    )
    assert saved["status"] == "pass"


def test_status_from_steps_fails_on_failed_step() -> None:
    assert acceptance._status_from_steps(
        [{"status": "pass"}, {"status": "fail"}]
    ) == "fail"
