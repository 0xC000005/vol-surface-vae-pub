#!/usr/bin/env python
"""Headless-browser screenshot QA for the narrative Gradio demo."""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence
from urllib.error import URLError
from urllib.request import Request, urlopen

from PIL import Image, ImageStat


DEFAULT_OUTPUT_DIR = (
    "experiments/backfill/block_ar/nl_scenario_demo_outputs/"
    "prefix_latent_browser_qa_839a"
)
DEFAULT_VIEWPORTS = ("desktop:1440x1200", "mobile:390x900")


def _write_json(path: str | Path, payload: dict[str, Any]) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def wait_for_http(url: str, *, timeout_seconds: float = 20.0) -> dict[str, Any]:
    deadline = time.monotonic() + timeout_seconds
    last_error = ""
    while time.monotonic() < deadline:
        try:
            request = Request(url, headers={"User-Agent": "nl-prefix-browser-qa"})
            with urlopen(request, timeout=3) as response:
                return {
                    "status": "pass",
                    "http_status": int(response.status),
                    "content_type": response.headers.get("content-type", ""),
                }
        except URLError as error:
            last_error = str(error)
            time.sleep(0.5)
    return {"status": "fail", "error": last_error}


def parse_viewport(value: str) -> tuple[str, int, int]:
    label, _, size = value.partition(":")
    width_text, _, height_text = size.partition("x")
    if not label or not width_text or not height_text:
        raise ValueError(f"invalid viewport spec: {value!r}")
    return label, int(width_text), int(height_text)


def find_chrome(candidates: Sequence[str] | None = None) -> str:
    for name in candidates or ("google-chrome", "chromium", "chromium-browser"):
        path = shutil.which(name)
        if path:
            return path
    return ""


def capture_screenshot(
    *,
    chrome_path: str,
    url: str,
    output_path: Path,
    width: int,
    height: int,
    virtual_time_budget_ms: int,
) -> subprocess.CompletedProcess[str]:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    return subprocess.run(
        [
            chrome_path,
            "--headless=new",
            "--disable-gpu",
            "--no-sandbox",
            f"--window-size={width},{height}",
            f"--virtual-time-budget={virtual_time_budget_ms}",
            f"--screenshot={output_path}",
            url,
        ],
        check=False,
        capture_output=True,
        text=True,
    )


def analyze_png(path: str | Path) -> dict[str, Any]:
    image_path = Path(path)
    if not image_path.is_file():
        return {"status": "fail", "error": "missing_screenshot"}
    with Image.open(image_path) as image:
        rgb = image.convert("RGB")
        stat = ImageStat.Stat(rgb)
        extrema = rgb.getextrema()
        width, height = rgb.size
        channel_std = [float(value) for value in stat.stddev]
        channel_mean = [float(value) for value in stat.mean]
        dynamic_ranges = [int(high - low) for low, high in extrema]
    nonblank = max(channel_std) > 1.0 and max(dynamic_ranges) > 8
    return {
        "status": "pass" if nonblank else "fail",
        "path": str(image_path),
        "bytes": int(image_path.stat().st_size),
        "width": int(width),
        "height": int(height),
        "channel_mean": channel_mean,
        "channel_stddev": channel_std,
        "dynamic_ranges": dynamic_ranges,
    }


def render_markdown(report: dict[str, Any]) -> str:
    lines = [
        "# Narrative Demo Browser QA",
        "",
        f"Status: `{report['status']}`",
        "",
        "## HTTP",
        "",
        f"- URL: `{report['url']}`",
        f"- Status: `{report['http']['status']}`",
        f"- HTTP status: `{report['http'].get('http_status', '')}`",
        "",
        "## Screenshots",
        "",
        "| Viewport | Status | Size | Bytes | Path |",
        "| --- | --- | --- | ---: | --- |",
    ]
    for row in report["screenshots"]:
        lines.append(
            f"| {row['label']} | `{row['status']}` | "
            f"{row.get('width', 0)}x{row.get('height', 0)} | "
            f"{row.get('bytes', 0)} | `{row.get('path', '')}` |"
        )
    lines.extend(
        [
            "",
            "## Scope",
            "",
            (
                "This is a browser-render smoke. It verifies that the Gradio "
                "page loads and produces nonblank screenshots at selected "
                "viewports. API smoke remains responsible for exercising the "
                "scenario run, factor fan redraw, and IV-cell redraw paths."
            ),
        ]
    )
    return "\n".join(lines)


def run_browser_qa(args: argparse.Namespace) -> dict[str, Any]:
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    chrome = str(args.chrome_path or find_chrome())
    http = wait_for_http(str(args.url), timeout_seconds=float(args.timeout_seconds))
    screenshots = []
    if chrome and http["status"] == "pass":
        for spec in args.viewport:
            label, width, height = parse_viewport(str(spec))
            screenshot_path = output_dir / f"{label}.png"
            proc = capture_screenshot(
                chrome_path=chrome,
                url=str(args.url),
                output_path=screenshot_path,
                width=width,
                height=height,
                virtual_time_budget_ms=int(args.virtual_time_budget_ms),
            )
            analysis = analyze_png(screenshot_path)
            analysis.update(
                {
                    "label": label,
                    "chrome_returncode": int(proc.returncode),
                    "stderr_tail": proc.stderr[-1000:],
                }
            )
            if proc.returncode != 0:
                analysis["status"] = "fail"
            screenshots.append(analysis)
    report = {
        "status": (
            "pass"
            if chrome
            and http["status"] == "pass"
            and screenshots
            and all(row.get("status") == "pass" for row in screenshots)
            else "fail"
        ),
        "created_at": datetime.now(timezone.utc).isoformat(),
        "url": str(args.url),
        "chrome_path": chrome,
        "http": http,
        "screenshots": screenshots,
        "artifact_paths": {
            "json": str(output_dir / "browser_qa_report.json"),
            "markdown": str(output_dir / "browser_qa_report.md"),
        },
    }
    if not chrome:
        report["error"] = "chrome_not_found"
    _write_json(report["artifact_paths"]["json"], report)
    Path(report["artifact_paths"]["markdown"]).write_text(
        render_markdown(report).rstrip() + "\n",
        encoding="utf-8",
    )
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://127.0.0.1:7860")
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--chrome-path", default="")
    parser.add_argument("--viewport", action="append", default=None)
    parser.add_argument("--timeout-seconds", type=float, default=20.0)
    parser.add_argument("--virtual-time-budget-ms", type=int, default=10000)
    args = parser.parse_args()
    if args.viewport is None:
        args.viewport = list(DEFAULT_VIEWPORTS)
    report = run_browser_qa(args)
    print(
        json.dumps(
            {
                "status": report["status"],
                "screenshot_count": len(report["screenshots"]),
                "json": report["artifact_paths"]["json"],
                "markdown": report["artifact_paths"]["markdown"],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
