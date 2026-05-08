import json
import sys
from pathlib import Path

from PIL import Image

sys.path.insert(0, ".")

from experiments.backfill.block_ar.nl_prefix_latent_browser_qa import (
    analyze_png,
    parse_viewport,
    render_markdown,
    run_browser_qa,
)


def test_parse_viewport_splits_label_and_size() -> None:
    assert parse_viewport("desktop:1440x1200") == ("desktop", 1440, 1200)


def test_parse_viewport_rejects_bad_spec() -> None:
    try:
        parse_viewport("bad")
    except ValueError as error:
        assert "invalid viewport" in str(error)
    else:  # pragma: no cover - defensive failure branch
        raise AssertionError("expected invalid viewport to fail")


def test_analyze_png_flags_blank_and_nonblank_images(tmp_path: Path) -> None:
    blank = tmp_path / "blank.png"
    nonblank = tmp_path / "nonblank.png"
    Image.new("RGB", (20, 20), "white").save(blank)
    image = Image.new("RGB", (20, 20), "white")
    for x in range(10):
        for y in range(20):
            image.putpixel((x, y), (0, 0, 0))
    image.save(nonblank)

    assert analyze_png(blank)["status"] == "fail"
    assert analyze_png(nonblank)["status"] == "pass"


def test_render_markdown_lists_screenshots() -> None:
    report = {
        "status": "pass",
        "url": "http://demo",
        "http": {"status": "pass", "http_status": 200},
        "screenshots": [
            {
                "label": "desktop",
                "status": "pass",
                "width": 1440,
                "height": 1200,
                "bytes": 123,
                "path": "desktop.png",
            }
        ],
    }

    markdown = render_markdown(report)

    assert "Narrative Demo Browser QA" in markdown
    assert "desktop.png" in markdown
    assert "API smoke remains responsible" in markdown


def test_run_browser_qa_fails_without_chrome(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr(
        "experiments.backfill.block_ar.nl_prefix_latent_browser_qa.find_chrome",
        lambda: "",
    )
    monkeypatch.setattr(
        "experiments.backfill.block_ar.nl_prefix_latent_browser_qa.wait_for_http",
        lambda url, timeout_seconds: {"status": "pass", "http_status": 200},
    )

    class Args:
        url = "http://demo"
        output_dir = str(tmp_path)
        chrome_path = ""
        viewport = ["desktop:1440x1200"]
        timeout_seconds = 1.0
        virtual_time_budget_ms = 100

    report = run_browser_qa(Args())

    assert report["status"] == "fail"
    assert report["error"] == "chrome_not_found"
    saved = json.loads((tmp_path / "browser_qa_report.json").read_text())
    assert saved["status"] == "fail"
