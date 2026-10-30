# -*- coding: utf-8 -*-
"""End-to-end test of the Streamlit app on a saved trial project: open the saved project and press every step's button.
Skipped when the trial files or the saved test project are not on this computer."""
import json
import os
import shutil
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROJ = os.path.join(ROOT, "workspace", "test_trial29")


def _inputs_present():
    """The saved project AND the trial files it points at (they live outside the app folder)."""
    pj = os.path.join(PROJ, "project.json")
    if not os.path.exists(pj):
        return False
    with open(pj, encoding="utf-8") as fh:
        f = json.load(fh)["config"]["files"]
    paths = ([f.get("design"), f.get("yield"), f.get("boundary")] + list((f.get("asapplied") or {}).values())
             + list((f.get("extra_designs") or {}).values()))
    return all(os.path.exists(p) for p in paths if p)


pytestmark = pytest.mark.skipif(not _inputs_present(), reason="Trial 29 test project or its input files are not on this computer")


def _button(at, label):
    for b in at.button:
        if b.label == label:
            return b
    raise AssertionError(f"no button {label!r}: {[b.label for b in at.button]}")


def test_full_flow(tmp_path, monkeypatch):
    ws = tmp_path / "ws"
    shutil.copytree(PROJ, ws / "20260101-000000_trial29", ignore=shutil.ignore_patterns("output"))
    monkeypatch.setenv("OCVR_WORKSPACE", str(ws))
    from streamlit.testing.v1 import AppTest
    at = AppTest.from_file(os.path.join(ROOT, "app", "app.py"), default_timeout=900)
    at.run()
    assert not at.exception
    _button(at, "Open").click().run()
    assert not at.exception
    # small resampling so the test stays quick
    for ni in at.number_input:
        if ni.label == "Bootstrap":
            ni.set_value(10)
        if ni.label == "Cross-fit repeats":
            ni.set_value(5)
    _button(at, "Load the trial").click().run()
    assert not at.exception, at.exception
    _button(at, "Build zones").click().run()
    assert not at.exception, at.exception
    _button(at, "Join yield, plots and zones").click().run()
    assert not at.exception, at.exception
    for ni in at.number_input:
        if ni.label == "Bootstrap":
            ni.set_value(10)
        if ni.label == "Cross-fit repeats":
            ni.set_value(5)
    at.run()
    _button(at, "Run the economics").click().run()
    assert not at.exception, at.exception
    _button(at, "Build the PDF report").click().run()
    assert not at.exception, at.exception
    out = ws / "20260101-000000_trial29" / "output"
    pdfs = [f for f in os.listdir(out) if f.endswith(".pdf")]
    assert pdfs, os.listdir(out)
    assert (out / "zones_rx.shp").exists()
    texts = " ".join(str(x.value) for x in at.markdown) + " ".join(str(getattr(x, "value", "")) for x in at.info)
    assert "Flat rate" in texts or "Variable rate" in texts


def test_tested_inputs_selector(tmp_path, monkeypatch):
    """'What was tested?': Seed and urea shows both ready rows; the urea needs its own rate column before the trial
    loads (the seed column is never taken for it); Urea alone does not take the seed design column either; back to
    Seed the project is as before."""
    ws = tmp_path / "ws"
    shutil.copytree(PROJ, ws / "20260101-000000_trial29", ignore=shutil.ignore_patterns("output"))
    monkeypatch.setenv("OCVR_WORKSPACE", str(ws))
    from streamlit.testing.v1 import AppTest
    at = AppTest.from_file(os.path.join(ROOT, "app", "app.py"), default_timeout=900)
    at.run()
    _button(at, "Open").click().run()
    radio = lambda: [r for r in at.radio if r.label == "What was tested?"][0]
    rate = lambda: [s.value for s in at.selectbox if s.label == "Rate"]
    assert radio().value == "Seed" and rate() == ["Tgt_Rate_l"]
    radio().set_value("Seed and urea").run()
    assert not at.exception, at.exception
    assert [m.value for m in at.markdown if m.value in ("**Seed**", "**Urea**")] == ["**Seed**", "**Urea**"]
    assert [s.value for s in at.selectbox if s.label.startswith("Rate column for urea")] == ["(choose)"]
    assert _button(at, "Load the trial").disabled
    radio().set_value("Urea").run()
    assert not at.exception, at.exception
    assert rate() == ["(choose)"] and _button(at, "Load the trial").disabled
    radio().set_value("Seed").run()
    assert not at.exception, at.exception
    assert rate() == ["Tgt_Rate_l"] and not _button(at, "Load the trial").disabled
