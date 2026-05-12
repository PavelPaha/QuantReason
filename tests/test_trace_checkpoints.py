from __future__ import annotations

import tempfile
from pathlib import Path

from quantlab.artifacts.store import ArtifactStore
from quantlab.core.trace import Trace, TraceSegment
from quantlab.evaluation.judge import JudgementResult


def test_staged_wave_checkpoint_roundtrip():
    t1 = Trace(example_id="ex1", prompt="P1\n")
    t1.append_segment(
        TraceSegment(actor_id="a", text="plan", token_count=1, start_token_idx=0),
    )
    t2 = Trace(example_id="ex2", prompt="P2\n")
    t2.append_segment(
        TraceSegment(actor_id="a", text="other", token_count=1, start_token_idx=0),
    )

    with tempfile.TemporaryDirectory() as td:
        base = Path(td)
        store = ArtifactStore(base_dir=str(base))
        run_id = store.new_run("test", {})
        store.save_staged_wave_checkpoint(run_id, 0, {"ex2": t2, "ex1": t1})

        path = base / run_id / "trace_checkpoints" / "wave_0.jsonl"
        assert path.exists()
        lines = path.read_text().strip().split("\n")
        assert len(lines) == 2

        loaded = store.load_staged_wave_checkpoint(run_id, 0)
        assert [t.example_id for t in loaded] == ["ex1", "ex2"]
        assert loaded[0].segments[0].text == "plan"
        assert loaded[1].segments[0].text == "other"


def test_staged_wave_checkpoint_empty_skips_file():
    with tempfile.TemporaryDirectory() as td:
        store = ArtifactStore(base_dir=str(td))
        run_id = store.new_run("test", {})
        store.save_staged_wave_checkpoint(run_id, 0, {})
        assert not (Path(td) / run_id / "trace_checkpoints").exists()


def test_list_judged_and_load_judgements():
    with tempfile.TemporaryDirectory() as td:
        store = ArtifactStore(base_dir=str(td))
        run_id = store.new_run("x", {})

        assert store.list_judged_example_ids(run_id) == set()
        assert store.load_judgements(run_id) == []

        store.save_judgement(
            run_id,
            JudgementResult(
                example_id="a",
                predicted="1",
                ground_truth="1",
                is_correct=True,
                parse_success=True,
            ),
        )
        store.save_judgement(
            run_id,
            JudgementResult(
                example_id="b",
                predicted=None,
                ground_truth="2",
                is_correct=False,
                parse_success=False,
            ),
        )

        assert store.list_judged_example_ids(run_id) == {"a", "b"}
        jj = store.load_judgements(run_id)
        assert len(jj) == 2
        assert {j["example_id"] for j in jj} == {"a", "b"}
