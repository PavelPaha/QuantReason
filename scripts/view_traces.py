#!/usr/bin/env python3
"""Pretty-print traces from a run (read-friendly alternative to traces.jsonl)."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import click


def _divider(ch: str = "─") -> str:
    return ch * 88


def _format_prompt(prompt: str, max_chars: int | None) -> str:
    if max_chars is not None and len(prompt) > max_chars:
        return prompt[:max_chars] + f"\n\n… [{len(prompt) - max_chars} chars truncated]"
    return prompt


def render_trace_human(d: dict, *, prompt_max_chars: int | None) -> str:
    lines: list[str] = []
    eid = d.get("example_id", "?")
    lines.append(_divider("═"))
    lines.append(f"Trace: {eid}")
    meta = d.get("metadata") or {}
    exec_state = meta.get("_executor") or meta.get("_executor_state")
    if exec_state:
        lines.append(f"Executor state: {exec_state}")
    if d.get("total_generated_tokens") is not None:
        lines.append(f"Total generated tokens (declared): {d['total_generated_tokens']}")

    lines.append("")
    lines.append(_divider("─"))
    lines.append("PROMPT (full_text prefix — user / system / scaffold)")
    lines.append(_divider("─"))
    lines.append(_format_prompt(d.get("prompt", ""), prompt_max_chars))
    lines.append("")

    for i, seg in enumerate(d.get("segments", [])):
        lines.append(_divider("·"))
        role = seg.get("role", "unknown")
        aid = seg.get("actor_id", "?")
        tc = seg.get("token_count", "?")
        t0 = seg.get("start_token_idx", "?")
        tm = ""
        timing = seg.get("timing") or {}
        ms = timing.get("total_ms")
        if ms is not None:
            tm = f" · timing ≈ {ms:.0f} ms"
        lines.append(f"Segment {i} · actor={aid} · role={role} · tokens={tc} · start_idx={t0}{tm}")
        lines.append(_divider("·"))
        lines.append(seg.get("text", ""))
        lines.append("")

    lines.append(_divider("═"))
    return "\n".join(lines)


def render_trace_markdown(d: dict, *, prompt_max_chars: int | None) -> str:
    eid = d.get("example_id", "?")
    parts = [f"## Trace `{eid}`\n"]

    md_prompt = _format_prompt(d.get("prompt", ""), prompt_max_chars)
    parts.append("### Prompt\n\n```text\n" + md_prompt + "\n```\n")

    for i, seg in enumerate(d.get("segments", [])):
        aid = seg.get("actor_id", "?")
        role = seg.get("role", "unknown")
        tc = seg.get("token_count", "?")
        parts.append(f"### Segment {i} · `{aid}` · {role} · tokens={tc}\n")
        parts.append("```text\n" + seg.get("text", "") + "\n```\n")

    return "\n".join(parts)


@click.command(context_settings={"help_option_names": ["-h", "--help"]})
@click.argument(
    "path",
    type=click.Path(exists=True, file_okay=True, dir_okay=True, path_type=Path),
)
@click.option(
    "--example-id",
    "-e",
    multiple=True,
    help="Show only traces with this example_id (repeatable).",
)
@click.option(
    "--prompt-max-chars",
    type=int,
    default=None,
    help="Truncate prompt in display after N characters (full prompt still on disk).",
)
@click.option(
    "--markdown",
    "-m",
    "as_markdown",
    is_flag=True,
    help="Markdown with fenced blocks (good for copying to notes / viewers).",
)
@click.option(
    "--output",
    "-o",
    type=click.Path(path_type=Path),
    help="Write to file instead of stdout (UTF-8).",
)
def main(
    path: Path,
    example_id: tuple[str, ...],
    prompt_max_chars: int | None,
    as_markdown: bool,
    output: Path | None,
) -> None:
    """PATH is a run folder (expects traces.jsonl) or a traces.jsonl file."""

    traces_path = path / "traces.jsonl" if path.is_dir() else path
    if not traces_path.is_file():
        raise click.ClickException(f"Not found: {traces_path}")

    wanted = set(example_id) if example_id else None
    blocks: list[str] = []

    with traces_path.open(encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                d = json.loads(line)
            except json.JSONDecodeError as e:
                raise click.ClickException(f"Line {line_no}: invalid JSON: {e}") from e
            eid = d.get("example_id", "")
            if wanted is not None and eid not in wanted:
                continue
            if as_markdown:
                blocks.append(render_trace_markdown(d, prompt_max_chars=prompt_max_chars))
            else:
                blocks.append(render_trace_human(d, prompt_max_chars=prompt_max_chars))

    if not blocks:
        raise click.ClickException("No traces matched (check --example-id or file contents).")

    text = "\n\n".join(blocks)
    if output is None:
        sys.stdout.write(text)
        if not text.endswith("\n"):
            sys.stdout.write("\n")
    else:
        output.write_text(text + ("\n" if not text.endswith("\n") else ""), encoding="utf-8")


if __name__ == "__main__":
    main()