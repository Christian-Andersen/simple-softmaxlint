from __future__ import annotations

import math
import re
import sys
from itertools import pairwise
from pathlib import Path
from typing import Any, NamedTuple

from openai import OpenAI, OpenAIError

DEFAULT_URL = "http://127.0.0.1:18020/v1"
TOP_LOGPROBS = 20  # candidates at the hole; vLLM's server-side ceiling is 20
BATCH_SIZE = 256  # prompts per request, so a big file cannot produce an unstreamable payload
TOP_REPORTED = 10  # how many findings to print; the true count is always shown
WARN_RATIO, CRIT_RATIO, MIN_TOP_PROB = 0.30, 0.10, 0.5  # MIN_TOP_PROB: below this the distribution is flat

#: Model control traffic, skipped when picking the top candidate because past ~1000 chars of
#: FIM context the model falls back to ```` at 0.6-0.9. Matching still uses the full list.
CONTROL_TOKENS = frozenset(
    {"", "<|fim_prefix|>", "<|fim_suffix|>", "<|fim_middle|>", "<|fim_pad|>", "<|file_sep|>", "<|repo_name|>"}
) | frozenset({"<PRE>", "<SUF>", "<MID>", "<EOT>"})
CONTROL_TOKEN_RE = re.compile(r"^<\|.*\|>$")  # pipe-wrapped fallback for other model families


def _is_control(token: str) -> bool:
    return token in CONTROL_TOKENS or CONTROL_TOKEN_RE.match(token) is not None


class SoftmaxLintError(Exception):
    pass


class FimMarkers(NamedTuple):
    prefix: str = "<|fim_prefix|>"
    suffix: str = "<|fim_suffix|>"
    middle: str = "<|fim_middle|>"


class Slice(NamedTuple):
    start: int
    end: int
    text: str


def token_boundaries(source: str, choice: dict[str, Any]) -> list[Slice]:
    logprobs: dict[str, Any] = choice.get("logprobs") or {}
    offsets: list[int] = [o for o in logprobs.get("text_offset") or [] if 0 <= o < len(source)]
    if not offsets:
        msg = f"server echoed no usable text_offset (snippet length={len(source)})"
        raise SoftmaxLintError(msg)
    if offsets[0] != 0 or offsets != sorted(offsets) or offsets[-1] >= len(source):
        msg = (
            f"server returned an unusable tokenization (first={offsets[0]!r}, last={offsets[-1]!r}, "
            f"length={len(source)}); probes must land on the model's own boundaries"
        )
        raise SoftmaxLintError(msg)
    return [Slice(a, b, source[a:b]) for a, b in pairwise([*offsets, len(source)]) if b > a]


def build_prompt(source: str, start: int, markers: FimMarkers) -> str:
    return f"{markers.prefix}{source[:start]}{markers.suffix}{source[start:]}{markers.middle}"


def _match_source(candidates: dict[str, float], slice_: Slice) -> tuple[str, float] | None:
    exact = candidates.get(slice_.text)
    if exact is not None:
        return slice_.text, exact
    matches = [(token, lp) for token, lp in candidates.items() if token and slice_.text.startswith(token)]
    return max(matches, key=lambda item: (len(item[0]), item[1])) if matches else None


def parse_choice(choice: dict[str, Any], slice_: Slice) -> dict[str, Any]:
    logprobs: dict[str, Any] = choice.get("logprobs") or {}
    candidates: dict[str, float] = (logprobs.get("top_logprobs") or [{}])[0] or {}
    if not candidates:
        msg = "server returned no candidate logprobs for the FIM middle token"
        raise SoftmaxLintError(msg)

    scored = {t: lp for t, lp in candidates.items() if not _is_control(t)} or candidates
    top_token, top_logprob = max(scored.items(), key=lambda item: item[1])
    top_prob = math.exp(top_logprob)

    match = _match_source(candidates, slice_)
    if match is None:
        actual_prob, ratio, surprisal, rank = 0.0, 0.0, -min(scored.values()), len(candidates) + 1
    else:
        actual_token, actual_logprob = match
        ordered = sorted(candidates.items(), key=lambda item: item[1], reverse=True)
        actual_prob = math.exp(actual_logprob)
        ratio = actual_prob / top_prob if top_prob > 0 else 1.0  # a zero-probability top pick
        surprisal, rank = -actual_logprob, ordered.index((actual_token, actual_logprob)) + 1

    return {
        "token": slice_.text,
        "start": slice_.start,
        "end": slice_.end,
        "actual_prob": actual_prob,
        "top_prob": top_prob,
        "top_token": top_token,
        "ratio": ratio,
        "surprisal": surprisal,
        "rank": rank,
    }


def request(client: OpenAI, model: str, prompt: str | list[str], logprobs: int, *, echo: bool = False) -> list[Any]:
    try:
        response = client.completions.create(
            model=model, prompt=prompt, max_tokens=1, temperature=0, logprobs=logprobs, echo=echo
        )
    except OpenAIError as exc:
        msg = f"could not reach vLLM at {client.base_url}: {exc}"
        raise SoftmaxLintError(msg) from exc
    if not (choices := response.model_dump().get("choices")):
        msg = "server returned no choices"
        raise SoftmaxLintError(msg)
    return choices


def score_source(client: OpenAI, model: str, source: str, markers: FimMarkers) -> list[dict[str, Any]]:
    if not source.strip():
        return []
    (echo,) = request(client, model, source, logprobs=0, echo=True)
    slices = token_boundaries(source, echo)
    runnable = [index for index, s in enumerate(slices) if s.text.strip()]
    prompts = [build_prompt(source, slices[i].start, markers) for i in runnable]

    scored: list[Any] = []
    for offset in range(0, len(prompts), BATCH_SIZE):
        batch = prompts[offset : offset + BATCH_SIZE]
        choices = request(client, model, batch, logprobs=TOP_LOGPROBS)
        if len(choices) != len(batch):
            msg = f"expected {len(batch)} scored prompts, got {len(choices)}"
            raise SoftmaxLintError(msg)
        scored.extend(sorted(choices, key=lambda choice: choice.get("index") or 0))
    return [parse_choice(choice, slices[i]) for i, choice in zip(runnable, scored, strict=True)]


def classify(analysis: dict[str, Any], crit: float, warn: float) -> str | None:
    top = analysis["top_token"].strip()
    if analysis["top_prob"] < MIN_TOP_PROB or not top or top in ('"""', "'''"):
        return None
    if analysis["ratio"] < crit:
        return "CRITICAL"
    return "WARNING" if analysis["ratio"] < warn else None


def discover_model(client: OpenAI) -> str:
    try:
        names = sorted({entry.id for entry in client.models.list().data if entry.id})
    except OpenAIError as exc:
        msg = f"could not reach vLLM at {client.base_url}: {exc}"
        raise SoftmaxLintError(msg) from exc
    if len(names) != 1:
        msg = f"{client.base_url} advertises {len(names)} models, expected exactly one"
        raise SoftmaxLintError(msg)
    return names[0]


def main(argv: list[str] | None = None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    if len(argv) != 1:
        print(f"usage: {Path(sys.argv[0]).name} FILE", file=sys.stderr)
        return 2
    try:
        source = Path(argv[0]).read_text(encoding="utf-8")
        client = OpenAI(base_url=DEFAULT_URL, api_key="EMPTY")
        results = score_source(client, discover_model(client), source, FimMarkers())
    except (SoftmaxLintError, OSError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    graded = [(classify(a, CRIT_RATIO, WARN_RATIO), a) for a in results]
    findings = [(source.count("\n", 0, a["start"]) + 1, level, a) for level, a in graded if level]
    findings.sort(key=lambda finding: finding[2]["ratio"])
    if not findings:
        print(f"no surprises: {len(results)} tokens scored, all at or above ratio {WARN_RATIO}")
        return 0

    print(f"{len(findings)} surprising tokens (of {len(results)} scored):")
    for line, level, analysis in findings[:TOP_REPORTED]:
        print(
            f"  line {line:<4} {level:<8} ratio {analysis['ratio']:.3f}  rank {analysis['rank']:>3} "
            f"{analysis['token']!r} -> {analysis['top_token']!r}"
        )
    if len(findings) > TOP_REPORTED:
        print(f"  ... and {len(findings) - TOP_REPORTED} more")
    return 1 if any(level == "CRITICAL" for _, level, _ in findings) else 0


if __name__ == "__main__":
    sys.exit(main())
