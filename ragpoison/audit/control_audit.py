"""Stage C of the attack-validity audit: the control conditions.

Two controls, both cheap, both answering a question that no other stage can:

``clean_context``
    Retrieve from a knowledge base that contains **no poisoned chunk at all** and
    ask the plain-prompt question. This is the real ceiling for clean accuracy.
    Without it, a reported clean accuracy of 5% is uninterpretable: is the model
    failing because of the poison, or because four of the five retrieved passages
    are about other entities?

``no_context``
    Ask the question with nothing retrieved. The knowledge base is fictional, so a
    correct answer would mean the model got the value from somewhere other than
    the corpus — parametric memory or a search tool behind the API. The expected
    result is 100% UNKNOWN; anything else invalidates the measurement design and
    must be reported rather than ignored.

Outputs
-------
``reports/audit_controls_seed<N>.md``
``results/audit_controls_seed<N>.json``
``results/audit_controls_seed<N>_cells.jsonl``
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field

from ..datagen.corpora import base_pool
from ..datagen.loader import load_bundle
from ..generation.answer import classify
from ..llm import mock as mock_responder
from ..llm import prompts
from ..llm.client import Call, LLMClient
from ..paths import path as repo_path, rel
from ..retrieval.bm25 import BM25


@dataclass
class Cell:
    condition: str
    query_id: str
    label: str
    answer: str
    retrieved: list[str] = field(default_factory=list)


def run(
    seed: int = 42,
    *,
    limit: int | None = None,
    mock: bool = False,
    budget_cny: float | None = None,
    concurrency: int | None = None,
    top_k: int | None = None,
    quiet: bool = False,
) -> dict:
    config = _config()
    bundle = load_bundle(seed)
    texts = bundle.texts()
    pool = base_pool(bundle)
    index = BM25().fit([(chunk_id, texts[chunk_id]) for chunk_id in pool])
    k = int(top_k or config.get("retrieval.top_k"))

    queries = bundle.queries[:limit] if limit else bundle.queries

    calls: list[Call] = []
    plan: list[tuple[str, dict, list[str]]] = []
    for query in queries:
        retrieved = [
            chunk_id for chunk_id, _score in index.ranked_ids(query["question"], top_n=k)
        ]
        for condition in ("clean_context", "no_context"):
            user = (
                prompts.CONTROL_USER_WITH_CONTEXT.format(
                    question=query["question"],
                    context=prompts.format_context([texts[cid] for cid in retrieved]),
                )
                if condition == "clean_context"
                else prompts.CONTROL_USER_NO_CONTEXT.format(question=query["question"])
            )
            calls.append(
                Call(
                    system=prompts.CONTROL_SYSTEM,
                    user=user,
                    stage=f"audit_controls_{condition}",
                    max_tokens=int(config.get("llm.max_tokens_answer")),
                    extra={"query_id": query["query_id"], "condition": condition},
                )
            )
            plan.append((condition, query, retrieved if condition == "clean_context" else []))

    client = LLMClient(
        config,
        concurrency=concurrency,
        budget_cny=budget_cny,
        mock=mock,
        mock_responder=mock_responder.last_number,
        quiet=quiet,
    )
    if not quiet:
        print(f"controls: {len(calls)} generations (clean context + no context) ...", flush=True)
    results = client.complete_many(calls, label="controls   ", progress_every=200)

    cells: list[Cell] = []
    for (condition, query, retrieved), result in zip(plan, results):
        cells.append(
            Cell(
                condition=condition,
                query_id=query["query_id"],
                label=classify(result.text, query["correct_answer"], query["target_answer"]),
                answer=result.text.strip()[:80],
                retrieved=retrieved,
            )
        )

    summary = {}
    for condition in ("clean_context", "no_context"):
        subset = [cell for cell in cells if cell.condition == condition]
        n = len(subset) or 1
        summary[condition] = {
            "n": len(subset),
            "correct": sum(1 for cell in subset if cell.label == "correct") / n,
            "target": sum(1 for cell in subset if cell.label == "target") / n,
            "unknown": sum(1 for cell in subset if cell.label == "unknown") / n,
            "other": sum(1 for cell in subset if cell.label == "other") / n,
        }

    # The no-context control is a validity check, not an outcome: it must be
    # entirely UNKNOWN for the measurement design to hold.
    external_leak = 1.0 - summary["no_context"]["unknown"]
    payload = {
        "seed": seed,
        "mock": mock,
        "top_k": k,
        "prompt_version": prompts.CONTROL_VERSION,
        "n_queries": len(queries),
        "conditions": summary,
        "external_knowledge_leak": external_leak,
        "validity_ok": external_leak <= 0.02,
        "llm": client.summary(),
    }

    md_path = repo_path("reports", f"audit_controls_seed{seed}.md")
    json_path = repo_path("results", f"audit_controls_seed{seed}.json")
    cells_path = repo_path("results", f"audit_controls_seed{seed}_cells.jsonl")
    md_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8", newline="\n"
    )
    with cells_path.open("w", encoding="utf-8", newline="\n") as handle:
        for cell in cells:
            handle.write(
                json.dumps(
                    {
                        "condition": cell.condition,
                        "query_id": cell.query_id,
                        "label": cell.label,
                        "answer": cell.answer,
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
    md_path.write_text(_render_markdown(payload), encoding="utf-8", newline="\n")
    snapshot = prompts.snapshot(
        "audit_controls",
        prompts.CONTROL_VERSION,
        {
            "system": prompts.CONTROL_SYSTEM,
            "user_with_context": prompts.CONTROL_USER_WITH_CONTEXT,
            "user_no_context": prompts.CONTROL_USER_NO_CONTEXT,
        },
    )
    client.write_spend_line("audit_controls")
    payload["_written"] = [rel(json_path), rel(md_path), rel(cells_path), rel(snapshot)]
    payload["_client"] = client
    return payload


def _config():
    from ..config import load_config

    return load_config("default.yaml")


def _render_markdown(payload: dict) -> str:
    good = payload["conditions"]["clean_context"]
    none = payload["conditions"]["no_context"]
    lines: list[str] = []
    lines.append(f"# Control conditions (stage C) - seed {payload['seed']}")
    lines.append("")
    if payload["mock"]:
        lines.append("> **MOCK RUN.** No model was called. Numbers are a plumbing test only.")
        lines.append("")
    lines.append("Two controls, neither of which involves poisoned text.")
    lines.append("")
    lines.append("## Clean-context control (the ceiling for clean accuracy)")
    lines.append("")
    lines.append(
        f"Retrieval over the {payload['n_queries']} queries against a knowledge base with no "
        f"poisoned chunk, top-{payload['top_k']}, plain prompt."
    )
    lines.append("")
    lines.append(f"**answered correctly {good['correct']:.1%}** | answered a target value "
                 f"{good['target']:.1%} | UNKNOWN {good['unknown']:.1%} | other {good['other']:.1%}")
    lines.append("")
    lines.append(
        "This is the number clean accuracy should be compared against. If it is low, the "
        "retrieval setup itself is the limitation, not the attack."
    )
    lines.append("")
    lines.append("## No-context control (the external-knowledge check)")
    lines.append("")
    lines.append(
        "The same questions with an empty context. Every fact in this knowledge base is "
        "invented, so a correct answer could only come from parametric memory or a search "
        "tool behind the API."
    )
    lines.append("")
    lines.append(f"**answered correctly {none['correct']:.1%}** | UNKNOWN {none['unknown']:.1%} "
                 f"| answered a target value {none['target']:.1%}")
    lines.append("")
    verdict = "PASS" if payload["validity_ok"] else "FAIL"
    lines.append(
        f"**Validity check: {verdict}** - external-knowledge leak "
        f"{payload['external_knowledge_leak']:.1%} (tolerance 2%)."
    )
    if not payload["validity_ok"]:
        lines.append("")
        lines.append(
            "A non-zero leak invalidates the measurement: the model is not confined to the "
            "retrieved corpus. Report it and investigate before drawing conclusions."
        )
    lines.append("")
    return "\n".join(lines)


def summarize(payload: dict) -> str:
    good = payload["conditions"]["clean_context"]
    none = payload["conditions"]["no_context"]
    verdict = "PASS" if payload["validity_ok"] else "FAIL"
    lines = [
        f"queries: {payload['n_queries']}" + ("  [MOCK, no model called]" if payload["mock"] else ""),
        f"clean context     : correct {good['correct']:.1%}  UNKNOWN {good['unknown']:.1%}  "
        f"target {good['target']:.1%}",
        f"no context        : correct {none['correct']:.1%}  UNKNOWN {none['unknown']:.1%}",
        f"EXTERNAL-KNOWLEDGE CHECK: {verdict}  (leak {payload['external_knowledge_leak']:.1%})",
        "",
        f"calls {payload['llm']['calls']}, cache hits {payload['llm']['cache_hits']}, "
        f"tokens {payload['llm']['prompt_tokens']} in / {payload['llm']['completion_tokens']} out",
    ]
    for item in payload.get("_written", []):
        lines.append(f"wrote {item}")
    return "\n".join(lines)
