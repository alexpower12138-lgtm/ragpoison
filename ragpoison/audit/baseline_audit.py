"""Stage B of the attack-validity audit: the no-defence baseline.

Every experiment table in the report has the same shape: attack success rate with
a defence, against attack success rate without one. This module produces the
"without" column.

It measures two conditions on identical retrieved contexts:

``plain``
    No abstention instruction. This is the real baseline: what the attack does
    when nothing defends it.
``guarded``
    The product prompt, which orders the model to answer UNKNOWN when the context
    is contradictory. That sentence is itself a defence, so it is measured as a
    condition rather than baked into the baseline.

The gap between the two is a result in its own right: it is how much a single
sentence of prompt-level defence buys, and what it costs in clean accuracy.

Nothing here is a detector. No filter runs, no threshold is applied.

Outputs
-------
``reports/audit_baseline_seed<N>.md``
``results/audit_baseline_seed<N>.json``
``results/audit_baseline_seed<N>_cells.jsonl``
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field

from ..datagen.corpora import iter_query_corpora
from ..datagen.loader import load_bundle
from ..generation.answer import classify
from ..llm import mock as mock_responder
from ..llm import prompts
from ..llm.client import Call, LLMClient
from ..paths import path as repo_path, rel
from ..retrieval.bm25 import BM25

#: (condition name, system prompt, prompt version)
CONDITIONS: tuple[tuple[str, str, str], ...] = (
    ("plain", prompts.GEN_PLAIN_SYSTEM, prompts.GEN_PLAIN_VERSION),
    ("guarded", prompts.GEN_GUARDED_SYSTEM, prompts.GEN_GUARDED_VERSION),
)


@dataclass
class Cell:
    condition: str
    corpus_id: str
    attack_family: str
    injection: int
    query_id: str
    label: str
    answer: str
    atr: float


@dataclass
class SliceResult:
    condition: str
    attack_family: str
    injection: int
    cells: list[Cell] = field(default_factory=list)

    @property
    def n(self) -> int:
        return len(self.cells)

    def rate(self, label: str) -> float:
        if not self.cells:
            return 0.0
        return sum(1 for cell in self.cells if cell.label == label) / len(self.cells)

    @property
    def mean_atr(self) -> float:
        return sum(cell.atr for cell in self.cells) / len(self.cells) if self.cells else 0.0


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
    queries = {row["query_id"]: row for row in bundle.queries}
    k = int(top_k or config.get("retrieval.top_k"))

    instances: list[tuple[dict, dict, list[str]]] = []
    for corpus in iter_query_corpora(bundle):
        query = queries[corpus["query_id"]]
        index = BM25().fit([(chunk_id, texts[chunk_id]) for chunk_id in corpus["chunk_ids"]])
        retrieved = [chunk_id for chunk_id, _score in index.ranked_ids(query["question"], top_n=k)]
        instances.append((corpus, query, retrieved))
        if limit is not None and len(instances) >= limit:
            break

    calls: list[Call] = []
    plan: list[tuple[str, dict, dict]] = []
    for condition, system, _version in CONDITIONS:
        for corpus, query, retrieved in instances:
            calls.append(
                Call(
                    system=system,
                    user=prompts.GEN_USER.format(
                        question=query["question"],
                        context=prompts.format_context([texts[cid] for cid in retrieved]),
                    ),
                    stage=f"audit_baseline_{condition}",
                    max_tokens=int(config.get("llm.max_tokens_answer")),
                    extra={"corpus_id": corpus["corpus_id"], "condition": condition},
                )
            )
            plan.append((condition, corpus, query))

    client = LLMClient(
        config,
        concurrency=concurrency,
        budget_cny=budget_cny,
        mock=mock,
        mock_responder=mock_responder.last_number,
        quiet=quiet,
    )
    if not quiet:
        print(
            f"baseline: {len(instances)} contexts x {len(CONDITIONS)} conditions "
            f"= {len(calls)} generations ...",
            flush=True,
        )
    results = client.complete_many(calls, label="baseline   ", progress_every=200)

    by_corpus_poison = {
        corpus["corpus_id"]: set(corpus["poison_chunk_ids"]) for corpus, _, _ in instances
    }
    retrieved_by_instance = {
        corpus["corpus_id"]: retrieved for corpus, _query, retrieved in instances
    }

    grouped: dict[tuple[str, str, int], SliceResult] = {}
    for (condition, corpus, query), result in zip(plan, results):
        key = (condition, corpus["attack_family"], int(corpus["injection"]))
        slice_result = grouped.setdefault(key, SliceResult(*key))
        poison = by_corpus_poison[corpus["corpus_id"]]
        retrieved = retrieved_by_instance[corpus["corpus_id"]]
        atr = sum(1 for cid in retrieved if cid in poison) / len(retrieved) if retrieved else 0.0
        slice_result.cells.append(
            Cell(
                condition=condition,
                corpus_id=corpus["corpus_id"],
                attack_family=corpus["attack_family"],
                injection=int(corpus["injection"]),
                query_id=query["query_id"],
                label=classify(result.text, query["correct_answer"], query["target_answer"]),
                answer=result.text.strip()[:80],
                atr=atr,
            )
        )

    slices = [grouped[key] for key in sorted(grouped)]
    overall = {
        condition: _overall([row for row in slices if row.condition == condition])
        for condition, _system, _version in CONDITIONS
    }
    payload = {
        "seed": seed,
        "mock": mock,
        "top_k": k,
        "prompt_versions": {name: version for name, _system, version in CONDITIONS},
        "n_instances": len(instances),
        "overall": overall,
        "delta": _delta(slices),
        "llm": client.summary(),
        "slices": [
            {
                "condition": row.condition,
                "attack_family": row.attack_family,
                "injection": row.injection,
                "n": row.n,
                "asr": row.rate("target"),
                "clean_accuracy": row.rate("correct"),
                "abstention": row.rate("unknown"),
                "other": row.rate("other"),
                "mean_atr": row.mean_atr,
            }
            for row in slices
        ],
    }

    md_path = repo_path("reports", f"audit_baseline_seed{seed}.md")
    json_path = repo_path("results", f"audit_baseline_seed{seed}.json")
    cells_path = repo_path("results", f"audit_baseline_seed{seed}_cells.jsonl")
    md_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8", newline="\n"
    )
    with cells_path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in slices:
            for cell in row.cells:
                handle.write(
                    json.dumps(
                        {
                            "condition": cell.condition,
                            "corpus_id": cell.corpus_id,
                            "attack_family": cell.attack_family,
                            "injection": cell.injection,
                            "query_id": cell.query_id,
                            "label": cell.label,
                            "answer": cell.answer,
                            "atr": round(cell.atr, 3),
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )
    md_path.write_text(_render_markdown(payload, slices), encoding="utf-8", newline="\n")
    snapshots = [
        prompts.snapshot(f"audit_baseline_{name}", version, {"system": system, "user": prompts.GEN_USER})
        for name, system, version in CONDITIONS
    ]
    client.write_spend_line("audit_baseline")
    payload["_written"] = [rel(json_path), rel(md_path), rel(cells_path)] + [
        rel(item) for item in snapshots
    ]
    payload["_slices"] = slices
    payload["_client"] = client
    return payload


def _config():
    from ..config import load_config

    return load_config("default.yaml")


def _overall(slices: list[SliceResult]) -> dict:
    total = sum(row.n for row in slices)
    if not total:
        return {"asr": 0.0, "clean_accuracy": 0.0, "abstention": 0.0, "other": 0.0, "mean_atr": 0.0}
    hits = {label: sum(sum(1 for cell in row.cells if cell.label == label) for row in slices) for label in ("target", "correct", "unknown", "other")}
    return {
        "asr": hits["target"] / total,
        "clean_accuracy": hits["correct"] / total,
        "abstention": hits["unknown"] / total,
        "other": hits["other"] / total,
        "mean_atr": sum(row.mean_atr * row.n for row in slices) / total,
    }


def _delta(slices: list[SliceResult]) -> list[dict]:
    """Per slice, what the guarded prompt changed relative to the plain one."""
    by_key: dict[tuple[str, int], dict[str, SliceResult]] = {}
    for row in slices:
        by_key.setdefault((row.attack_family, row.injection), {})[row.condition] = row
    out: list[dict] = []
    for (family, injection), pair in sorted(by_key.items()):
        plain = pair.get("plain")
        guarded = pair.get("guarded")
        if plain is None or guarded is None:
            continue
        out.append(
            {
                "attack_family": family,
                "injection": injection,
                "asr_drop": plain.rate("target") - guarded.rate("target"),
                "clean_drop": plain.rate("correct") - guarded.rate("correct"),
                "abstention_rise": guarded.rate("unknown") - plain.rate("unknown"),
            }
        )
    return out


def _render_markdown(payload: dict, slices: list[SliceResult]) -> str:
    lines: list[str] = []
    lines.append(f"# No-defence baseline (stage B) - seed {payload['seed']}")
    lines.append("")
    if payload["mock"]:
        lines.append("> **MOCK RUN.** No model was called. Numbers are a plumbing test only.")
        lines.append("")
    lines.append(
        f"Retrieve the top-{payload['top_k']} exactly as the product would and generate twice on "
        "the same context: once with a plain instruction, once with the product prompt that "
        "orders the model to answer UNKNOWN when the context is contradictory. No detector and "
        "no filter runs in either condition."
    )
    lines.append("")
    lines.append(
        f"- instances: {payload['n_instances']} contexts x {len(CONDITIONS)} conditions"
    )
    lines.append(
        "- prompt versions: "
        + ", ".join(f"`{name}`={version}" for name, version in payload["prompt_versions"].items())
    )
    lines.append(f"- api calls: {payload['llm']['calls']}, cache hits: {payload['llm']['cache_hits']}")
    lines.append(f"- tokens: {payload['llm']['prompt_tokens']} prompt / {payload['llm']['completion_tokens']} completion")
    lines.append("")
    lines.append("## Overall, both conditions")
    lines.append("")
    lines.append("| condition | ASR | clean accuracy | abstention | other | mean ATR |")
    lines.append("|---|---|---|---|---|---|")
    for name in payload["prompt_versions"]:
        row = payload["overall"][name]
        lines.append(
            f"| {name} | {row['asr']:.1%} | {row['clean_accuracy']:.1%} | {row['abstention']:.1%} "
            f"| {row['other']:.1%} | {row['mean_atr']:.2f} |"
        )
    lines.append("")
    for name in payload["prompt_versions"]:
        rows = [row for row in slices if row.condition == name]
        lines.append(f"## Condition `{name}`, per family and injection")
        lines.append("")
        lines.append("| family | inj | n | ASR | clean accuracy | abstention | other | mean ATR |")
        lines.append("|---|---|---|---|---|---|---|---|")
        for row in rows:
            lines.append(
                f"| {row.attack_family} | {row.injection} | {row.n} "
                f"| {row.rate('target'):.1%} | {row.rate('correct'):.1%} "
                f"| {row.rate('unknown'):.1%} | {row.rate('other'):.1%} "
                f"| {row.mean_atr:.2f} |"
            )
        lines.append("")
    lines.append("## What the abstention instruction costs")
    lines.append("")
    lines.append("| family | inj | ASR drop | clean-accuracy drop | abstention rise |")
    lines.append("|---|---|---|---|---|")
    for row in payload["delta"]:
        lines.append(
            f"| {row['attack_family']} | {row['injection']} "
            f"| {row['asr_drop']:+.1%} | {row['clean_drop']:+.1%} | {row['abstention_rise']:+.1%} |"
        )
    lines.append("")
    lines.append("## How to read this")
    lines.append("")
    lines.append(
        "- The **plain** condition is the baseline every defence is measured against. "
        "The **guarded** condition is the product prompt and therefore already contains a "
        "prompt-level defence."
    )
    lines.append(
        "- **ASR** is the share of queries where the model answered the attacker's value. "
        "**clean accuracy** is the share where it answered the true value."
    )
    lines.append(
        "- The difference between the two conditions is what a single sentence buys: the ASR "
        "drop is the protection, the clean-accuracy drop is the price, and the abstention rise "
        "is the mechanism."
    )
    lines.append(
        "- ATK2 and ATK6 passages contain both values, so answering the true value is a "
        "legitimate outcome there, not a scoring error."
    )
    lines.append("")
    return "\n".join(lines)


def summarize(payload: dict) -> str:
    lines = ["instances: {} contexts x {} conditions".format(payload["n_instances"], len(CONDITIONS))]
    if payload["mock"]:
        lines[0] += "  [MOCK, no model called]"
    lines.append("")
    lines.append(f"{'condition':<10}{'ASR':>8}{'clean':>8}{'UNK':>8}{'ATR':>7}")
    for name, row in payload["overall"].items():
        lines.append(
            f"{name:<10}{row['asr']:>7.1%}{row['clean_accuracy']:>8.1%}"
            f"{row['abstention']:>8.1%}{row['mean_atr']:>7.2f}"
        )
    lines.append("")
    lines.append("plain condition, per family and injection")
    lines.append(f"{'family':<26}{'inj':>4}{'n':>6}{'ASR':>8}{'clean':>8}{'UNK':>8}{'ATR':>7}")
    for row in payload["_slices"]:
        if row.condition != "plain":
            continue
        lines.append(
            f"{row.attack_family:<26}{row.injection:>4}{row.n:>6}"
            f"{row.rate('target'):>7.1%}{row.rate('correct'):>8.1%}"
            f"{row.rate('unknown'):>8.1%}{row.mean_atr:>7.2f}"
        )
    llm = payload["llm"]
    lines.append("")
    lines.append(
        f"calls {llm['calls']}, cache hits {llm['cache_hits']}, failures {llm['failures']}, "
        f"tokens {llm['prompt_tokens']} in / {llm['completion_tokens']} out"
    )
    for item in payload.get("_written", []):
        lines.append(f"wrote {item}")
    return "\n".join(lines)
