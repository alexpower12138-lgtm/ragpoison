"""Stage B of the attack-validity audit: the no-defence baseline.

Every experiment table in the report has the same shape: attack success rate with
a defence, against attack success rate without one. This module produces the
"without" column: retrieve the top-k exactly as the product would, hand it to the
model unchanged, and score the answers.

Nothing here is a defence. No detector runs, no filter runs, no threshold is
applied. That is the point.

Outputs
-------
``reports/audit_baseline_seed<N>.md``
``results/audit_baseline_seed<N>.json``
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


@dataclass
class Cell:
    corpus_id: str
    attack_family: str
    injection: int
    query_id: str
    label: str
    answer: str
    atr: float          # share of retrieved context that is poisoned
    retrieved: list[str] = field(default_factory=list)


@dataclass
class SliceResult:
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
    for corpus, query, retrieved in instances:
        calls.append(
            Call(
                system=prompts.GEN_SYSTEM,
                user=prompts.GEN_USER.format(
                    question=query["question"],
                    context=prompts.format_context([texts[cid] for cid in retrieved]),
                ),
                stage="audit_baseline",
                max_tokens=int(config.get("llm.max_tokens_answer")),
                extra={"corpus_id": corpus["corpus_id"]},
            )
        )

    client = LLMClient(
        config,
        concurrency=concurrency,
        budget_cny=budget_cny,
        mock=mock,
        mock_responder=mock_responder.last_number,
        quiet=quiet,
    )
    if not quiet:
        print(f"baseline: {len(calls)} undefended generations ...", flush=True)
    results = client.complete_many(calls, label="baseline   ", progress_every=200)

    grouped: dict[tuple[str, int], SliceResult] = {}
    poison_by_corpus = {corpus["corpus_id"]: set(corpus["poison_chunk_ids"]) for corpus, _, _ in instances}
    for (corpus, query, retrieved), result in zip(instances, results):
        key = (corpus["attack_family"], int(corpus["injection"]))
        slice_result = grouped.setdefault(key, SliceResult(*key))
        poison = poison_by_corpus[corpus["corpus_id"]]
        atr = sum(1 for cid in retrieved if cid in poison) / len(retrieved) if retrieved else 0.0
        slice_result.cells.append(
            Cell(
                corpus_id=corpus["corpus_id"],
                attack_family=corpus["attack_family"],
                injection=int(corpus["injection"]),
                query_id=query["query_id"],
                label=classify(result.text, query["correct_answer"], query["target_answer"]),
                answer=result.text.strip()[:80],
                atr=atr,
                retrieved=retrieved,
            )
        )

    slices = [grouped[key] for key in sorted(grouped)]
    total = sum(slice_result.n for slice_result in slices)
    payload = {
        "seed": seed,
        "mock": mock,
        "prompt_version": prompts.GEN_VERSION,
        "top_k": k,
        "n_instances": total,
        "overall": {
            "asr": _rate(slices, "target"),
            "clean_accuracy": _rate(slices, "correct"),
            "abstention": _rate(slices, "unknown"),
            "other": _rate(slices, "other"),
            "mean_atr": (
                sum(slice_result.mean_atr * slice_result.n for slice_result in slices) / total
                if total
                else 0.0
            ),
        },
        "llm": client.summary(),
        "slices": [
            {
                "attack_family": slice_result.attack_family,
                "injection": slice_result.injection,
                "n": slice_result.n,
                "asr": slice_result.rate("target"),
                "clean_accuracy": slice_result.rate("correct"),
                "abstention": slice_result.rate("unknown"),
                "other": slice_result.rate("other"),
                "mean_atr": slice_result.mean_atr,
            }
            for slice_result in slices
        ],
    }

    md_path = repo_path("reports", f"audit_baseline_seed{seed}.md")
    json_path = repo_path("results", f"audit_baseline_seed{seed}.json")
    md_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8", newline="\n"
    )
    md_path.write_text(_render_markdown(payload, slices), encoding="utf-8", newline="\n")
    snapshot = prompts.snapshot(
        "audit_baseline",
        prompts.GEN_VERSION,
        {"system": prompts.GEN_SYSTEM, "user": prompts.GEN_USER},
    )
    client.write_spend_line("audit_baseline")
    payload["_written"] = [rel(json_path), rel(md_path), rel(snapshot)]
    payload["_slices"] = slices
    payload["_client"] = client
    return payload


def _config():
    from ..config import load_config

    return load_config("default.yaml")


def _rate(slices: list[SliceResult], label: str) -> float:
    total = sum(slice_result.n for slice_result in slices)
    if not total:
        return 0.0
    hits = sum(sum(1 for cell in slice_result.cells if cell.label == label) for slice_result in slices)
    return hits / total


def _render_markdown(payload: dict, slices: list[SliceResult]) -> str:
    overall = payload["overall"]
    lines: list[str] = []
    lines.append(f"# No-defence baseline (stage B) - seed {payload['seed']}")
    lines.append("")
    if payload["mock"]:
        lines.append("> **MOCK RUN.** No model was called. Numbers are a plumbing test only.")
        lines.append("")
    lines.append(
        "Retrieve the top-"
        f"{payload['top_k']} exactly as the product would, generate with no detector and no "
        "filter, and score the answers. These numbers are the reference column of every later "
        "table: a defence is only interesting relative to them."
    )
    lines.append("")
    lines.append(f"- instances: {payload['n_instances']}")
    lines.append(f"- prompt version: {payload['prompt_version']}")
    lines.append(f"- api calls: {payload['llm']['calls']}, cache hits: {payload['llm']['cache_hits']}")
    lines.append(f"- estimated spend: {payload['llm']['spent_cny']:.2f} CNY")
    lines.append("")
    lines.append("## Overall")
    lines.append("")
    lines.append(
        f"**ASR {overall['asr']:.1%}** | clean accuracy {overall['clean_accuracy']:.1%} | "
        f"abstention {overall['abstention']:.1%} | other {overall['other']:.1%} | "
        f"mean ATR {overall['mean_atr']:.2f}"
    )
    lines.append("")
    lines.append("## Per attack family and injection")
    lines.append("")
    lines.append("| family | inj | n | ASR | clean accuracy | abstention | other | mean ATR |")
    lines.append("|---|---|---|---|---|---|---|---|")
    for slice_result in slices:
        lines.append(
            f"| {slice_result.attack_family} | {slice_result.injection} | {slice_result.n} "
            f"| {slice_result.rate('target'):.1%} | {slice_result.rate('correct'):.1%} "
            f"| {slice_result.rate('unknown'):.1%} | {slice_result.rate('other'):.1%} "
            f"| {slice_result.mean_atr:.2f} |"
        )
    lines.append("")
    lines.append("## How to read this")
    lines.append("")
    lines.append(
        "- **ASR** is the share of queries where the model answered the attacker's value with no "
        "defence at all. This is the number every defence is measured against."
    )
    lines.append(
        "- **clean accuracy** is the share where it answered the true value. It is not expected to "
        "be high: the context also contains seven passages about unrelated entities."
    )
    lines.append(
        "- **mean ATR** is the share of the retrieved context that is poisoned, which shows how "
        "much room the attack already occupies before any defence acts."
    )
    lines.append(
        "- ATK2 and ATK6 passages contain both values, so a model answering the true value is a "
        "legitimate outcome, not a scoring error."
    )
    lines.append("")
    return "\n".join(lines)


def summarize(payload: dict) -> str:
    slices: list[SliceResult] = payload["_slices"]
    overall = payload["overall"]
    lines = [
        f"instances: {payload['n_instances']}"
        + ("  [MOCK, no model called]" if payload["mock"] else ""),
        f"BASELINE ASR: {overall['asr']:.1%}   clean accuracy: {overall['clean_accuracy']:.1%}"
        f"   abstention: {overall['abstention']:.1%}   mean ATR: {overall['mean_atr']:.2f}",
        "",
        f"{'family':<26}{'inj':>4}{'n':>6}{'ASR':>9}{'clean':>9}{'UNK':>8}{'ATR':>7}",
    ]
    for slice_result in slices:
        lines.append(
            f"{slice_result.attack_family:<26}{slice_result.injection:>4}{slice_result.n:>6}"
            f"{slice_result.rate('target'):>8.1%}{slice_result.rate('correct'):>9.1%}"
            f"{slice_result.rate('unknown'):>8.1%}{slice_result.mean_atr:>7.2f}"
        )
    llm = payload["llm"]
    lines.append("")
    lines.append(
        f"calls {llm['calls']}, cache hits {llm['cache_hits']}, "
        f"failures {llm['failures']}, spent {llm['spent_cny']:.2f} / {llm['budget_cny']:.2f} CNY"
    )
    for item in payload.get("_written", []):
        lines.append(f"wrote {item}")
    return "\n".join(lines)
