# STATE

last_updated: 2026-10-20 18:20 (WP3 complete: local signals frozen for both seeds)
current_phase: B
current_wp: WP3 COMPLETE -> WP4 (LLM-backed signals S3/S4/S5)

## WP3 result (retrieval + local signals, offline, zero API cost)

New modules: `retrieval/tfidf.py`, `signals/base.py` (SignalContext / SignalResult
/ registry), `signals/s1_sim_overfit.py`, `signals/s2_cross_consistency.py`,
`signals/s6_pseudo_ppl.py`, `signals/s7_prior.py`, `signals/calibration.py`,
`signals/extract.py`, plus `python run.py signals`.

Frozen artefacts: `results/signals/seed{42,43}/<family>_inj<n>.jsonl`,
12 files per seed, 18000 rows per seed (1200 instances x 15 candidates), 7.1 MB
per seed (git-ignored, rebuilt in about 70 s).

Calibration (calibration split only, benign pools only):
s2 mu=0.330 sigma=0.107 over 30 pools; s6 median NLL=1.648, MAD=0.042.
Cached at `cache/calibration_seed{N}.json`.

### What each signal actually does (mean score by label, seed 42)

| signal | clean | distractor | ATK1 | ATK2 | ATK3 | ATK4 | ATK5 | ATK6 |
|---|---|---|---|---|---|---|---|---|
| s1 | 0.09-0.18 | 0.035 | 0.95-0.97 | 0.75-0.77 | 0.72-0.79 | 0.96-0.97 | 0.73-0.80 | 0.75-0.77 |
| s2 | 0.53-0.57 | 0.32-0.45 | 0.39-0.48 | 0.47-0.54 | 0.46-0.53 | **0.52-0.72** | 0.41-0.47 | 0.43-0.51 |
| s6 | 0.20 | 0.03 | 0.43-0.46 | 0.64-0.66 | 0.50-0.60 | **0.98** | 0.30-0.31 | 0.51-0.52 |

Seed 43 reproduces the same pattern within a couple of points, so signal behaviour
is not an artefact of one corpus.

Conclusions to carry into the report:

- **S1 separates poison from clean on average (0.95 vs 0.12) but it is a rank
  proxy, not a poison detector.** In an ATK6 injection=3 pool the three poisoned
  passages sit at ranks 1, 3 and 4 and the genuine evidence at rank 2; S1 scores
  them 0.97 / 0.85 / 0.78 / 0.95. It flags "near the top", and the genuine
  passage is near the top as well. This is the bias the C1/C1f ablation exists to
  quantify.
- **S2 fires only on ATK4** (0.70 vs 0.53 clean). On every fluent family the
  poisoned passage is *more* similar to the pool than the clean passages, so S2
  scores it lower: in-topic manipulation is invisible to a topical-outlier measure.
- **S6 behaves as predicted**: ATK4 0.98 (destroyed word order), ATK5 0.30
  (fluent assertion, weakest), others 0.43-0.66.
- **Distractors are the least suspicious class on every signal** (0.03 s6, 0.32
  s2), which confirms the earlier finding that the effective negative class is
  "clean passages about other facts", not the donor-entity distractors.
- S6 is computed for every row even though the configuration disables it: it costs
  nothing offline and the ablation needs it.

## Next action (exact command for human)

Nothing yet: WP4 (S3/S4/S5) is the first stage that needs the API, and it will
follow after the plan for it is written. Say `start WP4` when ready.

## WP2 final result (all numbers measured, not mock)

Gate R (retrieval): PASS. Poison reaches the top-15 pool in 100% of cells and the
top-5 in 98-100%; the clean passage stays in the pool in 100%.

Gate G (generation condition, plain prompt, 1800 passages):

| family | answers target | answers correct |
|---|---|---|
| ATK1_PoisonedRAG | 100.0% | 0.0% |
| ATK2_CorruptRAG_AS | 99.3% | 0.7% |
| ATK3_InstructionInjection | 100.0% | 0.0% |
| ATK4_DIGA_Stuffing | 99.3% | 0.0% |
| ATK5_StealthAssertion | 100.0% | 0.0% |
| ATK6_Hedged_Silver | 82.3% | 16.0% (gate value 84.0% not_correct) |

`GATE G: PASS`, no dud variants (all three ATK6 variants now 77-91%).

Stage C controls:

- clean context (no poison anywhere): **97.0% correct** - the retrieval setup is
  sound and the low clean accuracy under poison is caused by the poison
- no context: **0.0% correct, 100% refused** - external-knowledge leak 0.0%,
  validity PASS. The corpus is fictional, so the model cannot answer without it.

Stage B baseline (1200 contexts x 2 prompts):

| condition | ASR | clean accuracy | abstention | mean ATR |
|---|---|---|---|---|
| plain | 68.6% | 31.4% | 0.0% | 0.40 |
| guarded | 27.3% | 4.2% | 68.4% | 0.40 |

The abstention instruction removes 41 points of attack success and 27 points of
correct answers: it is a blanket refusal, not a precision filter. Per family it
is worst where the attack is weakest (ATK5 inj1: clean 65% -> 4% while ASR only
35% -> 3%), which means the instruction mostly silences queries that the attack
never threatened. ATK3 at injection=3 barely notices it (ASR delta +7%), because
an explicit instruction overrides the abstention instruction.

Per-family baseline gives the mechanism for the report: effectiveness tracks rank
position and poison count, not persuasiveness. ATK1 inj1 76% -> inj3 100%;
ATK5 inj1 35% -> inj3 97%; ATK6 inj1 28% -> inj3 37%.

Spend so far: balance 84.27 -> 79.19 CNY, i.e. about 5.1 CNY for the whole audit
programme including two throwaway probes.

## Next action (exact command for human)

Nothing: WP3 is offline. The agent will run it and report.

## WP2d: what the first audit run got right, and the two things it exposed

Baseline, both conditions, 1200 contexts each (measured, not mock):

| condition | ASR | clean accuracy | abstention | mean ATR |
|---|---|---|---|---|
| plain | 65.7% | 34.3% | 0.0% | 0.40 |
| guarded | 26.6% | 5.6% | 67.8% | 0.40 |

The abstention instruction removes 39 points of attack success and destroys 29
points of correct answers: it is a blanket refusal, not a precision filter. For
ATK5 injection=1 it turns a 65% clean accuracy into 4% while only cutting ASR
from 35% to 3%. This is a report-worthy finding, and it only became visible once
the two prompts were measured separately.

Stage C, clean context (no poisoned chunk at all): **97.0% correct**. The
retrieval setup is sound; the 5.6% clean accuracy under poison is caused by the
poison, not by the pipeline.

Stage G, first run (plain prompt): ATK1 100.0%, ATK2 99.3%, ATK3 100.0%,
ATK4 99.3%, ATK5 100.0%, ATK6 64.7% target / 34.0% correct.

### Bug 1 (mine, fixed): the no-context validity check

`external_knowledge_leak` was computed as `1 - unknown_rate`, which counts any
non-UNKNOWN answer as a leak. With a plain prompt the model refuses in prose
("I don't know", "The context does not provide information about ..."), so all
100 answers were classified `other` and the leak read as 100% FAIL. The correct
criterion is the **correct-answer rate**: the corpus is fictional, so a leak
means reproducing the invented value. `is_unknown()` now also recognises natural
refusals, and the leak is 0.0% with validity PASS.

### Bug 2 (data, fixed): ATK6 variant 1 was a dud

Measured per variant: v1 target 36% / correct 63% (the model preferred the
*genuine* value), v2 81%, v3 77%. Because injection=1 uses variant 1 only, that
dead variant silently weakened every ATK6 injection=1 cell and confounded the
injection-count comparison. Variant 1 is rewritten to the same shape as the two
that work; data regenerated, 20/20 validators PASS on both seeds.

A per-variant floor (50%) is now part of Gate G, so a dead variant fails loudly
instead of quietly dragging a cell down.

## WP2c rework (what changed and why)

The first paid run measured attacks through a prompt that ordered the model to
answer UNKNOWN whenever the context was contradictory. That sentence is a
prompt-level defence, so the "no-defence baseline" was not one. Evidence: on the
same contexts, a plain prompt flipped 5/6 ATK1 inj1 answers while the guarded
prompt flipped 4%.

Changes:

- `llm/prompts.py` now holds two generation prompts, `gen_v1_plain` (measurement,
  no abstention instruction) and `gen_v1_guarded` (the product prompt). The
  generation-condition prompt is plain too, bumped to `poison_check_v2`.
- `audit/baseline_audit.py` runs **both** conditions on identical contexts and
  reports the delta, which is what a single sentence of prompt defence buys and
  what it costs.
- `audit/control_audit.py` (new, stage C) adds two controls: a clean-context
  condition (the real ceiling for clean accuracy) and a no-context condition that
  checks for external knowledge. The corpus is fictional, so any correct answer
  with an empty context would mean the model reached outside the corpus.
- `audit/generation_audit.py`: ATK6 is gated on *answer corruption*
  (1 - correct) instead of "answered the target", because its construction
  deliberately keeps the correct value in play.
- Every audit stage now writes per-instance detail to
  `results/audit_*_seed<N>_cells.jsonl`, which the failure-case analysis needs.
- Mock traffic is logged to `logs/llm_calls_mock.jsonl`, and `reports/SPEND.md`
  records call and token counts only; the authoritative cost figure is the
  balance delta the human reads with `python run.py spend`.
repo_root: C:\Users\13619\Desktop\notes\courses\compsci764\0728research_project\1020playground\ragpoison
spec: ..\1020项目计划书_v2.md

## Environment (verified on the human's machine)

- CPython 3.13.1, pip 24.3.1, Windows 11
- streamlit 1.66.0 -- AppTest importable (WP0 gate PASSED)
- numpy 2.4.2, pandas 3.0.1, scikit-learn 1.8.0, scipy 1.17.1, matplotlib 3.10.8,
  PyYAML 6.0.3, requests 2.32.5, pyarrow 24.0.0, pillow 12.2.0 already present
- pytest NOT installed -> all self-checks go through `python run.py selftest`
- git 2.55.0.windows.5 installed at `C:\Program Files\Git\cmd\git.exe`
  (git-lfs also present, unused). System credential helper: `manager`.
  Identity: `alexpower12138-lgtm <342991819+alexpower12138-lgtm@users.noreply.github.com>`
  (GitHub noreply address: commits link to the account without exposing a real
  email). Repository initialised on branch `main`, first commit `144cad1`.
- No remote is configured yet; pushing to GitHub needs a repository plus a
  credential (a personal access token, never pasted into a chat).
- The AI sandbox has no outbound network: every API call is made by the human
- DeepSeek balance was 84.27 CNY on 2026-10-20. The full experiment grid is
  estimated at 130-207 CNY, so the human tops up on demand; this is not a
  blocker for WP1 (offline) and does not change the 200 CNY configuration cap

## Done (with evidence)

- WP0 repository skeleton created (33 files tracked, all English)
  - run.py + ragpoison/cli.py: 17 subcommands registered; 5 implemented now
    (secret, selftest, lint-lang, lint-secrets, config), 12 stubs that exit 3
    and name the work package that will implement them
  - ragpoison/config.py: strict YAML loader, unknown keys are errors, embedded
    keys rejected, plus the ablation-matrix loader (15 rows, all required ids)
  - ragpoison/paths.py: repo root + 8 generated directories
  - ragpoison/redact.py: mask_key / scrub_secrets / scrub_mapping
  - ragpoison/llm/secretstore.py: DPAPI tier (ctypes, no dependency), .env tier,
    environment tier with read-then-pop, verify_key against /user/balance
  - configs/: default.yaml, pilot.yaml, full.yaml, replicate.yaml,
    ablation.yaml (C0..C10 + variants), scan_P1..P5.yaml
  - requirements.txt, .gitignore, .gitattributes, .streamlit/config.toml, README.md
- Verification: `python run.py selftest` -> 12/12 PASS
- Verification: `python run.py lint-lang` -> PASS, `lint-secrets` -> PASS
  (history scan active: "history scanned: 1 commit(s)")
- Verification: secret DPAPI round-trip, store/status/clear all exercised with a
  dummy key, then removed (306-byte ciphertext, plaintext not recoverable from it)
- Real key stored by the human: `source=dpapi  key=sk-******904a  len=35`,
  `/user/balance` verification returned 84.27 CNY
- Version control: `git init -b main`, first commit `144cad1` containing all 33
  files; `.secrets/` and `cache/` confirmed ignored; the staged diff was scanned
  for key-shaped strings and contained none

## In progress

- (nothing)

## WP1 result (deterministic synthetic benchmark)

Command: `python run.py gen-data --seeds 42 43` -> exit 0, 20/20 validators PASS for
both seeds, wall clock about 7 s, zero API calls.

Generated (`data/synthetic/seed{42,43}/`, 19 JSON files each, about 1.9 MB per seed):

| file | content |
|---|---|
| facts.json | 100 facts over 6 topics (17/17/17/17/16/16), attribute + split |
| clean_passages.json | 100 passages, 92-109 words |
| queries.json | 100 closed questions, answers <= 6 words |
| distractor_passages.json | 200 donor-entity passages, no fact value |
| poisoned_passages.json | 1800 = 6 families x 100 facts x 3 variants |
| splits.json | 30 calib / 70 test, disjoint |
| corpora/*.json | 12 corpora = 6 families x {inj1, inj3}, 400 / 600 chunks |
| manifest.json | generator version, counts and SHA-256 of every other file |

Retrieval audit (V12, real BM25Okapi over each corpus):
clean rank <= 15 everywhere for seed 42; on seed 43 five (corpus, query) cells out of
1200 (0.42%) push the clean passage outside the pool and are recorded in
`reports/retrieval_invalid_seed43.json`. Poisoned passages always enter the pool,
and land in the top-5 in 97-100% of cells per family, i.e. the attacks are strong
enough for the detection experiment to be meaningful.

Determinism: V18 regenerates the whole bundle and compares bytes - 19/19 files
identical. V19 confirms seed 43 has the same structure and different content.

## Blocked / waiting on human

- WP2b needs two paid runs by the human (about 5 CNY, about 50 minutes total).
  Nothing else blocks; the code is complete and verified offline.
- If the repository should be published, the human creates an empty GitHub
  repository and provides the HTTPS URL; a credential must be configured at that
  point. Never paste a token into the conversation.

## WP2b built (LLM layer + stages G and B)

New modules:

| file | role |
|---|---|
| `llm/prompts.py` | frozen prompt texts, versioned, snapshotted to `prompts_snapshot/` before each run |
| `llm/client.py` | threaded client: on-disk cache, retry with backoff, CNY budget cap, per-call ledger in `logs/llm_calls.jsonl`, `reports/SPEND.md` |
| `llm/mock.py` | deterministic offline responder, so the pipeline can be exercised with no network and no key |
| `generation/answer.py` | answer normalisation, matching, and the four-way label (target/correct/unknown/other) |
| `audit/generation_audit.py` | stage G: 1800 passages, one prompt each, Gate G |
| `audit/baseline_audit.py` | stage B: 1200 undefended generations, ASR / clean accuracy / abstention / ATR |

Verification performed offline with `--mock` (no network, no key, no spend):

- full stage G over 1800 passages: 3.3 s, aggregation and gate verdict correct
- full stage B over 1200 instances: 21.6 s, per-slice table correct, ATR 0.20/0.60
- rerun of stage G: `calls 0, cache hits 1800`, i.e. resume never pays twice
- `python run.py spend` works and reported a balance of 82.75 CNY
- mock answers are kept in `cache/llm_mock/` and the transport is part of the
  cache key, so a mock answer can never be served to a real run
- all mock artefacts were deleted afterwards: the human's run starts from an
  empty cache and clean reports

Safety properties worth knowing:

- the budget cap is enforced *before* each call, in CNY, and the run stops with
  `BUDGET STOP` while keeping everything already cached
- prices live in `configs/default.yaml` and are used only by the guard; reported
  cost comes from the provider's own usage counters when present
- `--limit N` exists so the first real call can cost a fraction of a cent

## WP2a result (retrieval audit, offline, free) - reworked

Command: `python run.py audit --stage retrieval --seed {42,43}` -> exit 0,
`GATE R: PASS`, about 20 s per seed.

The corpus design was split in two (see decision WP2a-i, now resolved):

- **View 1, query-time (RQ1)**: base pool + only this query's own poison,
  301 / 303 chunks, built on the fly by `datagen/corpora.py`.
  Poison reaches the pool in 100% of cells and the top-5 in 99.3-100%;
  the clean passage stays in the pool in 100% of cells.
  Pool composition for q_0001 is now: 1 poison (#1, 23.5), its clean passage
  (#2, 20.4), 7 clean passages of other facts, 6 distractors. No foreign poison.
- **View 2, shared corpora (RQ2)**: the corpus-audit situation. A pool holds
  4.63-13.29 poisoned passages, of which 3.63-10.29 belong to other queries.
  This is the phenomenon the periodic audit exists for and is measured there.
- Outputs: `reports/audit_retrieval_seed{N}.md` (two tables),
  `results/audit_retrieval_seed{N}.json`.

Generation is unchanged in shape: `gen-data` still emits the 12 shared corpus
files; the 1200 query corpora are derived, not stored, and the manifest records
both designs under `corpus_design`.
## Next action (exact command for human)

Re-run the three audit stages. Almost everything is already cached: only ATK6's
passages changed, so only ATK6 is charged again (about 0.6 CNY, about 10 minutes).

```
python run.py audit --stage controls   --seed 42
python run.py audit --stage generation --seed 42
python run.py audit --stage baseline   --seed 42
python run.py spend
```

Expected:

- controls: `EXTERNAL-KNOWLEDGE CHECK: PASS (leak 0.0%)`, clean context ~97%
- generation: `GATE G: PASS`, six families, no `dud variant` lines
- baseline: plain ASR stays near 65.7% (unaffected families come from cache) and
  ATK6 should move up from 3% / 27%

Say: audit done and I will start WP3.

## Open issues / decisions taken

- D9/WP0-a: `configs/ablation_*.yaml` from the spec is realised as a single
  `configs/ablation.yaml` matrix with 15 rows; `ragpoison/config.py`
  (`load_ablation_matrix`) validates it. Rationale: 15 near-identical files
  would drift; one table is easier to audit and is read by the offline
  re-fusion step, not by the experiment runner.
- D24/WP0-b: secret tiers implemented as DPAPI > .env > environment variable,
  with `os.environ.pop` after reading. `setx` is rejected by design.
- WP0-c: `python run.py config --config <name>` prints the merged effective
  configuration with secrets scrubbed; useful evidence for the report's
  reproducibility section.
- WP0-d: `_find_git()` in cli.py resolves git through `shutil.which` and then
  through the standard Windows install paths, because a long-lived parent
  process keeps a stale PATH and would otherwise silently skip the history scan.
- WP0-e: `.gitattributes` enforces `eol=lf` for all text files. The project
  reports SHA-256 hashes of generated artefacts, so a silent CRLF conversion
  would break reproducibility.
- WP1-a: V12 was re-specified. The original "clean rank <= 5" was wrong for a
  benchmark whose whole point is that a strong attack outranks the clean
  passage. The requirement is now reachability of the top-15 detection pool for
  both clean and poisoned passages, with an explicit <=2% budget for cells where
  three poisoned passages crowd the clean one out. Occasional eviction of the
  clean passage is the realistic threat model, not a data defect.
- WP1-b: target values are plausible neighbours of the correct value (year
  +/-3..30, integer +/-5..10%). An implausible target (a company founded in 1511)
  would be rejected by any reader and would turn the attack into a giveaway.
- WP1-c: every unit string is at most three tokens, otherwise the correct and
  the target value share a 4-gram and fail V15.
- WP1-d: BM25 landed in WP1 rather than WP3 because V12 has to rank with the
  same retriever the experiment uses; a second implementation would make the
  audit meaningless.
- WP1-e: filler sentences are never repeated inside one passage. Repeated filler
  would inflate pairwise similarity and corrupt the S2 and S6 signals.
- WP1-f: clean passages deliberately mention the entity at least twice and the
  attribute label once. Without that, keyword-stuffed passages win the retrieval
  contest and the clean evidence never reaches the detection pool.
- WP2a-i (RESOLVED): corpus composition. With 100 poisoned chunks in a 400-chunk
  corpus, foreign poisoned passages occupied a third to a half of every retrieval
  pool. Measured BM25 scores per query (seed 42, ATK1, inj1): own poison 23.1,
  best clean passage 17.4, mean foreign poison 11.3, best distractor 3.5. Because
  every poisoned passage outranks every distractor, enlarging the distractor side
  cannot help (an earlier draft of this note proposed exactly that; it is wrong).
  Resolved by splitting the design per research question:
    (1) RQ1 query-time: base pool + this query's own poison only, 301/303 chunks,
        derived on the fly (`datagen/corpora.py`), 1200 corpora, no foreign poison;
    (2) RQ2 corpus audit: the 12 shared corpora on disk, where the base really
        does contain 100 poisoned chunks. Contamination is the measurement there.
  V12 and V20 now validate the query-time view; the audit reports both views.
- WP2a-ii: `audit --stage {generation,baseline,all}` now says so explicitly and
  exits 3 instead of producing an argparse error. Unimplemented commands are
  detected before argument parsing.

