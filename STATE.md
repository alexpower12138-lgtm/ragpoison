# STATE

last_updated: 2026-10-21 14:10 (end-to-end defence stage written; waiting for the paid run)
current_phase: C
current_wp: WP7b code complete -> human runs `python run.py defence`

## WP7b: end-to-end defence

`python run.py defence` closes the gap WP6 leaves open. It filters the pool with a
configuration from the ablation matrix, then generates an answer from what is left
with the *same* prompt the baseline used, and classifies the reply. That answers the
question the ablation cannot: does the attack still work after filtering, and did the
filter also remove what the model needed to answer.

Design points that make the comparison honest:

* `gen_v1_plain`, the measurement prompt. The guarded prompt would bundle a second
  defence into the treatment.
* **C0 is free and is the reference**: with no defence the context is the plain
  top-k, the request is byte-identical to the baseline audit's, and the cache answers
  all 840 of them. The dry run confirms 840/840 cached.
* pooled filtering means filtered contexts collapse: 5040 generations are only
  1657 unique prompts and 753 of those are unpaid, because many configurations
  produce the same surviving set. The whole stage costs under 1 CNY.
* an empty pool is asked with the no-context prompt (the stage-C control), which is
  itself mostly cached.

Configurations carried forward: C0, C1, C4, C7, C8, C8l. C1 is included on purpose
as the case where a high clean_rate comes from cutting too much.

Outputs: `results/defence_seed{N}.json`,
`results/defence_seed{N}_cells.jsonl`, `reports/defence_seed{N}.md`. The report gives
ASR, correct rate, abstention, poison in context, refills, abstentions, Δ against C0,
per-family and per-injection breakdowns, and a McNemar test on "the attack still
worked".

## Next action (exact command for human)

```
python run.py defence
```
753 paid generations, under 1 CNY, a few minutes. Seed 42 only; add `--seeds 42 43`
for the replication (1664 paid generations in total).

## WP6: the offline ablation

`python run.py refusion` scores the fifteen configurations of `configs/ablation.yaml`
against the frozen tables. Zero API calls, 19 seconds for both seeds, because every
configuration sees exactly the same stored signal values and only the fusion rule
changes. Outputs: `results/summary_ablation.csv`,
`results/summary_ablation_details.json` (thresholds, weights, logistic coefficients,
per-injection and per-family breakdowns, paired tests) and `reports/ablation.md`.

New modules: `signals/load.py` (the join, plus the row identity), `fusion/fuse.py`
(three rules), `fusion/calibrate.py` (threshold and weight fitting on the calibration
split only), `fusion/policy.py` (filtering with refill), `evaluation/aggregate.py`
(metric definitions), `evaluation/stats.py` (pool-clustered bootstrap, exact McNemar),
`evaluation/ablation.py` (orchestration and report).

### Two bugs found while building it

1. **A document-id collision that corrupted every score.** Scores were stored in a
   dictionary keyed by `doc_id`. Document ids are not unique: the 12600 test
   candidates share only 1484 distinct ids, one of them appearing 198 times, because
   the same base-pool passage is retrieved for many queries. Each lookup therefore
   returned the score of some *other* query's candidate. It was caught by an
   inconsistency that the numbers themselves could not explain - S5 alone reported a
   0.000 false-positive rate at a 5% budget - and by then the first version of the
   table had already been written. Everything now goes through `signals.load.row_key`,
   and selftest item 6 fails the build if the identity is ever not unique.
   This affected the ablation only: the `full` and `replicate` reports read the
   columns directly and were re-derived independently.

2. **The "95th percentile" threshold was not a 5% false-positive rate.** S5 can only
   take 0, 0.25, 0.5 or 0.75, and 89% of benign candidates sit at exactly 0, so its
   95th percentile is 0.5 and the threshold actually flagged 1.7% of benign
   candidates. Each configuration was therefore operating at a different point of its
   own ROC curve, and the comparison was partly a comparison of operating points.
   The threshold is now the lowest cut that keeps the benign false-positive rate at
   or below the 5% budget; discrete signals land under the budget and the achieved
   rate is recorded.

## The result

Pool-level `clean_rate` - the fraction of pools whose poisoned passages are all gone
from the context - on the test split, 840 pools:

| config | seed 42 | seed 43 | what it is |
|---|---|---|---|
| C0 | 0.000 | 0.000 | no defence |
| C1 | 0.515 | 0.351 | S1 alone |
| C1f | 0.000 | 0.000 | S1 with the sign flipped (control) |
| C2 | 0.051 | 0.035 | S2 alone |
| C3 | 0.407 | 0.386 | S3 alone |
| C3b | 0.000 | 0.000 | S3b alone (negative control) |
| C4 | 0.545 | 0.539 | S4 alone |
| C5 | 0.464 | 0.468 | S5 alone |
| C6 | 0.530 | 0.448 | S1+S2+S3 |
| C7 | 0.577 | 0.582 | S4+S5 |
| C8 | 0.823 | 0.815 | all six, mean |
| C8w | 0.843 | 0.857 | all six, weighted |
| C8l | **0.892** | **0.919** | all six, logistic |
| C9 | 0.846 | 0.848 | all minus S3 |
| C10 | 0.818 | 0.810 | all minus S5 |

H2 is supported where it matters: the strongest single signal reaching the same 5%
budget is C4 at 0.545/0.539, and the fused logistic model reaches 0.892/0.919, with
McNemar p < 1e-4 on both seeds. Dropping S3 from the mean rule (C9) is very slightly
better than keeping it, which is consistent with S3 ranking the genuine evidence
above the poison.

The controls behave: flipping the sign of S1 takes it from 0.515 to 0.000, and the
majority-vote variant S3b removes nothing at all.

## The gap this stage deliberately leaves open

`clean_rate` cannot tell "removed the poison" from "removed everything". C1 is the
case in point: it scores 0.515 while its hard-subset AUC against genuine evidence is
0.396, because cutting the top-ranked candidates removes poison and evidence alike.
Answerability after filtering needs a generation from the filtered context, which is
a paid step that has not been run: the pilot for it would be 5 configurations x 1200
instances, and C0 reuses the cached baseline prompts exactly.

## Next action (human choice)

Either `python run.py defence` (to be written: generate from the filtered contexts,
about 3 CNY) or WP8, the Streamlit dashboard over the tables that already exist.

## R2: valid, and it overturns the pilot's impression

Data checks: 18000 local rows, 18000 LLM rows, 1200 instances (100 queries x 6
families x 2 injection sizes), 2400 positives (600 pools x 1 or 3), 0 rows with
`call_failed`, 0 S3/S4 parse failures, 2 partial S5 arrays out of 10800 batches
(0.02%). Actual spend 6.68 CNY, i.e. the configured price guard overestimates by
about 2.5x - the measured rate is close to 1.3 CNY per million input tokens.

The pilot had only injection size 1. Splitting the hard subset by injection size
shows that redundancy changes the problem qualitatively:

| signal | pilot (inj 1 only) | R2 inj=1 | R2 inj=3 |
|---|---|---|---|
| instruction (S4) | 0.779 | 0.787 | **0.801** |
| ppl (S6) | 0.905 | 0.841 | 0.842 |
| cross (S2) | 0.717 | 0.637 | 0.610 |
| multiview (S5) | 0.794 | 0.778 | **0.546** |
| query (S1) | 0.495 | 0.444 | 0.392 |
| conflict (S3) | 0.438 | 0.380 | 0.377 |
| conflict_majority (S3b) | 0.491 | 0.489 | **0.021** |

### The finding: multi-injection poisoning inverts consensus detection

Measured, not inferred:

* the poisons of a pool agree with each other - at inj=3 there is a single distinct
  extracted answer in **533 of 600 pools** (two answers in 58, three in 9);
* majority-vote support is therefore 3/15 for each poison and 1/15 for the single
  genuine passage, so s3b = 0.80 for the poison and 0.93 for the evidence;
* mean s3b on the hard subset: poison 0.775 against evidence 0.929 at inj=3
  (0.912 against 0.932 at inj=1). The genuine passage becomes the outlier.

An AUC of 0.021 is not "weak", it is close to a perfect inversion: majority-vote
consensus detection names the one correct passage as the anomaly. This is the
strongest result the project has produced, and it is the reason the report cannot
be built on a single-injection pilot.

S5 loses about a quarter of its discrimination for the same reason: its
`contradiction` view asks whether a passage conflicts with the others about the same
subject, and three agreeing poisons do not conflict with anything.

What survives is exactly what reads a passage on its own: S4 is unchanged by
redundancy (0.787 -> 0.801) and S6 likewise. That is H3 with numbers.

### Caveats to carry into WP6

* Positives are clustered 3-per-pool at inj=3, so any confidence interval must be
  bootstrapped over the 600 pools, not over 2400 rows.
* S6's strength is partly a synthetic-template artefact, as already recorded.
* The pilot's multiview figure (0.794) was flattered by having only injection size 1.

## Next action (exact command for human)

```
python run.py replicate
```
10256 calls, about 7 CNY at the measured rate, roughly 1.5-3 hours.

## Pilot #2: clean, and it already answers the main research question

```
instances 120 (configured 120), rows 1800, positives 120
parse failures: S3 0.0%, S4 0.0%, S5 batches 0
S3 UNKNOWN 86.8% (c_ref 0.0968, calib_clean_answered)
calls 497, cache hits 2587, failures 0
```

Confirmed afterwards: 0 rows with `call_failed`, 120/120 instances, and a re-run
with the same table cache issues 0 calls, so the pipeline is complete and
deterministic.

### The result

| signal | table | AUC | AUC on the hard subset |
|---|---|---|---|
| query (S1) | local | 0.961 | **0.495** |
| cross (S2) | local | 0.594 | 0.717 |
| conflict (S3) | llm | 0.957 | **0.438** |
| conflict_majority (S3b) | llm | 0.948 | 0.491 |
| instruction (S4) | llm | 0.777 | 0.779 |
| multiview (S5) | llm | 0.984 | 0.794 |
| ppl (S6) | local | 0.934 | 0.905 |

The hard subset holds the 120 poisoned passages against the 120 *clean passages
that actually answered the question* - poison against genuine evidence, no easy
negatives.

This is the paper's central finding and it is exactly the predicted split:

* S1, S3 and S3b look excellent on the whole pool and collapse to chance, or below
  chance, once the trivially irrelevant candidates are removed. They measure "this
  passage answers the question", not "this passage is poisoned". S3 is *below*
  chance at 0.438: it ranks the genuine evidence as more suspicious than the
  poison, because the genuine passage concentrates the question words slightly more
  (0.097 against 0.086).
* S4 and S5 hold up. Per family, instruction scores 1.000 on ATK2, ATK3 and ATK4
  and 0.521 - chance - on ATK5 and ATK6, exactly as designed, because the stealth
  attacks carry no traces. multiview stays between 0.958 and 1.000 on all six,
  so the semantic judge is not fooled by the missing traces. That is H3.

### Caveat to argue in the report

S6 reaches 0.905 on the hard subset, which is too good to be a property of
pseudo-perplexity. The clean passages here come from a small template bank, so a
character-trigram model trained on benign text memorises that style, and the poison
templates are a different distribution. Part of S6's strength is therefore a
synthetic-data artefact, and the report must say so rather than claim a style
detector works in general.

### Also settled: the parser ambiguity

Nine of 1080 S5 requests contained a second, revised array. Only three distinct
batches disagree with their first array, and switching between "first" and "last"
moves the S5 AUC by 0.0006. The parser now takes the first balanced JSON span in
the order it appears (it used to take the longest, which made the result depend on
which revision was wordier), and the count is reported.

## Next action (exact command for human)

```
python run.py full
```
9112 new calls, about 17 CNY at the measured rate, roughly 1.5-3 hours.

## Pilot run #1 inspection (the first paid pilot)

The run produced plausible numbers, but two real problems were found by reading the
artefacts rather than the summary.

### 1. 26 network failures (a network drop during the run)

Every one of them is a `ConnectionError` or `ReadTimeout`, distributed as 5 x S3,
6 x S4, 15 x S5. A failed call is never cached, so it was *not* mistaken for an
answer, but the rows it should have produced were written as neutral values:

| stage | what a failure became | consequence |
|---|---|---|
| S3 | answer UNKNOWN | indistinguishable from a genuine "cannot answer" |
| S4 | score 0.0 | the passage looks perfectly clean - the worst possible loss |
| S5 | all flags 0 | the view contributes nothing |

That is why `instruction` scored AUC 0.777, the lowest of the LLM signals: six of
its zeroes are not judgements. Verified separately that no *cached* answer is an
error page: all 12162 entries were scanned for html, error wording, empty text and
zero token counts; the only hits were three S5 responses discussed below.

Fixes: a retry pass over failed calls (two extra rounds), a per-row `call_failed`
flag, a loud warning in the console and in the report, and the runner prints
`instances (configured: N)` so a partial table is visible.

### 2. S5 responses were being truncated while the model second-guessed itself

Three of 1080 S5 batches returned a valid array, then `"Wait, I need to
reconsider..."`, then ran out of the token budget. The parser keeps the first
array, i.e. the version the model wanted to revise. The prompt now says "Decide
once. Do not reconsider", the token budget went from 3x to 5x, and the version is
bumped to `s5_multiview_v2`, which invalidates the S5 cache (about 300 unique
prompts) - cheap now, expensive after R2.

### 3. A `--limit` run could join two runs

Running with `--limit 6` and then `--limit 2` left both runs' files in the directory
and the report claimed 106 instances. The runner now clears the target directory's
`.jsonl` files at the start of every run (safe: local tables rebuild for free and
LLM answers come from the cache) and prints how many files it cleared.

### Also added: the hard-subset AUC

The headline AUC is easy to inflate: most negatives cannot answer the question at
all, so a signal that merely detects "this passage answers the question" scores
well. The report now also gives an AUC computed only over the positives and the
clean passages that did answer, i.e. poison against genuine evidence.

## Next action (exact command for human)

Re-run the pilot. Only 497 calls are new (S5 was invalidated by the prompt fix),
about 1.4 CNY, about 6 minutes.

```
python run.py pilot
```

Say `pilot done` afterwards.

## WP5: the experiment runner

`python run.py {pilot|full|replicate} [--dry-run] [--seeds ..] [--limit ..] [--mock]`

Each mode reads its own config, selects instances, builds the local table, builds
the LLM table, joins them on `(corpus_id, query_id, doc_id)`, and writes
`reports/run_{mode}_seed{N}.md` plus `results/run_{mode}_seed{N}.json`. The report
contains the per-signal AUC, the mean score per label, the per-family AUC, and
three whole instances for a human to eyeball.

- `pilot` writes to `results/signals/seed42_pilot/` so it can never clobber the
  full table; `full` and `replicate` use `results/signals/seed{N}/`.
- `--dry-run` builds every request and reports how many would actually be paid
  for, without calling the API. This exists because the dedupe is not obvious.

### A cost bug found while verifying the runner

The pilot configuration asks for injection size 1 only, but the runner passed
`injection=None` down to the extractors, so both injection sizes were built: 240
instances instead of 120, i.e. twice the money. Found by checking that the mock
pilot had produced 12 signal files instead of 6. The injection filter is now
pushed into the extractors, and the report prints `instances (configured: N)` so a
mismatch is visible.

### Measured budget (dry run, real cache state)

| mode | instances | requests built | unique prompts | already cached | **to pay** | estimate |
|---|---|---|---|---|---|---|
| R1 pilot | 120 | 3084 | 1092 | 102 | **990** | 2.70 CNY |
| R2 full | 1200 | 30833 | 10204 | 102 | **10102** | 29.38 CNY |
| R3 replicate | 1200 | 30832 | 10256 | 0 | **10256** | 29.70 CNY |

The estimate column uses the configured price (2.0 CNY per million input tokens);
the measured rate from the balance is about 0.65 of that, so the realistic total
for R1+R2+R3 is closer to **40 CNY, not the 76 CNY estimated earlier**. The reason
is that 30833 requests collapse into 10204 unique prompts: base-pool passages recur
across attack families and injection sizes, and the cache recognises them.

R1 also caches 1092 of R2's unique prompts, so running the pilot first costs almost
nothing extra in the end.

## Next action (exact command for human)

```
python run.py pilot
```

About 990 paid calls, 8-12 minutes, roughly 2 CNY. Say `pilot done` afterwards and
I will read `reports/run_pilot_seed42.md` and decide whether R2 is safe to launch.

## WP0-WP4 sign-off: independent re-derivation of every headline number

Recomputed from the raw per-instance files rather than trusting the summaries:

| check | result |
|---|---|
| stage G rates recomputed from `*_cells.jsonl` vs the headline JSON | identical, both seeds, 1800 cells each |
| stage B ASR / clean / abstention recomputed from cells | identical, both seeds, 2400 cells each |
| stage C leak and clean-accuracy recomputed from cells | 0.000 / 0.970 (seed 42), 0.000 / 0.990 (seed 43), identical |
| local signal tables | 12 files, 18000 rows, 1200 instances x 15 rows, 0 out-of-range values, no `cross_query_poison` (correct for query-time corpora) |
| the two signal tables join | 45/45 smoke rows joined on `(corpus_id, query_id, doc_id)`, label agreement 100% |
| generated data vs its manifest | sha256 verified, both seeds, generator 1.1.0 |
| prompt snapshots vs current constants | 11/11 identical - no drift between what was paid for and what the code now says |
| module health | 43 modules import, 0 failures |
| S1 and S3 formulas vs the plan document | numeric check agrees (sigmoid((z-0.5)/0.15); clip(c/(2 c_ref))) |
| `selftest` | 16/16 PASS; `lint-lang` and `lint-secrets` PASS; 61 tracked files, 18 commits |

### Problems found by the double-check and fixed

1. **Latent defect: the ablation matrix spoke a language no table understood.**
   `configs/ablation.yaml` refers to `query`, `cross`, `conflict`,
   `conflict_majority`, `instruction`, `multiview`, `ppl`; none of those is a
   column in either signal table. The mapping existed only in prose, so WP6 would
   have had to reinvent it and could silently pick the wrong column. It is now
   `signals/base.py: ABLATION_COLUMNS`, and selftest item 5 fails the build if
   any name in the matrix has no column or if a mapped column disappears.
2. Dead constant `SIGNAL_ORDER` was 6 names long (missing `conflict_majority`) and
   used nowhere; replaced by the mapping above.
3. Noise removed: `logs/llm_calls_mock.jsonl` (8.2 MB of mock traffic, no
   evidential value) and the 45-row partial `ATK1_PoisonedRAG_inj1.llm.jsonl`
   left by the smoke test, which could have been mistaken for a complete table.
   Re-creating the smoke table costs nothing, since every answer is cached.

### Known design facts (not defects)

- S3 and S3b do not separate poison from genuine evidence on this benchmark, for
  the structural reason recorded above. C3 and C3b measure it.
- S6 is disabled in the configuration but computed anyway: it is free offline and
  the ablation needs the column.
- The `enabled` column in the local table records the configuration, not what was
  computed; the fusion step is what decides.
- Paid cache now holds 11198 answers; balance 75.50 CNY, i.e. 8.8 CNY spent on the
  whole programme so far.

## WP4 smoke test (3 instances, real API, 0.08 CNY)

```
llm signals: 3 instances, 102 requests (30 distinct passages for S4)
parse failures: S3 0/45 (0.0%), S4 0/30 (0.0%), S5 batches 0/27
S3 UNKNOWN answers: 86.7%
```

The parsers are the part that could have invalidated R2, and they are clean: the
model returns well-formed JSON for all three prompt shapes, including the batched
array. 102 requests for 3 instances confirms the S4 dedupe (45 candidate rows
collapse to 30 distinct passages).

### Two S3 design errors, both found by reading the raw rows

1. **Sign.** The plan's formula was `s3 = 1 - clip(c / (2 c_ref))` while its own
   rationale says "a poisoned passage compresses the words into a short span". The
   formula produced the opposite of the intent: poison 0.553 against clean 0.822.
   Corrected to rise with density. Same class of mistake as the sign error in the
   first version of S1, and the report should say so.
2. **UNKNOWN is not a score.** The plan scored an UNKNOWN extraction as maximally
   suspicious. 86.7% of candidates cannot answer a question about another entity,
   so the signal saturated at 1.0 and flagged the *clean* rows hardest. UNKNOWN is
   now recorded as `s3_unknown` and the density falls back to question tokens.

### Honest status of the four LLM signals after the fixes

Per-label means on the smoke slice (poison / clean / distractor):

| signal | poison | clean | distractor |
|---|---|---|---|
| s3 | 0.447 | 0.178 | 0.054 |
| s3b | 0.933 | 0.219 | 0.133 |
| s4 | 0.667 | 0.000 | 0.000 |
| s5 | 0.750 | 0.089 | 0.000 |

S4 and S5 separate cleanly, which is what the project needs: the style-level
signals can tell a poisoned passage from genuine evidence.

**S3 and S3b do not really separate, and the table above overstates them.** The
clean mean is dragged down by 25 rows that answered UNKNOWN. Per pool, the one
genuine evidence row has density 0.097 against the poison's 0.086, so both score
alike: these two signals measure "this passage answers the question", and in a pool
exactly two passages answer it - the attack and the evidence. That is a finding to
report (C3, C3b), not something to hide behind label means. The ablation must be
read on the ranked pool, never on these means.

## Next action (exact command for human)

None yet: R1 needs the pilot runner, which is WP5. Say `start WP5`.

## WP4: LLM-backed signals

New modules: `llm/parse.py` (strict JSON recovery), `signals/s3_freq_density.py`,
`signals/s3b_majority.py`, `signals/llm_extract.py` (orchestration), and S3/S4/S5
prompts in `llm/prompts.py`. Command: `python run.py signals --stage llm`.

Two tables, joined on `(corpus_id, query_id, doc_id)` in WP6:

| table | content | cost |
|---|---|---|
| `{family}_inj{n}.jsonl` | S1/S2/S6/S7 (WP3) | free, rebuildable in 70 s |
| `{family}_inj{n}.llm.jsonl` | S3, S3b, S4, S5 + the extracted answer | paid, written once |

Deliberate cost controls:

- S4 depends only on passage text, so it is issued once per distinct passage per
  process. A full 1200-instance run needs **30833 requests of which only 2033 are
  S4**: without the dedupe it would be 46800.
- S5's instruction view reuses S4, so the four views cost three calls per batch.
- S3's raw density `s3_c` is stored rather than the normalised score, so `c_ref`
  can be taken from the calibration split at fusion time. A run that does not cover
  the calibration split falls back to all clean rows and says so
  (`c_ref_source=all_clean_fallback`).
- S3b (naive majority vote) is computed from the same extractions, so the negative
  control C3b costs nothing.

Verified offline with the mock transport: one instance gives 39 requests
(15 S3 + 15 S4 + 9 S5), the single-instance table renders, the full matrix gives
1200 instances / 30833 requests / 2033 distinct S4 passages in 124 s, and the
JSON parsers pass fenced, prose-wrapped and malformed inputs (selftest 15/15).

Deviations recorded:

- The first S5 view is phrased as "asserts an answer without support" instead of
  the plan's "relevance": a relevance flag fires on distractors, which is the
  opposite of useful. The configuration key keeps its original name.
- Missing ids in an S5 array are treated as not-flagged and counted in the
  parse-failure tally rather than triggering a re-ask; the real run will show
  whether that is acceptable.

## Next action (exact command for human)

Smoke test with the real key (a few cents, 1-2 minutes):

```
python run.py signals --stage llm --seed 42 --limit 3
```

Then report the parse-failure lines. Expected: `parse failures: S3 0/45`,
`S4 0/..`, `S5 batches 0/27`, and an `S3 UNKNOWN answers` rate that is not
near 100%.

## WP2 on seed 43: the cross-knowledge-base replication result (I5)

Both seeds now have Gate R, Gate G, the controls and the two-condition baseline.
The two knowledge bases were generated independently, share no facts, and the
conclusions reproduce:

| measure | seed 42 | seed 43 |
|---|---|---|
| Gate G canonical worst | 99.3% | 98.0% |
| Gate G stealth (ATK6) | 82.3% target / 84.0% not_correct | 82.7% / 82.7% |
| dud variants | none | none |
| baseline plain ASR | 68.6% | 68.1% |
| baseline plain clean accuracy | 31.4% | 31.9% |
| baseline guarded ASR | 27.3% | 25.8% |
| baseline guarded clean accuracy | 4.2% | 3.6% |
| clean-context control | 97.0% | 99.0% |
| external-knowledge leak | 0.0% | 0.0% |

Per-family ASR under the plain prompt, worst disagreement between seeds:

| family | seed 42 inj1 / inj3 | seed 43 inj1 / inj3 | max gap |
|---|---|---|---|
| ATK1 | 76% / 100% | 80% / 100% | 4 |
| ATK2 | 68% / 96% | 61% / 95% | 7 |
| ATK3 | 54% / 99% | 45% / 99% | 9 |
| ATK4 | 55% / 78% | 60% / 71% | 7 |
| ATK5 | 35% / 97% | 37% / 100% | 3 |
| ATK6 | 28% / 37% | 29% / 40% | 3 |

Every qualitative pattern holds: injection=3 dominates injection=1, ATK6 is the
weakest attack, ATK5 is the weakest at injection=1 and near-perfect at 3, and the
abstention instruction costs clean accuracy everywhere.

Cache accounting for seed 43: 1 control prompt and 3 baseline prompts were already
cached. Verified that all four are genuine duplicate prompts (identical question
and identical context, either across injection sizes or across the two seeds), not
a key collision: the baseline produced 2400 prompts of which 2398 are unique, and
the control prompts are all unique within seed 43.

Spend so far: 84.27 -> 76.55 CNY, i.e. about 7.7 CNY for the whole audit
programme on both seeds (the in-file estimate reads higher because the price table
is deliberately conservative; the balance is the authoritative number).

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

