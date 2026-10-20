# RAG Poisoning Detection

Detection of adversarially poisoned retrieval contexts in a Retrieval-Augmented
Generation (RAG) pipeline, using composable multi-signal judgement, plus a
budgeted periodic corpus audit (PCA).

> **Status: WP0 (repository skeleton).** Only the CLI skeleton, configuration
> loader, secret handling and repository layout exist. Detection, retrieval,
> data generation, experiments, evaluation and the dashboard are implemented in
> later work packages. Every not-yet-implemented command exits with a message
> naming the work package that will provide it.

---

## 1. What the project does (target design)

Given a user query `q` and the `top-N` candidates returned by a BM25 retriever,
the system scores every candidate with a set of pluggable, independently
switchable signals, fuses those signals into a single poison score, removes
candidates above a threshold, and only then calls the generator:

```
q -> BM25 (top-N=15) -> signal extractor -> fusion -> filter -> generation -> scoring
                                   |
                        S1 similarity over-fit
                        S2 cross-passage consistency
                        S3 answer centrality (frequency density)
                        S4 instruction / template traces
                        S5 multi-view judgement (4 views)
                        S6 pseudo-perplexity
                        S7 retrieval prior (fusion weight only, never a verdict)
```

A second, independent path audits the whole knowledge base periodically instead
of per query:

```
scheduler -> scan policy -> priority queue -> stage-1 coarse filter (no LLM)
          -> stage-2 batched LLM scan -> chunk status store -> scan report
```

Six attack families are used, split into a *canonical* tier that leaves visible
traces and a *stealth* tier that does not, so that detection difficulty can be
attributed to trace visibility rather than to "severity".

## 2. Requirements

* Windows 10/11 (DPAPI is used for secret storage; Linux/macOS fall back to `.env`)
* Python 3.11+ (developed and tested on CPython 3.13.1)
* A DeepSeek API key -- only needed from work package WP2 onwards

## 3. Install

```powershell
python -m pip install -r requirements.txt
python -c "import streamlit; print(streamlit.__version__)"    # must be >= 1.28
```

Other people can reproduce the environment with exactly the same two lines.

## 4. Quick start

```powershell
python run.py --help
python run.py secret set --api-key-env none    # interactive, hidden input
python run.py secret status
python run.py selftest
```

## 5. Secrets -- read this before anything else

The API key must never appear in code, configuration, logs, the repository, a
commit, or a chat log.

**Preferred (DPAPI, encrypted at rest, no environment variable involved):**

```powershell
python run.py secret set        # hidden prompt, stored in .secrets/deepseek.dpapi
python run.py secret status     # prints source + masked key, e.g. sk-******a1b2
python run.py secret clear      # remove the stored ciphertext
```

**Alternative (`.env`, git-ignored):**

```powershell
" DEEPSEEK_API_KEY=sk-..." > .env      # one line, no leading space
icacls .env /inheritance:r /grant:r "$($env:USERNAME):F"
```

**Do not use `setx`.** It writes the key into the PowerShell history file, into
`HKCU\Environment` in plaintext, and it is inherited by every subsequently
started process of the same user. Removal is also non-obvious.

Resolution order is `DPAPI > .env > DEEPSEEK_API_KEY`, and the environment
variable is popped from `os.environ` as soon as it is read.

## 6. Repository layout

```
run.py                     single CLI entry point
ragpoison/
  cli.py                   argument parsing and dispatch
  config.py                YAML loading, strict validation, secret rejection
  paths.py                 repository root and generated-directory helpers
  redact.py                key masking and log scrubbing
  llm/secretstore.py       DPAPI / .env / environment secret resolution
  datagen/                 synthetic data generator + validators      (WP1)
  audit/                   attack-validity audits                      (WP2)
  retrieval/               BM25 and TF-IDF                             (WP3)
  signals/                 S1..S7 signal extractors                   (WP3/WP4)
  llm/                     API client, cache, prompts, parsing        (WP4)
  fusion/                  threshold / weighted / logistic policy     (WP6)
  generation/              answer generation and matching             (WP4)
  evaluation/              metrics, statistics, offline re-fusion     (WP6)
  scan/                    periodic corpus audit (PCA)                (WP9)
  viz/                     Streamlit application, data layer, charts  (WP8)
configs/                   YAML experiment configurations
```

## 7. Commands

| Command | Purpose | Available |
|---|---|---|
| `secret set` / `status` / `verify` / `clear` | manage the API key | now |
| `selftest` | offline self-checks (no API, no network) | now |
| `lint-lang` | reject non-English text in the repository | now |
| `lint-secrets` | reject committed key material | now |
| `gen-data` | generate the synthetic benchmark | WP1 |
| `audit` | retrieval / generation-condition / baseline audits | WP2 |
| `pilot` / `full` / `replicate` | experiment runs | WP5 / WP7 |
| `refusion` / `metrics` / `figures` | offline ablation, statistics, plots | WP6 |
| `ui-check` / `dashboard` | headless Streamlit check, launch | WP8 |
| `scan` | periodic corpus audit policies P1..P5 | WP9 |
| `spend` | query the account balance (free endpoint) | WP4 |

## 8. Reproducibility notes

* All random behaviour is derived from an explicit integer seed; the synthetic
  data generator is deterministic and byte-reproducible (verified by SHA-256).
* Experiment ablations are computed offline from a frozen per-candidate signal
  table, so an ablation never triggers additional API calls.
* Every experiment run records the exact prompts used under `prompts_snapshot/`.
* `python -m pip freeze > requirements-lock.txt` produces the pinned environment
  used for the final results.

## 9. Academic use

Course project for COMPSCI 764. Synthetic knowledge base; no real people,
companies or events are used anywhere in the generated data.
