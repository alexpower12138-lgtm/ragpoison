# STATE

last_updated: 2026-10-20 09:10 (WP1 complete)
current_phase: B
current_wp: WP1 (complete) -> WP2 (ready to start)
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

- Nothing blocks WP2. WP2 needs the stored API key (present) and about 50 minutes.
- If the repository should be published, the human creates an empty GitHub
  repository and provides the HTTPS URL; a credential must be configured at that
  point. Never paste a token into the conversation.

## Next action (exact command for human)

```
python run.py audit --stage retrieval --seed 42
```

That command arrives with WP2. Until then the entry point is:

```
python run.py gen-data --seeds 42 43
```

Say: start WP2.

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

