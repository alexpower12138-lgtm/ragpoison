# STATE

last_updated: 2026-10-20 00:00 (WP0)
current_phase: B
current_wp: WP0 (complete) -> WP1 (ready to start)
repo_root: C:\Users\13619\Desktop\notes\courses\compsci764\0728research_project\1020playground\ragpoison
spec: ..\1020项目计划书_v2.md

## Environment (verified on the human's machine)

- CPython 3.13.1, pip 24.3.1, Windows 11
- streamlit 1.66.0 -- AppTest importable (WP0 gate PASSED)
- numpy 2.4.2, pandas 3.0.1, scikit-learn 1.8.0, scipy 1.17.1, matplotlib 3.10.8,
  PyYAML 6.0.3, requests 2.32.5, pyarrow 24.0.0, pillow 12.2.0 already present
- pytest NOT installed -> all self-checks go through `python run.py selftest`
- git NOT installed -> no version control; the pipeline is resumable via the
  per-instance cache, and each work package leaves verifiable artefacts on disk
- The AI sandbox has no outbound network: every API call is made by the human
- DeepSeek balance was 84.27 CNY on 2026-10-20. The full experiment grid is
  estimated at 130-207 CNY, so the human tops up on demand; this is not a
  blocker for WP1 (offline) and does not change the 200 CNY configuration cap
- git is being installed via winget by the human (WP0 follow-up). Planned
  identity: user.name `alexpower12138-lgtm`, user.email the GitHub noreply
  address `342991819+alexpower12138-lgtm@users.noreply.github.com`

## Done (with evidence)

- WP0 repository skeleton created (30 files, all English)
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
  - requirements.txt, .gitignore, .streamlit/config.toml, README.md
- Verification: `python run.py selftest` -> 12/12 PASS
- Verification: `python run.py lint-lang` -> PASS, `lint-secrets` -> PASS
- Verification: secret DPAPI round-trip, store/status/clear all exercised with a
  dummy key, then removed (306-byte ciphertext, plaintext not recoverable from it)

## In progress

- (nothing)

## Blocked / waiting on human

- Store the real DeepSeek API key: `python run.py secret set`
  This is the only remaining WP0 action. Not needed until WP2, but doing it now
  closes the WP0 gate.

## Next action (exact command for human)

```
python run.py secret set
```

Then paste back the single line printed by `python run.py secret status`, which
looks like `source=dpapi  key=sk-******a1b2  len=35`, and say: start WP1.

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
- WP0-d: no git available on this machine. If a repository is needed later,
  install git first; `lint-secrets` skips the history scan when `.git` is absent.
