"""Command-line interface: argument parsing and dispatch.

Each subcommand is a thin wrapper; all real work lives in the modules named in
the corresponding work package. Commands that are not implemented yet exit with
status 3 and a message naming the work package that will provide them, so that
the interface contract is stable from WP0 onwards.
"""

from __future__ import annotations

import argparse
import getpass
import json
import os
import re
import shutil
import subprocess
import sys
import traceback
from pathlib import Path
from typing import Callable

from . import __version__
from .config import (
    REQUIRED_ABLATIONS,
    ConfigError,
    load_ablation_matrix,
    load_config,
    render_config,
)
from .llm import secretstore
from .paths import REPO_ROOT, ensure_dirs, path as repo_path, rel
from .redact import SECRET_RE, mask_key, scrub_secrets

EXIT_OK = 0
EXIT_FAILED = 1
EXIT_NO_SECRET = 2
EXIT_PLANNED = 3

#: Commands promised by the specification, mapped to the work package that
#: implements them. Registered now so `--help` documents the full contract.
PLANNED: dict[str, str] = {
    "pilot": "WP5: R1 pilot experiment",
    "full": "WP7: R2 main experiment",
    "replicate": "WP7: R3 seed replication",
    "refusion": "WP6: offline ablation",
    "metrics": "WP6: statistics and figures",
    "figures": "WP6: report figures",
    "ui-check": "WP8: headless Streamlit check",
    "dashboard": "WP8: Streamlit application",
    "scan": "WP9: periodic corpus audit",
    "spend": "WP4: account balance",
}

_CJK_RE = re.compile(r"[\u2e80-\u9fff\uf900-\ufaff\uff00-\uffef\u3000-\u303f]")

#: Substrings that are allowed to contain non-ASCII characters because they are
#: verbatim filenames that the specification requires us to reference exactly.
_ALLOWED_CJK: tuple[str, ...] = ("1020项目计划书_v2.md",)

_SKIP_DIRS = {
    ".git",
    ".secrets",
    ".venv",
    "venv",
    "__pycache__",
    ".pytest_cache",
    ".mypy_cache",
    ".ruff_cache",
    "data",
    "cache",
    "logs",
    "results",
    "figures",
    "reports",
    "prompts_snapshot",
    "node_modules",
}

_TEXT_SUFFIXES = {
    ".py",
    ".md",
    ".txt",
    ".yaml",
    ".yml",
    ".toml",
    ".cfg",
    ".ini",
    ".html",
    ".css",
    ".js",
    ".json",
    ".tex",
    ".bib",
}


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #


def _iter_text_files(root: Path = REPO_ROOT) -> list[Path]:
    files: list[Path] = []
    for candidate in sorted(root.rglob("*")):
        if not candidate.is_file():
            continue
        if any(part in _SKIP_DIRS for part in candidate.relative_to(root).parts):
            continue
        if candidate.suffix.lower() not in _TEXT_SUFFIXES:
            continue
        if candidate.name.endswith(".dpapi"):
            continue
        files.append(candidate)
    return files


def _read_text(candidate: Path) -> str | None:
    try:
        return candidate.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return None


def _cjk_lines(text: str) -> list[int]:
    """Return the 1-based line numbers that contain non-English characters.

    Occurrences of the allow-listed verbatim filenames are removed first, so that
    referring to the (Chinese-named) specification file is not a violation.
    """
    hits: list[int] = []
    for lineno, line in enumerate(text.splitlines(), start=1):
        probe = line
        for allowed in _ALLOWED_CJK:
            probe = probe.replace(allowed, "")
        if _CJK_RE.search(probe):
            hits.append(lineno)
    return hits


def _print_planned(command: str) -> int:
    owner = PLANNED.get(command, "a later work package")
    print(f"'{command}' is not implemented yet; it arrives in {owner}.")
    print("See the project specification (section 1.4) for the work-package order.")
    return EXIT_PLANNED


# --------------------------------------------------------------------------- #
# secret
# --------------------------------------------------------------------------- #


def cmd_secret(args: argparse.Namespace) -> int:
    action = args.action

    if action == "set":
        key = ""
        if args.from_stdin:
            key = sys.stdin.readline()
        elif args.from_env:
            key = os.environ.get(args.from_env, "")
            if not key:
                print(f"environment variable {args.from_env} is empty or unset")
                return EXIT_FAILED
        else:
            print("Paste your DeepSeek API key and press Enter (input is hidden).")
            key = getpass.getpass("sk-...: ")
        try:
            tier = secretstore.set_key(key)
        except (ValueError, RuntimeError, OSError) as exc:
            print(f"failed to store the key: {scrub_secrets(str(exc))}")
            return EXIT_FAILED

        stored, _source = secretstore.read_key()
        print(f"Secret stored ({tier}): {rel(secretstore.DPAPI_FILE)}")
        print(f"key={mask_key(stored)}  len={len(stored or '')}")

        if args.no_verify:
            print("skipped verification (--no-verify)")
            return EXIT_OK
        result = secretstore.verify_key(stored)
        if result["ok"]:
            print(f"Verifying with GET /user/balance ... OK (balance {result['balance']})")
            return EXIT_OK
        print(f"Verifying with GET /user/balance ... {result['message']}")
        return EXIT_OK

    if action == "status":
        info = secretstore.status()
        print(f"source={info['source']}  key={info['masked']}  len={info['length']}")
        if info["dpapi_file"]:
            print(f"dpapi={info['dpapi_file']}")
        if info["dotenv_file"]:
            print(f"dotenv={info['dotenv_file']}")
        for note in info["notes"]:  # type: ignore[union-attr]
            print(f"note: {note}")
        return EXIT_OK if info["present"] else EXIT_NO_SECRET

    if action == "verify":
        result = secretstore.verify_key()
        print(f"status={result['status']} ok={result['ok']} balance={result['balance']}")
        print(str(result["message"]))
        return EXIT_OK if result["ok"] else EXIT_FAILED

    if action == "clear":
        removed = secretstore.clear_stored()
        print("DPAPI ciphertext removed." if removed else "No DPAPI ciphertext found.")
        print("Remember: rotating the key on the provider side is the only real revocation.")
        return EXIT_OK

    print(f"unknown secret action: {action}")
    return EXIT_FAILED


# --------------------------------------------------------------------------- #
# selftest
# --------------------------------------------------------------------------- #


def _check(name: str, fn: Callable[[], str]) -> tuple[bool, str, str]:
    try:
        return True, name, fn()
    except Exception as exc:  # noqa: BLE001 - a self-test must never abort the run
        detail = f"{type(exc).__name__}: {scrub_secrets(str(exc))}"
        return False, name, detail


def cmd_selftest(args: argparse.Namespace) -> int:
    checks: list[tuple[str, Callable[[], str]]] = []

    def check_default_config() -> str:
        config = load_config("default.yaml")
        return f"source={config.source.name} top_n={config.get('retrieval.top_n')} top_k={config.get('retrieval.top_k')}"

    def check_all_configs() -> str:
        names = ["pilot.yaml", "full.yaml", "replicate.yaml"]
        names += [f"scan_P{i}.yaml" for i in range(1, 6)]
        loaded = [name for name in names if (repo_path("configs", name)).exists()]
        for name in loaded:
            load_config(name)
        return f"{len(loaded)} experiment configs valid"

    def check_ablation_matrix() -> str:
        specs = load_ablation_matrix()
        expected = len(REQUIRED_ABLATIONS)
        if len(specs) != expected:
            raise RuntimeError(f"expected {expected} ablation rows, found {len(specs)}")
        return f"{len(specs)} rows, all required ids present"

    def check_dirs() -> str:
        created = ensure_dirs()
        missing = [str(d) for d in created if not d.exists()]
        if missing:
            raise RuntimeError(f"directories not created: {missing}")
        return f"{len(created)} generated directories present"

    def check_mask() -> str:
        masked = mask_key("sk-" + "abcdef1234567890" + "wxyz")
        if masked != "sk-******wxyz":
            raise RuntimeError(f"unexpected mask: {masked}")
        if mask_key(None) != "(missing)":
            raise RuntimeError("mask of a missing key must be '(missing)'")
        return masked

    def check_scrub() -> str:
        raw = "Authorization: Bearer sk-" + "A1b2C3d4E5f6G7h8I9j0K1l2"
        scrubbed = scrub_secrets(raw)
        if SECRET_RE.search(scrubbed):
            raise RuntimeError("scrub_secrets left a key-shaped token behind")
        return "key removed from log line"

    def check_config_rejects_secret() -> str:
        probe = repo_path("cache", "_selftest_secret_probe.yaml")
        probe.parent.mkdir(parents=True, exist_ok=True)
        probe.write_text("runtime:\n  boom: " + "sk-" + "0123456789abcdefghij\n", encoding="utf-8")
        try:
            try:
                load_config(probe)
            except ConfigError as exc:
                if "API key" in str(exc):
                    return "config loader rejects embedded keys"
                raise RuntimeError(f"unexpected ConfigError: {exc}") from exc
            raise RuntimeError("config loader accepted a file containing a key")
        finally:
            probe.unlink(missing_ok=True)

    def check_secret_status() -> str:
        info = secretstore.status()
        if "source" not in info:
            raise RuntimeError("secret status did not report a source")
        return f"source={info['source']} present={info['present']}"

    def check_dpapi_roundtrip() -> str:
        if not secretstore.IS_WINDOWS:
            return "skipped (non-Windows host)"
        probe = "sk-" + "roundtrip-0123456789"
        blob = secretstore.dpapi_protect(probe.encode("utf-8"))
        if probe.encode("utf-8") in blob:
            raise RuntimeError("ciphertext contains the plaintext")
        back = secretstore.dpapi_unprotect(blob).decode("utf-8")
        if back != probe:
            raise RuntimeError("DPAPI round-trip mismatch")
        return f"{len(blob)} bytes ciphertext, round-trip equal"

    def check_streamlit() -> str:
        try:
            import streamlit

            version = streamlit.__version__
            major, minor = (int(x) for x in version.split(".")[:2])
            if (major, minor) < (1, 28):
                raise RuntimeError(f"streamlit {version} < 1.28 (AppTest unavailable)")
            return f"streamlit {version}"
        except ImportError:
            raise RuntimeError("streamlit is not installed (frontend is mandatory, D20)") from None

    def check_english_only() -> str:
        offenders = []
        files = _iter_text_files()
        for candidate in files:
            text = _read_text(candidate)
            if text and _cjk_lines(text):
                offenders.append(rel(candidate))
        if offenders:
            raise RuntimeError(f"non-English text in {offenders[:5]}")
        return f"{len(files)} files are English-only"

    checks += [
        ("config: default.yaml loads and validates", check_default_config),
        ("config: experiment configs load", check_all_configs),
        ("config: ablation matrix is complete", check_ablation_matrix),
        ("paths: generated directories exist", check_dirs),
        ("redact: mask_key format", check_mask),
        ("redact: scrub_secrets", check_scrub),
        ("config: embedded API keys rejected", check_config_rejects_secret),
        ("secrets: status resolves a tier", check_secret_status),
        ("secrets: DPAPI round-trip", check_dpapi_roundtrip),
        ("secrets: no key material in the tree", lambda: _lint_secrets_summary()),
        ("env: streamlit importable", check_streamlit),
        ("lint: repository is English-only", check_english_only),
    ]

    total = len(checks)
    failures = 0
    width = max(len(name) for name, _ in checks)
    for index, (name, fn) in enumerate(checks, start=1):
        ok, _name, detail = _check(name, fn)
        status = "PASS" if ok else "FAIL"
        print(f"[{index:2d}/{total}] {name.ljust(width)}  {status}  {detail}")
        if not ok:
            failures += 1

    print()
    if failures:
        print(f"SELFTEST: {total - failures}/{total} PASS, {failures} FAILED")
        return EXIT_FAILED
    print(f"SELFTEST: {total}/{total} PASS")
    return EXIT_OK


def _lint_secrets_summary() -> str:
    findings = _scan_for_secrets()
    if findings:
        raise RuntimeError(f"key-shaped material found in: {findings[:5]}")
    return "no key-shaped material found"


def _scan_for_secrets() -> list[str]:
    findings: list[str] = []
    for candidate in _iter_text_files():
        text = _read_text(candidate)
        if text and SECRET_RE.search(text):
            findings.append(rel(candidate))
    return findings


# --------------------------------------------------------------------------- #
# attack-validity audit (WP2)
# --------------------------------------------------------------------------- #


def cmd_audit(args: argparse.Namespace) -> int:
    from .audit import retrieval_audit

    stage = args.stage
    if stage == "retrieval":
        payload = retrieval_audit.run(seed=args.seed, top_n=args.top_n)
        print(retrieval_audit.summarize(payload))
        return EXIT_OK if payload["gate_r"]["passed"] else EXIT_FAILED

    if stage in {"generation", "baseline", "all"}:
        print(f"audit stage '{stage}' needs the language-model layer and arrives in WP2b.")
        print("Stage R (retrieval) is available now and costs nothing:")
        print("    python run.py audit --stage retrieval --seed 42")
        return EXIT_PLANNED

    print(f"unknown audit stage: {stage}")
    return EXIT_FAILED


# --------------------------------------------------------------------------- #
# data generation (WP1)
# --------------------------------------------------------------------------- #


def cmd_gen_data(args: argparse.Namespace) -> int:
    from .datagen.generator import generate, topic_histogram
    from .datagen.validator import summarize, validate

    seeds = args.seeds or [args.seed]
    exit_code = EXIT_OK
    log_lines: list[str] = []

    for seed in seeds:
        print(f"generating synthetic bundle for seed {seed} ...")
        result = generate(seed)
        counts = result.counts()
        print(
            f"Generated {counts['facts']} facts / {counts['clean_passages']} clean / "
            f"{counts['distractor_passages']} distractor / {counts['poisoned_passages']} poisoned / "
            f"{counts['corpora']} corpora"
        )
        histogram = topic_histogram(result.facts)
        print("topic histogram: " + ", ".join(f"{k}={v}" for k, v in histogram.items()))
        lengths = [row["n_words"] for row in result.clean + result.distractors + result.poisoned]
        print(f"passage words: min={min(lengths)} max={max(lengths)} mean={sum(lengths) / len(lengths):.1f}")
        print(f"output: {rel(result.out_dir)}")

        notes: list[dict] = []
        outcomes = validate(
            result,
            check_determinism=not args.skip_determinism,
            check_seed_cross=not args.skip_cross_seed,
            notes=notes,
        )
        ok, text = summarize(outcomes)
        print(text)
        if not ok:
            exit_code = EXIT_FAILED

        if notes:
            invalid_path = repo_path("reports", f"retrieval_invalid_seed{seed}.json")
            invalid_path.parent.mkdir(parents=True, exist_ok=True)
            invalid_path.write_text(
                json.dumps(notes, indent=2, ensure_ascii=False) + "\n", encoding="utf-8", newline="\n"
            )
            print(f"invalid instances recorded: {rel(invalid_path)} ({len(notes)} rows)")

        log_lines.append(f"=== seed {seed} ===")
        log_lines.append(f"generator_version: 1.0.0")
        log_lines.append(f"counts: {counts}")
        log_lines.append(f"topic_histogram: {histogram}")
        log_lines.append(f"words: min={min(lengths)} max={max(lengths)}")
        for name, digest in sorted(result.sha256.items()):
            log_lines.append(f"sha256 {digest}  {name}")
        log_lines.append(text)
        log_lines.append("")

    log_path = repo_path("reports", "gen_log.txt")
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8", newline="\n") as handle:
        handle.write("\n".join(log_lines) + "\n")
    print(f"validator log: {rel(log_path)}")
    return exit_code


# --------------------------------------------------------------------------- #
# linters
# --------------------------------------------------------------------------- #


def cmd_lint_lang(args: argparse.Namespace) -> int:
    violations = 0
    for candidate in _iter_text_files():
        text = _read_text(candidate)
        if not text:
            continue
        for lineno in _cjk_lines(text):
            print(f"{rel(candidate)}:{lineno}: non-English text")
            violations += 1
    if violations:
        print(f"LINT-LANG: FAIL ({violations} violations)")
        return EXIT_FAILED
    print("LINT-LANG: PASS")
    return EXIT_OK


def _find_git() -> str | None:
    """Locate the git executable.

    ``shutil.which`` is tried first; on Windows the standard install locations
    are probed as well, because a long-lived parent process (an IDE, a shell
    started before the install) keeps a stale PATH and would otherwise silently
    skip the history scan.
    """
    found = shutil.which("git")
    if found:
        return found
    if sys.platform == "win32":
        candidates = [
            r"C:\Program Files\Git\cmd\git.exe",
            r"C:\Program Files (x86)\Git\cmd\git.exe",
            os.path.join(os.environ.get("LOCALAPPDATA", ""), "Programs", "Git", "cmd", "git.exe"),
        ]
        for candidate in candidates:
            if Path(candidate).exists():
                return candidate
    return None


def cmd_lint_secrets(args: argparse.Namespace) -> int:
    findings = _scan_for_secrets()
    if findings:
        for item in findings:
            print(f"{item}: key-shaped material")
        print(f"LINT-SECRETS: FAIL ({len(findings)} files)")
        return EXIT_FAILED

    if not (REPO_ROOT / ".git").exists():
        print("LINT-SECRETS: PASS (working tree only; not a git repository)")
        return EXIT_OK

    git = _find_git()
    if git is None:
        print("warning: git not found, so repository history was NOT scanned")
        print("LINT-SECRETS: PASS (working tree only)")
        return EXIT_OK

    try:
        counted = subprocess.run(
            [git, "rev-list", "--count", "--all"],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            timeout=120,
            check=False,
        )
        commits = int((counted.stdout or "0").strip() or 0)
        if commits == 0:
            print("LINT-SECRETS: PASS (working tree; the repository has no commits yet)")
            return EXIT_OK

        history = subprocess.run(
            [git, "log", "-p", "--all"],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            timeout=180,
            check=False,
            encoding="utf-8",
            errors="replace",
        )
        if SECRET_RE.search(history.stdout or ""):
            print("git history contains key-shaped material; rotate the key and rewrite history")
            print("LINT-SECRETS: FAIL (history)")
            return EXIT_FAILED
        print(f"history scanned: {commits} commit(s)")
    except (OSError, subprocess.SubprocessError, ValueError) as exc:
        print(f"warning: could not scan git history ({exc})")

    print("LINT-SECRETS: PASS")
    return EXIT_OK


# --------------------------------------------------------------------------- #
# config inspection
# --------------------------------------------------------------------------- #


def cmd_config(args: argparse.Namespace) -> int:
    try:
        config = load_config(args.config)
    except ConfigError as exc:
        print(f"config error: {exc}")
        return EXIT_FAILED
    print(render_config(config))
    return EXIT_OK


# --------------------------------------------------------------------------- #
# argument parsing
# --------------------------------------------------------------------------- #


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="run.py",
        description=(
            "RAG poisoning detection: multi-signal judgement plus a budgeted "
            "periodic corpus audit."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Work packages:\n"
            "  WP0 skeleton/secret  WP1 data  WP2 audit  WP3 retrieval+local signals\n"
            "  WP4 LLM layer        WP5 pilot  WP6 evaluation  WP7 main experiment\n"
            "  WP8 dashboard        WP9 corpus audit  WP10 report"
        ),
    )
    parser.add_argument("--version", action="version", version=f"ragpoison {__version__}")
    subparsers = parser.add_subparsers(dest="command", metavar="<command>")

    secret = subparsers.add_parser("secret", help="manage the DeepSeek API key")
    secret_sub = secret.add_subparsers(dest="action", metavar="<action>", required=True)
    set_parser = secret_sub.add_parser("set", help="store the key (hidden input by default)")
    set_parser.add_argument("--from-stdin", action="store_true", help="read the key from stdin")
    set_parser.add_argument("--from-env", metavar="VAR", help="read the key from an environment variable")
    set_parser.add_argument("--no-verify", action="store_true", help="skip the balance check")
    set_parser.set_defaults(func=cmd_secret)
    secret_sub.add_parser("status", help="show the resolved tier and the masked key").set_defaults(
        func=cmd_secret
    )
    secret_sub.add_parser("verify", help="validate the key against the free balance endpoint").set_defaults(
        func=cmd_secret
    )
    secret_sub.add_parser("clear", help="delete the stored ciphertext").set_defaults(func=cmd_secret)

    subparsers.add_parser("selftest", help="run offline self-checks (no API, no network)").set_defaults(
        func=cmd_selftest
    )
    subparsers.add_parser("lint-lang", help="reject non-English text in the repository").set_defaults(
        func=cmd_lint_lang
    )
    subparsers.add_parser("lint-secrets", help="reject key material in the tree or git history").set_defaults(
        func=cmd_lint_secrets
    )

    config_parser = subparsers.add_parser("config", help="print the effective configuration")
    config_parser.add_argument("--config", default="default.yaml", help="configuration file name or path")
    config_parser.set_defaults(func=cmd_config)

    gen = subparsers.add_parser("gen-data", help="generate the deterministic synthetic benchmark")
    gen.add_argument("--seed", type=int, default=42, help="primary seed (default 42)")
    gen.add_argument("--seeds", type=int, nargs="+", help="generate several seeds in one run")
    gen.add_argument(
        "--skip-determinism",
        action="store_true",
        help="skip V18 (regenerate and compare bytes); faster during template work",
    )
    gen.add_argument(
        "--skip-cross-seed",
        action="store_true",
        help="skip V19 (compare against the other seed); faster during template work",
    )
    gen.set_defaults(func=cmd_gen_data)

    audit = subparsers.add_parser("audit", help="attack-validity audit (stage R is offline)")
    audit.add_argument(
        "--stage",
        default="retrieval",
        choices=["retrieval", "generation", "baseline", "all"],
        help="retrieval (free) | generation (API) | baseline (API) | all",
    )
    audit.add_argument("--seed", type=int, default=42, help="bundle seed (default 42)")
    audit.add_argument("--top-n", type=int, default=None, help="override the detection pool size")
    audit.set_defaults(func=cmd_audit)

    for command in PLANNED:
        stub = subparsers.add_parser(command, help=f"not implemented yet ({PLANNED[command]})")
        stub.add_argument("--config", default=f"{command}.yaml", help=argparse.SUPPRESS)
        stub.set_defaults(func=lambda ns, cmd=command: _print_planned(cmd))

    return parser


def _first_command_token(argv: list[str]) -> str | None:
    """Return the first non-option token, i.e. the subcommand name."""
    for token in argv:
        if token.startswith("-"):
            if token in {"-h", "--help", "--version"}:
                return None
            continue
        return token
    return None


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)

    # A command that is not implemented yet must explain itself. Going through
    # argparse first produces "unrecognized arguments" for any flag the stub does
    # not declare, which reads like a bug in the invocation rather than a missing
    # work package.
    planned = _first_command_token(argv)
    if planned in PLANNED:
        return _print_planned(planned)

    parser = build_parser()
    args = parser.parse_args(argv)
    if not getattr(args, "command", None):
        parser.print_help()
        return EXIT_OK
    handler: Callable[[argparse.Namespace], int] = getattr(args, "func", None)  # type: ignore[assignment]
    if handler is None:  # pragma: no cover - argparse guarantees a handler
        parser.print_help()
        return EXIT_FAILED
    try:
        return handler(args)
    except KeyboardInterrupt:
        print("\ninterrupted")
        return EXIT_FAILED
    except Exception:  # noqa: BLE001 - never show a bare traceback to a student
        print("unexpected error:")
        print(scrub_secrets(traceback.format_exc()))
        return EXIT_FAILED
