"""Scan a tree (default: this repo) for things that should not be shared: absolute paths, home directories,
e-mail addresses, session / agent ids, GPU model numbers, plus any personal terms you list in an external rules file.

  python -X utf8 tools/sanitize_scan.py [ROOT] [--rules FILE] [--git] [--quiet]

Exit code 0 = no hits, 1 = hits (printed as path:line: [rule] text).

Personal terms (your name, collaborators, user name, host names, course codes, account names...) do NOT belong in
this repo, so they live in a rules file outside it: one rule per line, `label<TAB>regex` or just `regex`; lines
starting with # are comments. Pass it with --rules or the SANITIZE_RULES environment variable.
--git also scans commit authors / committers / messages (git log) of ROOT.
"""
import argparse
import os
import re
import subprocess
import sys
from pathlib import Path

# Built-in generic rules. Written so that this file does not match itself.
GENERIC = [
    ("windows-abs-path", r"\b[A-Za-z]:[\\/](?![sdwbSDWB](?:[^A-Za-z]|$))[^\s\"'<>|]"),   # not regex escapes like I:\s
    ("posix-home", r"(?<![\w.])/(?:Users|home)/[A-Za-z0-9_.-]+"),
    ("gitbash-home", r"(?<![\w.])/[a-z]/Users/"),
    ("conda-env-abs", r"[\\/]\.conda[\\/]envs[\\/]"),
    ("email", r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9-]+(?:\.[A-Za-z0-9-]+)*\.[A-Za-z]{2,}\b"),
    ("uuid", r"\b[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}\b"),
    ("hex8-id", r"(?<![0-9A-Za-z_#./-])(?=[0-9a-f]{8}(?![0-9A-Za-z_]))(?=[0-9a-f]*[a-f])(?=[0-9a-f]*[0-9])[0-9a-f]{8}(?![0-9A-Za-z_])"),
    ("claude-session", r"session_[0-9A-Za-z]{12,}|claude\.ai/code/"),
    ("gpu-model", r"\b(?:R[T]X|G[T]X)\s?\d{3,4}\b"),   # local GPU model; papers' training hardware is fine
]

TEXT_EXT = {".py", ".md", ".json", ".txt", ".html", ".csv", ".yaml", ".yml", ".toml", ".cfg", ".ini", ".ps1", ".sh",
            ".bat", ".js", ".css", ".srt", ".gitignore", ""}
SKIP_DIRS = {".git", "__pycache__", ".cache", "render", "_cache", "third_party", "weights", "audio"}


def load_rules(path):
    rules = []
    for ln in Path(path).read_text(encoding="utf-8").splitlines():
        if not ln.strip() or ln.lstrip().startswith("#"):
            continue
        label, _, rx = ln.partition("\t")
        if not rx:
            label, rx = "custom", label
        rules.append((label.strip(), rx.strip()))
    return rules


def iter_files(root):
    """In a git work tree: exactly the files git would commit (tracked + untracked, minus .gitignore'd ones).
    Otherwise: a walk that skips caches / outputs and the machine-local config."""
    if (Path(root) / ".git").exists():
        out = subprocess.run(["git", "-C", str(root), "ls-files", "-co", "--exclude-standard", "-z"],
                             capture_output=True, text=True, encoding="utf-8").stdout
        for rel in filter(None, out.split("\0")):
            p = Path(root) / rel
            if p.is_file() and (p.suffix.lower() in TEXT_EXT or p.name in {".gitignore", ".gitattributes"}):
                yield p
        return
    for dp, dns, fns in os.walk(root):
        if Path(dp) == Path(root) / "config":
            fns = [f for f in fns if f != "animkit.json"]
        dns[:] = [d for d in dns if d not in SKIP_DIRS]
        for fn in fns:
            p = Path(dp) / fn
            if p.suffix.lower() in TEXT_EXT or p.name in {".gitignore", ".gitattributes"}:
                yield p


def scan_text(text, rules):
    for i, line in enumerate(text.splitlines(), 1):
        for label, rx in rules:
            m = rx.search(line)
            if m:
                yield i, label, line.strip()[:200], m.group(0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("root", nargs="?", default=str(Path(__file__).resolve().parents[1]))
    ap.add_argument("--rules", default=os.environ.get("SANITIZE_RULES"))
    ap.add_argument("--git", action="store_true")
    ap.add_argument("--quiet", action="store_true", help="only the summary line")
    a = ap.parse_args()
    rules = list(GENERIC)
    if a.rules:
        rules += load_rules(a.rules)
    else:
        print("note: no --rules file given; only the generic rules run (add your personal terms to a rules file)")
    rules = [(lb, re.compile(rx)) for lb, rx in rules]
    root = Path(a.root)
    hits = 0
    for p in iter_files(root):
        try:
            text = p.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            text = p.read_text(encoding="utf-8", errors="replace")
        for i, label, line, m in scan_text(text, rules):
            hits += 1
            if not a.quiet:
                print(f"{p.relative_to(root)}:{i}: [{label}] {m!r} :: {line}")
    if a.git and (root / ".git").exists():
        def git_log(fmt):
            return subprocess.run(["git", "-C", str(root), "log", "--all", f"--format={fmt}"],
                                  capture_output=True, text=True, encoding="utf-8").stdout

        log = git_log("%an | %cn | %B")
        for i, label, line, m in scan_text(log, [r for r in rules if r[0] != "email"]):
            hits += 1
            if not a.quiet:
                print(f"git-log:{i}: [{label}] {m!r} :: {line}")
        emails = set(git_log("%ae%n%ce").split())     # author / committer identities (trailers are not checked)
        for e in sorted(emails):
            if not e.endswith((".invalid", "@users.noreply.github.com")):
                hits += 1
                print(f"git-log: [email] {e!r} (commit as yourself with your GitHub noreply address, "
                      f"<id>+<user>@users.noreply.github.com, so authorship stays yours without exposing an email)")
    print(f"sanitize_scan: {hits} hit(s) in {root}")
    sys.exit(1 if hits else 0)


if __name__ == "__main__":
    main()
