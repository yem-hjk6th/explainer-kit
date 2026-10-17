# Line {LINE}: animkit library work

Read `{PROJECT_ROOT}/_prompts/COMMON.md`, `animkit/README.md`, `references/STYLE.md` §C, `references/PITFALLS.md`.
Your folder: `animkit/`, `tools/`, `examples/` — but **develop in a dev copy** `{DEV_DIR}` (e.g. `_dev/`), because
episode lines may be rendering with the real files.

## Requests
{From elements_needed.md of episode {ID}: element name, input parameters, what it must look like, priority order.}

## Rules
- Run the dev copy with `ANIMKIT_CONFIG={REPO_ROOT}/config/animkit.json` and `ANIMKIT_CACHE={REPO_ROOT}/.cache`
  (they are not PYTHON* variables, so `-I` keeps them; the formula cache is shared).
- Each new element: an example shot in `examples/` with checks; API section in `animkit/README.md` (signature, what
  every parameter does, return values usable in checks); `tools/selftest.py` still all OK.
- Prove you changed nothing else: A/B render the same frames of a few existing shots with old and new library
  (pixel difference 0) — e.g. a source / translated-render regression or the episode shots that use the touched module.
- Library computes nothing it cannot verify: values come from the caller; the library may return geometry for checks.
- Install: ask the coordinator to confirm no render is running; then temp file + `os.replace`, new modules first,
  `__init__.py` last; run `selftest.py` again; append to the issues file.
- Report: element list, example paths, self-test output, A/B result, install time.

## Compute
{CORES / MASK / processes}
