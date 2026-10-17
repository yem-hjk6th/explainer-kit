# Line {LINE}: episode {ID}, review round {N} (takes over from a previous session)

Read, in order:
1. `{PROJECT_ROOT}/_prompts/COMMON.md` (compute share below overrides it)
2. `episodes/{ID}_{topic}/HANDOFF_{ID}_{stage}.md` (version, file map, per-shot dependencies, commands)
3. `episodes/{ID}_{topic}/review/round{N}/feedback.md` (**the reviewer's notes, verbatim**)
4. `references/STYLE.md` (§E grows with every round) + `episodes/STYLE_local.md`
Your folder: `episodes/{ID}_{topic}/`.

## Notes, one by one (restate each; write "how I changed it" per note in STATUS.md)
- {ID}-01: {note, restated precisely; what to do; what to keep}
- {ID}-02: ...
- General: {notes that apply to all shots}

## Requirements
- Everything newly explained (definitions, numbers, structures) → a new CLAIMS section; message the coordinator
  before rendering (verification). You may work on visuals meanwhile.
- Keep what the reviewer praised. Length may grow; propose cuts only with a reason each.
- Then: re-render affected shots → `concat.py` → `review/round{N+1}/changes.md` → `review_page.py --round {N+1}` →
  message the coordinator.
- Context ≤ ~300k; if you get close, write `HANDOFF_{ID}_round{N}b.md` and tell the coordinator.

## Compute
{CORES / MASK / processes; GPU none unless asked}
