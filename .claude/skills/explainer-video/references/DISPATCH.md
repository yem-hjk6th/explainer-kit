# DISPATCH: splitting work across sessions and sub-agents

A 20–60 s explainer fits in one session. A 5–8 minute episode with review rounds, a translated version and a
voice-over usually does not: split it into "lines" (one session each, one job each) plus one coordinator.

## Roles

| Line | Owns (writes only here) | Job |
|---|---|---|
| coordinator | a dispatch board file | starts lines, relays files and the reviewer's answers, launches independent verifiers, spot-checks, keeps compute shares |
| library | `animkit/` (via a dev copy) | new visual elements requested by episodes, self-tests, docs |
| episode line | `episodes/<ID>_<topic>/` | storyboard → CLAIMS → shots → review page; later rounds in fresh sessions |
| translation | `captions.json` of one episode | collect, translate (sub-agents per shot), fit, render `--lang <code>` |
| narration | `<EP>/.../narration/` | spoken script + speaker notes; writes only text |
| voice-over | `<EP>/.../voiced/` | TTS generate → pick → time → render voiced variant → mix → self-check report |
| verifier (sub-agent) | `verify/<ID>/` | independent checks; never edits the episode |

Templates for each line's brief: `templates/line_*.md`; shared rules every brief points to: `templates/COMMON.md`.

## Protocol (put it in COMMON.md for every line)

1. Write only your own folder; everything else is read-only. Never delete files: move them to `_moved/`.
2. Keep a short `STATUS.md`: stage / next step / waiting for whom / outputs for other lines (with paths) / questions
   for the reviewer / time (from `date`, never estimated).
3. Facts and numbers go into `CLAIMS_to_verify.md` with source, how to check, and measured / computed / inferred.
   The coordinator launches independent verification; corrections are applied as `~~old~~ → new`.
4. Log effort in `effort_log.csv` (`item,size,start,end,render_s,iterations,note`).
5. Questions for the reviewer and anything for another line go to the coordinator, not directly. Report each stage
   end in ≤ 8 lines: what was done, file paths, next step, what you are waiting for.
6. Tool pitfalls go to one shared issues file (read it first; append, do not duplicate; ≤ 10 lines per entry).

## Context and cost

Context, not render time, is the main cost: a session that never compacts re-reads its whole history on every turn,
and a long-lived sub-agent grows the same way. Rules of thumb:

- Keep a main session under roughly 300k tokens of context. Before that, write `HANDOFF_<ID>_<stage>.md` (see the
  checklist below) and continue in a fresh session that reads only the handoff + STATUS.
- A sub-agent gets **one shot** (or one narrow task) and is discarded when done; keep it under roughly 150k. Tell it
  what to read and what not (the API section only; grep the template shot, then read line ranges; at most 3
  contact-sheet iterations).
- Execution work (rendering a specified shot, translating a table, applying a worded fix list) can use a cheaper
  model; content judgement and verification stay on the strongest model.
- Budget an episode as "first pass + 2 review rounds": a review round costs about as much as a first pass.
- The reviewer's watching time is usually the bottleneck, not agent time or render time: batch what you send.

## Library changes while episodes render

A shot that imports a half-written library module fails with a missing attribute or name. So the library line works
in a dev copy (`ANIMKIT_CONFIG` / `ANIMKIT_CACHE` pointed back at the real ones), runs the self-tests plus A/B pixel
comparisons on existing shots, asks the coordinator to confirm no render is running, then installs with "write temp
file + `os.replace`", new modules first, the `__init__.py` that imports them last. A shot line that sees an attribute
/ name error inside `animkit` first checks the library files' modification times and waits instead of "fixing" the
library.

## Shared-machine etiquette

- Give each line an explicit CPU share (core mask + thread count) and a render-process limit; leave cores for the
  desktop. Higher-priority work on the same machine wins: when the coordinator says so, drop to the minimum share and
  stop GPU use at once.
- GPU: check `nvidia-smi` before starting; cap memory per process; one GPU job at a time; release when done. Do not
  trust a single CPU-load sample (a 5 s sample can read 100 % while the real load is a few cores).
- Re-check shares on a schedule (e.g. hourly), not continuously.

## Handoff checklist (what a fresh session needs)

Version and what was approved; file map (which file is authoritative for what); per-shot status and dependencies;
exact commands with the current CPU share; open review notes and which are answered; verifier reports and which fixes
are applied / not re-verified; what not to touch.
