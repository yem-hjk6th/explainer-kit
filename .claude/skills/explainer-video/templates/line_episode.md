# Line {LINE}: episode {ID} "{TITLE}"

Read `{PROJECT_ROOT}/_prompts/COMMON.md` first, then `references/PIPELINE.md`, `references/STYLE.md` (+ `STYLE_local.md`),
`references/STORYBOARD.md`. Your folder: `episodes/{ID}_{topic}/` (write only here).

## Why this episode
{2–4 lines: what it covers, why now, which building blocks it shares with earlier episodes}

## Materials (read-only; digest first, then line ranges)
- {source 1: paper / notes + digest path}
- {source 2}
- Model example to imitate (read-only): `episodes_example/{closest episode}/storyboard.md`, `shots/{closest shot}.py`.

## Stage 1: storyboard (no rendering, no installs)
`storyboard.md`, `CLAIMS_to_verify.md`, `episode.json`, `elements_needed.md` (only what the library lacks).
Length {MIN–MAX} min, {N} shots. Must cover: {list of mechanisms in order}. Audience: {who, what they know}.
Our own test scene: {synthetic scene with ground truth / reuse of an earlier scene}; introduce it as our test.
Real model runs (optional): in §0 list which shots, repo + commit, weights + size + license, new env. Do not install.
When done: message the coordinator (paths + ≤ 8 lines). It launches the verifier and asks the reviewer.

## Stage 2: shots (after verification round 1)
Apply the verifier's corrections (`~~old~~ → new`). One script per shot in `shots/`, checks for every computed number
(including the displayed string), sheets inspected, full renders. Write all on-screen text through animkit entry
points (I18N-ready). Missing library feature → ask the coordinator for the library line; temporary local code in your
shot is fine, report it. Then `concat.py`, `review_page.py --round 1`, message the coordinator.

## Stage 3: review rounds
The coordinator forwards the reviewer's notes (`review/roundN/feedback.md`). Answer each; new claims → CLAIMS (new
section) → tell the coordinator before rendering; `changes.md`; re-render; next round page.

## Cost discipline
Main session ≤ 300k: before that write `HANDOFF_{ID}_<stage>.md` and tell the coordinator. Sub-agents: one shot each
(`templates/shot_agent_brief.md`), ≤ 150k, discarded after; rendering / translation sub-agents may use a cheaper model.

## Compute
{CORES / MASK / processes / GPU, or "see COMMON"}
