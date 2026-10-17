# HANDOFF {ID} — {stage} ({date from `date`})

Written by the outgoing session before ~300k context. The next session reads this + STATUS.md, nothing else first.

## State
- Approved by the reviewer: {shots / rounds}. Current round: {N}. Current video: `render/{ID}.mp4` ({length} s, {time}).
- Verification: rounds done {list}; fixes applied {yes/no per report}; fixes not re-verified: {list}.

## File map (authoritative file per concern)
| Concern | File |
|---|---|
| captions / visuals | `storyboard.md` §… |
| numbers | `shots/{ep}_numbers.py` (printed table in storyboard §0) |
| claims | `CLAIMS_to_verify.md` (correction table on top) |
| translations | `captions.json` |
| reviewer notes | `review/roundN/feedback.md` |

## Per shot
| Shot | Script | Depends on | Status | Open items |
|---|---|---|---|---|
| {ID}-01 | `shots/s01_….py` | `{ep}_common.py`, model output X | rendered, checks pass | — |

## Commands (current compute share)
```
{exact render / concat / review commands with mask and threads}
```

## Open
- Reviewer notes not yet handled: {list with ids}
- Questions waiting for the reviewer: {list}
- Do not touch: {files / approved shots / other lines' folders}
