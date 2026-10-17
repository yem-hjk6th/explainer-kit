# Line {LINE}: {LANG} version of episode {ID}

Read `{PROJECT_ROOT}/_prompts/COMMON.md`, `references/I18N.md`, `references/STYLE.md` §A + §E. Your folder:
`episodes/{ID}_{topic}/` but only `captions.json`, `i18n_tools/`, `render/*_{LANG}.*` and `shots/render/*/*_{LANG}.*`.
Do not change shot scripts (if a string cannot fit after shortening, report it to the coordinator).

1. `tools/i18n_table.py episodes/{ID}_{topic} --lang {LANG} --jobs {P}` → `captions.json`; list strings that differ only in numbers.
2. Glossary first (`i18n_tools/glossary_{LANG}.md`): every term and symbol name once, fixed.
3. Translate per shot (sub-agents per shot are fine, cheaper model OK): same rules as the master — plain language,
   symbols defined, metric / unit / direction stated; captions ≤ 12 words. No new facts.
4. First render `--lang {LANG}`; compute per-box character budgets (`en_fit.py`); shorten everything over budget in
   one pass; re-render; look at the densest frames of each shot (stills tiled small), fix collisions by rewording.
5. `concat.py --lang {LANG}`; review page for the translated render (see REVIEW.md); message the coordinator: counts
   (strings, untranslated 0, off-canvas 0), paths, what you could not fit.
A verifier then checks the translation against the verified master.
