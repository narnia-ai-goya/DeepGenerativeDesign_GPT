# `docs/` — pipeline structure

What each stage consumes, what it writes, and which later stage reads it. Read this
before the per-script documentation: it is the map the rest of the package assumes.

| File | Contents |
|---|---|
| `pipeline_flow.png` | The whole pipeline as one figure — five stage cards, each **input → process → output**, plus the rail on the right showing that Stage 0's outputs are read by three later stages. |
| `pipeline_flow.html` | Same content in a browsable page, and it additionally carries the **file-lineage table**: for every artifact, which stage made it, which stage reads it, and why that version. Open it in a browser. |
| `draw_pipeline_flow.py` | Regenerates the PNG. Edit the `STAGES` list and re-run; card heights are computed from the content, so the figure resizes itself and nothing overflows. |

The labels are in Korean. Everything they name — script names, config keys, file names —
matches the code verbatim, so the figure is usable alongside the English documentation in
`../README.md`, `../code/README.md` and `../configs/README.md`.

## Regenerating the figure

```bash
python docs/draw_pipeline_flow.py            # writes docs/pipeline_flow.png
```

matplotlib is the only requirement (present in the generation environment). A
Korean-capable font is also needed; the script looks for one in the usual places and,
failing that, prints a warning and renders the Korean text as boxes. Point it at a font
explicitly with:

```bash
KR_FONT=/path/to/NotoSansKR.ttf python docs/draw_pipeline_flow.py
```

`fonts-nanum` or `fonts-noto-cjk` from the distribution's package manager also satisfies it.

## The three things the figure is there to explain

1. **The two entry points meet at one directory.** `run_from_text.py` produces the six
   conditioning views into `data/<domain>/conditioning/<style>/`; `run_from_image.py`
   just reads them. Everything after that point is one identical code path.

2. **Stage 0's outputs are read by every later stage.** `run_prep.py` runs once per
   domain, and generation, post-processing and verification all read what it wrote —
   which is why a run never re-derives the design domain.

3. **The same geometry exists in two versions, and the choice per consumer is
   deliberate.** The voxel grid and the in-loop FEM domain use the **raw** STL; the
   post-processing booleans and both FEA stages' boundary surfaces use the **remeshed**
   one. Each choice has a measured reason, given in the lineage table.
