# E00 animkit self-test set (storyboard format example)

review_page.py finds a shot's section by its number: the section whose heading contains `S01`, up to the next heading of the same level.

## S01 Epipolar geometry (regression) · M · 16 s
- Captions: see `epipolar.py`.
- Picture: two pinhole cameras; the left-image pixel $x_1$ is back-projected into a ray, a point slides along the ray, and the right image draws the epipolar line; epipolar plane, baseline, epipole $e_2$; the epipolar lines of three new rays meet at $e_2$.
- Formula: $x_2^{\mathsf T} F x_1 = 0$, lit up term by term.
- Computed: $F = K_2^{-\top}[t]_\times R K_1^{-1}$; the distance from each rendered epipolar point to $F x_1$ is <= 1e-9 px (measured about 1e-13).

## S02 Tensor elements self-test · M · 24 s
| Time | Caption | Picture |
|---|---|---|
| 0.5–4.5 | S images, N tokens each | tensor block $S\times N\times C$ splits along $S$ into three slices → three rows of tokens |
| 4.5–8.0 | frame attention: within one frame | each row gets a frame, flashing row by row |
| 8.0–12.0 | reshape into one long sequence | $(S,N,C)\to(1,S\cdot N,C)$, tokens move by C-order index |
| 12.0–17.0 | global attention: sees across frames | two heatmaps appear row by row: block-diagonal vs full matrix |
| 17.0–20.0 | each row sums to 1 after softmax | row 0 highlighted, readout of $\sum_j A_{0j}$ |
| 20.0–24.0 | The two attentions alternate | module boxes + arrows + pulse |

- Schematic: $S=3$, $N=6$ are chosen to be readable; corner tag "schematic: S=3, N=6".

## S03 P0 elements self-test · L · 20 s
- A (0–6.5 s): $P=R^{\mathsf T}(D\,K^{-1}[u,v,1]^{\mathsf T}-t)$ highlighted term by term with notes below; counter 1→24; vector $X^\star$ meters → normalized; boxes around the elements of the $K$ matrix; tensor block $(3,1374,1024)\to(1,4122,1024)$.
- B (6.5–13 s): 1374 tokens shown folded, with token 799 forced visible; special token of frame 1 recolored; color band on row 3; scale token at the row end; $(3,4)\to(1,12)$ rearrangement; a single flying token "frame 799 → global $1\times1374+799=2173$"; the Aggregator forks to four heads, with a pulse.
- C (13–20 s): synthetic scene (sphere + ground) with the frustums and thumbnails of three cameras; depth of camera 1 back-projected into about 60k points; ray fan, coordinate axes, $X^\star$ marker and measuring line; image panel + 14 px grid scan + vertical ruler; depth map + color bar.
- Computed: pixel residual of back-project then re-project <= 1e-9; reshape landing = numpy rearrangement; $1\times1374+799=2173$.

## S04 P1 elements self-test · M · 20 s
- A: frame / global attention blocks ($N=3$, $K=4$); single-row softmax bars: with two keys $(e/(1+e),\,1/(1+e)) = (0.731, 0.269)$.
- B: bar chart (lower is better, small source text, values are made up); multi-scale grid $148^2/74^2/37^2/19^2$ appears level by level.
- C: text paper card (example); two-column comparison table lit up row by row.
- D: optional inputs (dashed) randomly switched, required inputs (solid); split screen.
- Computed: reshape landing = indices after numpy reshape; attention = softmax($QK^{\mathsf T}/\sqrt d$), row sum = 1 (error <= 1e-12).
