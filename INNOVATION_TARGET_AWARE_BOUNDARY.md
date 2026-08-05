# TABDS innovation note

## Evidence from the project and paper

The supplied paper, *SP-KAN: Sparse-sine perception Kolmogorov–Arnold networks
for infrared small target detection* (ISPRS JPRS 234, 2026, DOI
10.1016/j.isprsjprs.2026.02.019), defines the training objective in Section 4.1,
Eq. (7) as equally weighted BCE terms for the five deep-supervision outputs and
the fused output. The code mirrors that design in `util/train_helpers.py`.

The paper's Section 5.9 (p. 16) reports incomplete contours for oversized targets
and false alarms around blind-pixel clusters, and explicitly lists stronger edge
representations as future work. These observations motivate a loss-level
extension that does not change the SP-KAN checkpoint architecture.

## Proposed research question

Does foreground-ratio adaptive, boundary-aware deep supervision improve IoU,
F1, probability of detection (Pd), and false-alarm rate (Fa) over the paper's
unweighted BCE objective, especially for few-pixel targets and cluttered edges?

## Implementation

The `innovation/target-aware-boundary-loss` branch adds
`util/losses.py::TargetAwareBoundaryLoss`:

1. Compute a per-batch positive/negative pixel ratio, clipped by
   `max_pos_weight`.
2. Compute a 3x3 morphological gradient of the mask and multiply its BCE weight
   by `1 + boundary_weight`.
3. Add a soft Dice term with `dice_weight`.
4. Apply normalized weights to the six SP-KAN deep-supervision predictions.

The baseline remains selectable with `--loss_name bce`. The runnable experiment
entry point is `scripts/run_target_aware.sh`; all hyperparameters are recorded in
the normal `train_config.json` artifact.

## Suggested validation protocol

Run the innovation and baseline with the same seed, split, epoch budget, and
checkpoint selection on SIRST V1, NUDT-SIRST, IRSTD-1K, and SIRST3. Report mean
and standard deviation over at least three seeds, plus a target-area bucket
breakdown (1–5, 6–20, >20 pixels). Compare IoU, F1, Pd, and Fa at the same
threshold. The branch is an implementation of a testable hypothesis; no gain is
claimed until these controlled experiments are run.
