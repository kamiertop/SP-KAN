# SP-KAN infrared-specific innovation catalog

## Material Passport

- **Type**: Research innovation candidates and experiment blueprint
- **Source**: Supplied SP-KAN paper, repository implementation, and SIRST3 data layout
- **Status**: Hypotheses; no performance gain is claimed before controlled experiments
- **Current branch**: `innovation/target-aware-boundary-loss`
- **Existing implementation**: TABDS target-aware boundary deep supervision
- **Literature mapping**: See `LITERATURE_STITCHING_REPORT.md` for verified
  CVPR/NeurIPS/ECCV/TGRS abstract evidence and concrete module seams.

## What the paper and repository leave open

The paper reports three directly actionable gaps:

1. Section 5.9 (p. 16) identifies incomplete contours and blind-pixel false
   alarms, and lists explicit edge representations as future work.
2. Section 5.5 evaluates Gaussian, stripe, and mixed noise only as a test-time
   robustness study. The training pipeline has no paired noisy-view consistency
   objective.
3. Section 5.7 studies target sizes from roughly 2 to 137 pixels, but the model
   has no explicit target-scale conditioning.

There is also an implementation-fidelity opportunity: the paper's Section 5.1
settings select SPKAL grid size 12, while `model/SP_KAN.py::KANLayer` hardcodes
`grid_size = 5`. This should be treated as a controlled fidelity branch before
calling it a novel architectural gain.

A read-only scan of the available training lists also supports the scale and
imbalance hypotheses: component-area medians are about 26 pixels on SIRST3,
21 on NUDT-SIRST, and 32 on IRSTD-1K, while the largest components exceed 1,000
pixels. A single fixed receptive-field or loss weighting is therefore unlikely
to be optimal across the entire corpus.

## Ranked candidate innovations

### 1. Local-Contrast Pyramid Gate (LCPG)

**Hypothesis**: IR targets are sparse local radiometric anomalies over a smooth
or self-similar background; multi-scale center-surround residuals can improve
target/background separation before the decoder loses detail.

**Module**: At `e2/e3/e4` skip features, compute depthwise local means with 3, 7,
and 15 pixel kernels, form positive center-surround residuals, and use a small
pointwise/depthwise gate to modulate the skip feature:

`Y = X * (1 + sigmoid(Conv([X, ReLU(X-M3), ReLU(X-M7), ReLU(X-M15)])))`.

**Branch**: `innovation/local-contrast-pyramid-gate`

**Ablation**: no gate, single-scale gate, three-scale gate; report IoU/F1/Pd/Fa,
parameter count, and performance by target area.

**Risk**: Raw contrast can amplify hot clutter. Use feature-space residuals and
retain the identity path so the module cannot erase context.

### 2. Contour-Guided Decoder (CGD)

**Hypothesis**: The paper's explicit-edge limitation can be addressed more
directly than with a loss alone by letting decoder features predict and inject a
contour map at every upsampling stage.

**Module**: Add a 1-channel contour head after `d5/d4/d3/d2`; feed the resized
contour feature into the next decoder block through a zero-initialized 1x1
projection. Supervise contours generated from the GT morphological gradient.

**Branch**: `innovation/contour-guided-decoder`

**Ablation**: contour loss only vs. contour injection only vs. both; measure
boundary F-score in addition to the paper's four metrics.

**Risk**: Over-sharpening can break larger, blurred targets. Use a soft boundary
band (not a one-pixel skeleton) and retain the segmentation residual path.

### 3. Noise-Consistent Dual-View Training (NCDV)

**Hypothesis**: Robustness reported under synthetic Gaussian/stripe noise should
be learned, not only evaluated. A clean/noisy pair with the same mask can teach
the model to preserve target responses and suppress noise-specific activations.

**Objective**:

`L = L_seg(clean) + L_seg(noisy) + lambda * JS(p_clean, stopgrad(p_noisy))`.

The noisy view samples Gaussian noise, row/column stripes, and sparse dead pixels
after normalization. Apply the consistency term only after a warm-up period.

**Branch**: `innovation/noise-consistency-training`

**Ablation**: each degradation separately, mixed degradation, and consistency
weight. Evaluate clean, Gaussian, stripe, mixed, and cross-dataset test sets.

**Risk**: Excessive noise can erase 2-pixel targets. Sample amplitudes from the
paper's ranges and log the sampled corruption parameters per run.

### 4. Hard-Negative Background Prototype Loss (HNBP)

**Hypothesis**: Blind-pixel clusters and water/urban clutter create high-scoring
false alarms. The most informative negatives are not random background pixels,
but the top-k background predictions.

**Objective**: Maintain a target prototype from positive decoder embeddings and a
background prototype from low-score embeddings. Penalize the top-k background
pixels whose prediction exceeds a margin, plus a prototype separation term.

**Branch**: `innovation/hard-negative-prototype-loss`

**Ablation**: top-k BCE, prototype separation, and combined objective. Report Fa
at fixed Pd (e.g. Pd=0.95) rather than only threshold 0.5.

**Risk**: Images without visible targets have no positive prototype. Fall back to
the hard-negative term and never fabricate a positive embedding.

### 5. Scale-Conditioned SPKAL (SC-SPKAL)

**Hypothesis**: A fixed sinusoidal/grid basis cannot be optimal for targets from
2 to 137 pixels. Let the neck select low- and high-frequency basis mixtures from
an estimated feature scale descriptor.

**Module**: Replace the fixed sine window with a small bank of learnable
frequencies and a softmax gate conditioned on feature variance and spatial
gradient energy. Keep the original SPKAL as an identity-weighted expert.

**Branch**: `innovation/scale-conditioned-spkal`

**Ablation**: fixed grid 5, paper-fidelity grid 12, learnable frequency bank,
and gated frequency bank.

**Risk**: This is the highest-complexity candidate and may confound scale with
noise. First run the grid-5/grid-12 fidelity comparison; only then add gating.

### 6. Radiometric-Invariant Dual-Path Stem (RIDS)

**Hypothesis**: Dataset-level mean/std normalization does not remove sensor and
scene radiometric shifts. A global intensity path plus a per-image robust local
contrast path should improve unseen-domain generalization.

**Module**: Compute `z_global` using the existing normalization and
`z_local=(x-median(x))/(MAD(x)+eps)` after percentile clipping. Fuse the two with
a 1x1 adapter before the existing stem; no raw image is exposed to later blocks.

**Branch**: `innovation/radiometric-invariant-stem`

**Ablation**: global only, local only, fixed fusion, learned fusion. Train on
NUDT-SIRST and test on SIRST V1 as in paper Table 8.

**Risk**: Per-image normalization can suppress a very large target. Clip robust
statistics and include a global residual channel.

### 7. Frequency-Selective Context Neck (FSCN)

**Hypothesis**: Background continuity is mostly low-frequency while target
boundaries and point-like anomalies occupy a complementary band. A gated
low/high-frequency decomposition can make CViT context safer for clutter.

**Module**: Add a depthwise Laplacian/DoG residual branch around `e5`, concatenate
it with the compressed-attention output, and learn a frequency gate before PCM.

**Branch**: `innovation/frequency-selective-neck`

**Ablation**: Laplacian only, DoG only, learned gate, and no-frequency baseline;
include FLOPs and memory because FFT is not required by this implementation.

**Risk**: Pure high-pass filtering can remove blurred targets. Use a residual
low-pass path and learn the gate rather than replacing the feature.

### 8. Uncertainty-Calibrated False-Alarm Head (UCFH)

**Hypothesis**: The fixed threshold 0.5 is not calibrated across datasets or
noise levels, while Fa is highly sensitive to a few bright background pixels.

**Module/objective**: Predict a mean logit and a positive uncertainty map. Use a
heteroscedastic BCE during training, then select the operating threshold on the
validation split for a declared target Pd (for example Pd >= 0.95).

**Branch**: `innovation/uncertainty-calibrated-head`

**Ablation**: fixed 0.5, validation threshold only, uncertainty head only, both.

**Risk**: Threshold tuning can leak test labels. The threshold must be selected
only on the held-out validation list and frozen before official testing.

## Recommended implementation order

1. `innovation/noise-consistency-training` — low architectural risk and directly
   aligned with the paper's robustness experiments.
2. `innovation/local-contrast-pyramid-gate` — strong IR-specific inductive bias
   with modest compute overhead.
3. `innovation/contour-guided-decoder` — strongest direct response to the paper's
   stated limitation.
4. `innovation/hard-negative-prototype-loss` — useful if false alarms dominate.
5. `innovation/scale-conditioned-spkal` — only after the grid-fidelity baseline.
6. RIDS, FSCN, and UCFH — follow-up branches after the first three are measured.

For every branch, use the same split, seed set, epoch budget, checkpoint rule,
and threshold protocol. Report mean ± standard deviation over at least three
seeds; do not treat a single SIRST3 run as evidence of a general improvement.
