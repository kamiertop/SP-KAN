# Literature-backed stitching report for SP-KAN

## Material Passport

- **Type**: Top-venue abstract scan and integration design
- **Search date**: 2026-08-05
- **Databases**: arXiv API for abstracts; Crossref for DOI/venue verification
- **Scope**: CVPR, NeurIPS, ECCV, TGRS, JSTARS, and closely related ISTD work
- **Status**: Integration hypotheses; no performance claim before ablation

## Verified source set and abstract-level takeaways

| Source | Venue / identifier | What the abstract contributes | SP-KAN seam |
|---|---|---|---|
| Masked-attention Mask Transformer for Universal Image Segmentation | CVPR 2022; [DOI](https://doi.org/10.1109/CVPR52688.2022.00135), [arXiv](https://arxiv.org/abs/2112.01527) | Mask2Former restricts cross-attention to predicted mask regions and uses one architecture for semantic, instance, and panoptic segmentation. | Add a binary mask-guided attention gate to decoder/skip fusion so clutter tokens outside the provisional target region receive less attention. |
| Focal Modulation Networks | NeurIPS 35 (2022); [DOI](https://doi.org/10.52202/068431-0304), [arXiv](https://arxiv.org/abs/2203.11926) | Replaces self-attention with hierarchical depthwise contextualization, gated aggregation, and element-wise modulation. | Replace one or more `CViT` blocks with a focal context block; preserve SPKAL as nonlinear neck modulation. |
| SegNeXt: Rethinking Convolutional Attention Design | NeurIPS 35 (2022); [DOI](https://doi.org/10.52202/068431-0084), [arXiv](https://arxiv.org/abs/2209.08575) | Shows that cheap convolutional attention can encode context more efficiently than self-attention for segmentation. | Use a multi-scale depthwise convolutional attention branch at `e3/e4`, where IR local structure is still spatially resolved. |
| Wave-ViT: Unifying Wavelet and Transformers | ECCV 2022; [DOI](https://doi.org/10.1007/978-3-031-19806-9_19), [arXiv](https://arxiv.org/abs/2207.04978) | Uses invertible wavelet downsampling instead of lossy K/V pooling and inverse wavelet context aggregation to preserve high-frequency information. | Replace `Attention_org` interpolation of K/V with a Haar/wavelet compression path; fuse high-frequency bands into PCM input. |
| Restormer: Efficient Transformer for High-Resolution Image Restoration | CVPR 2022; [DOI](https://doi.org/10.1109/CVPR52688.2022.00564), [arXiv](https://arxiv.org/abs/2111.09881) | Designs efficient long-range pixel interaction and feed-forward blocks for high-resolution denoising/deblurring. | Add a small restoration branch before the SP-KAN stem or use its transposed attention only at the bottleneck for noisy IR scenes. |
| SegFormer: Simple and Efficient Design for Semantic Segmentation with Transformers | [arXiv](https://arxiv.org/abs/2105.15203) | Uses a hierarchical transformer and lightweight multi-scale MLP decoder without positional-code interpolation. | Replace the current concatenation-heavy decoder with an ablation variant of lightweight multi-scale fusion; keep SPKAL unchanged. |
| EFLNet: Enhancing Feature Learning Network for Infrared Small Target Detection | TGRS 2024; DOI [10.1109/TGRS.2024.3365677](https://doi.org/10.1109/TGRS.2024.3365677), [arXiv](https://arxiv.org/abs/2307.14723) | Proposes adaptive-threshold focal loss for extreme target/background imbalance and a dynamic head for semantic-level weighting. | Extend the existing TABDS loss with an adaptive focal term; learn semantic-head weights instead of fixed `(0.5, 0.5, 0.75, 1, 1, 1)`. |
| Local Motion and Contrast Priors Driven Deep Network for Infrared Small Target Super-Resolution | JSTARS 2022; [arXiv](https://arxiv.org/abs/2201.01014) | Uses central-difference residual groups to encode local contrast and a local spatio-temporal attention prior. | Import the central-difference contrast operator into `e2/e3/e4` skip features; use the spatial part only for single-frame SP-KAN. |
| MiM-ISTD: Mamba-in-Mamba for Efficient Infrared Small Target Detection | ISTD-specific preprint; [arXiv](https://arxiv.org/abs/2403.02148) | Separates global visual sentences and local visual words with nested Mamba blocks to reduce long-sequence cost. | Replace only the deepest `CViT` with a local/global state-space block for high-resolution inference; retain CNN shallow stages. |
| Rethinking Generalizable Infrared Small Target Detection | Emerging preprint; [arXiv](https://arxiv.org/abs/2504.16487) | Introduces cross-view channel alignment, cross-view Top-K fusion, and noise-guided representation learning for domain shift. | Add clean/noisy or contrast-augmented dual-view alignment at the PCM input and a Top-K background/target feature fusion loss. Treat venue status as unverified. |

## Highest-value stitching candidates

### A. Wavelet-compressed CViT + SPKAL

This is the most architecture-specific combination. The current
`Attention_org` bilinearly interpolates K/V to `reduce_size=16`, which can erase
the high-frequency component of 2–5 pixel targets. A Wave-ViT-style invertible
Haar decomposition can produce low-frequency K/V plus detail bands. The detail
bands should be projected and added to the query-preserving path before PCM.

**Proposed branch**: `innovation/wavelet-cvit-spkal`

**Minimal experiment**: baseline CViT interpolation vs. Haar K/V compression;
keep FLOPs, channels, training schedule, and seed fixed. Report target-area
bucket IoU/F1 and GPU memory.

### B. Focal/SegNeXt local-global context replacement

SP-KAN already has global compressed attention and local depthwise convolutions,
but these are separate. FocalNet's gated aggregation or SegNeXt's convolutional
attention can be inserted into `CViT` while preserving the same tensor interface.
This is a cleaner ablation than adding another attention module after PCM.

**Proposed branch**: `innovation/focal-context-cvit`

**Minimal experiment**: replace only `TransH3` and `TransH4`; compare attention
maps, Fa, and latency. Do not replace all five blocks initially.

### C. Mask2Former-style target-region attention

Mask2Former's key idea is not the full query-based architecture; it is restricting
attention to a predicted mask region. For SP-KAN, the fused decoder mask from the
previous stage can produce a soft spatial gate on the next skip feature:

`X_skip' = X_skip * (epsilon + sigmoid(mask_{coarse}))`.

Use a detached mask for the first warm-up epochs or the gate can collapse before
the segmentation head learns. This directly targets blind-pixel false alarms.

**Proposed branch**: `innovation/mask-guided-skip-attention`

### D. EFLNet adaptive focal objective + current TABDS

EFLNet is the closest loss-level precedent. The current TABDS already handles
positive-ratio and boundary weighting, so the scientifically clean combination is
not to copy ATFL wholesale. Add a separate adaptive focal residual and test:

`BCE_boundary + Dice` vs. `BCE_boundary + Dice + adaptive_focal`.

**Proposed branch**: `innovation/adaptive-focal-tabds`

Keep the focal contribution zero at initialization and log its effective weight;
otherwise gains may simply come from a larger loss magnitude.

### E. Dual-view domain/noise alignment

The Rethinking ISTD abstract and the paper's own noise section point to the same
unresolved issue: domain and noise shifts. Construct two views of each image:
global-normalized and local-contrast/noisy. Apply a cross-view channel alignment
loss at `e5` and prediction Jensen–Shannon consistency. This fits SP-KAN without
changing its inference graph.

**Proposed branch**: `innovation/cross-view-noise-alignment`

Use only training views; never use test-domain images or test labels for alignment.

### F. MiM-style efficient global-local bottleneck

For 2048×2048 inference, MiM-ISTD's abstract reports a nested global/local state
space design and large memory savings. A direct full-backbone replacement would
erase SP-KAN's contribution, but replacing only `TransH5` with a two-level scan is
a defensible efficiency ablation.

**Proposed branch**: `innovation/mim-bottleneck`

Measure accuracy and peak memory jointly; do not report speed without a fixed
hardware and warm-up protocol.

## Recommended order and novelty safeguards

1. Run the paper-fidelity grid-size 5 vs. 12 baseline first.
2. Implement Wavelet-CViT + SPKAL, because it addresses the current lossy K/V
   interpolation and has a clear mechanism for preserving tiny targets.
3. Implement Mask-guided skip attention or Focal-CViT as a separate branch, not
   both at once.
4. Add adaptive focal residual to TABDS only after the existing TABDS baseline is
   reproduced.
5. Evaluate cross-view/noise alignment on cross-dataset and synthetic-noise tests.

Do not claim a new method merely because several modules are concatenated. Each
branch should change one causal mechanism, use the same train/validation split and
random seeds, and report IoU, F1, Pd, Fa, target-area buckets, parameter count,
FLOPs, peak memory, and latency.

## Search limitations

Crossref verified the DOI and venue for the CVPR/NeurIPS/ECCV/TGRS entries. Abstract
text was retrieved from arXiv records when Crossref did not store an abstract.
The MiM-ISTD and Rethinking Generalizable ISTD records are treated as preprints,
not peer-reviewed top-venue evidence. OpenAlex was unavailable because its free
API quota was exhausted, and Semantic Scholar returned no usable result in this
session.
