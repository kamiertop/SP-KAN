# Audit Resolution Record

The independent review from 2026-08-04 was resolved in the follow-up commit.

## Resolved Findings

1. **Dataset-level F1**: `util.metrics.F1` now accumulates TP, FP, and FN
   over every validation batch. Training and testing construct it with the
   configured threshold.
2. **Threshold consistency**: mIoU, nIoU, Pd/Fa, and F1 now use
   `--threshold`. Test F1 no longer depends on the fixed 0.5 ROC bin.
3. **Dataset caching**: training and test datasets eagerly populate their
   in-memory cache during initialization. Data loaders enable persistent
   workers whenever `--threads > 0`, so worker caches survive epoch changes.
4. **Evaluation pairing**: `test.py` broadcasts single model, dataset, or
   checkpoint values and otherwise requires equally sized lists. It no longer
   evaluates their Cartesian product.
5. **Loss reporting**: console, TensorBoard, checkpoint history, and JSONL
   now use the same single-epoch mean loss, named `train_loss`.
6. **Smoke-step limit**: `--max_train_steps` now counts completed optimizer
   steps and supports a batch size of one.

## Verification

- `python -m unittest tests/test_metrics.py`
- One-epoch training smoke test with `--batchSize 1 --max_train_steps 1`
- One-sample test smoke test with a synthetic checkpoint and `--threshold 0.7`
- Python compilation and `git diff --check`

The worker-process cache path is configured with `persistent_workers`; it was
not executed in the restricted sandbox because multiprocessing sockets are
blocked there.
