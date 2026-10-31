# Research implementation instructions

- Keep all research code, downloads, generated outputs, and project notes inside this directory.
- Follow PLAN.md as the agreed protocol; document any discrepancy before changing experimental behavior.
- Prioritize readable Python with small functions, explicit configuration, useful type hints, and actionable errors.
- Use BasicTS for training, validation, and evaluation.
- Use the September 2026 FRED-MD snapshot, a verified checksum, shared dependencies, and saved manifests.
- Confirm the shared team implementation before introducing a separate pipeline.
- Preserve raw inputs and existing results. Keep study outputs separated by group, model, schedule, and seed.
- Keep INDPRO windows independent of auxiliary missingness; fit per-series scaling only on transformed 1960-1999 observations using ddof=0.
- Reuse exact prepared batches and common initial weights across matched comparisons.
- Verify actual optimizer-update counts and evaluate final weights rather than a best-validation checkpoint.
- Use ordinary MSE and MAE including valid zeros. Never fabricate results.
