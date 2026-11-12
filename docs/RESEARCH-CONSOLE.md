# SOUL AI Research Console

Open `tools/soul-research-console.html` directly in a modern browser. No server, package install, network access or model runtime is required.

1. Download and extract a Colab export ZIP.
2. Select the extracted top-level folder in the console. It must contain `A/metrics.json`, `A/report.json`, `B1/metrics.json` and `B1/report.json`.
3. Review the checkpoint comparison, per-episode errors and hashes before interpreting the aggregate.

The page is a read-only renderer. It does not run inference, alter checkpoints, fit baselines, read Confirm data, or select thresholds. It displays exploratory results and should not be used as a promotion gate.
