# Reproduction notes

The repository packages existing work; no new model inference was performed during upload.

The original Kaggle baseline script is included as `scripts/run_baselines.py`. It tunes the clustering distance threshold on ES2011a and selects 0.8. The earlier local draft named `run_small.py` is deliberately not used as the canonical runner because it contains an older hard-coded threshold.

The runner schema is consistent across systems: meeting, microphone, system, setting, DER, missed speech, false alarms, confusion, reference count, predicted count and JER. Measured RTF is also present for the two larger models.

The pyannote script requires Kaggle's secret interface. It logs a message and exits when that secret is unavailable. Model terms, tokens and network downloads must be configured before a new inference run. The cloud/GPU runs should be isolated from unrelated working files.

`scripts/bootstrap.py` changes only path handling, loads the same derived reference-duration totals from JSON and writes its output to a different filename. It retains the original 10,000-resample seed-0 recording-level method and its known limitations. `scripts/original_bootstrap.py` preserves the original file byte-for-byte.

For a stronger future analysis, store exact scorer components, resample independent meetings with microphone variants together, add more independent sessions, and document an appropriate multiple-comparison procedure. Do not interpret current non-separation groups as proven equality.

The static dashboard can be regenerated without a frontend framework:

```python
from pathlib import Path
root = Path('dashboard')
template = (root / 'template.html').read_text(encoding='utf-8')
data = (root / 'data.json').read_text(encoding='utf-8')
(root / 'public' / 'index.html').write_text(
    template.replace('__DATA__', data), encoding='utf-8'
)
```

The checked-in dashboard is the existing deployed pilot snapshot. Hosting configuration is included; deploying it would be a separate action from uploading this repository.
