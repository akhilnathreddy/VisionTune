# Data

Nothing in this folder is committed except this file. Datasets are downloaded from the Hugging Face Hub at pinned
revisions (see `configs/base.yaml`) and turned into record files by:

```bash
python -m visiontune.data.prepare --config configs/base.yaml
```

Outputs in `data/processed/`:

| File | Contents |
|---|---|
| `scienceqa_{train,validation,test}.jsonl` | one validated record per image question (schema: `src/visiontune/data/records.py`) |
| `ai2d_test.jsonl` | AI2D evaluation records (never used for training) |
| `prepare_report.json` | raw and kept counts, every exclusion reason, near-duplicate flags |

Records hold metadata only; `hf_index` points back to the image in the Hugging Face split.
A copy of the report for the current pinned revisions is in `results/reports/data/`.
