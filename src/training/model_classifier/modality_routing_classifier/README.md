# Modality Routing Classifier

This pipeline trains a three-class prompt classifier:

| Label | Intended response |
|---|---|
| `AR` | text |
| `DIFFUSION` | generated image |
| `BOTH` | text plus an image or diagram |

The classifier predicts requested output modality, not whether an image model
is available or whether image generation is safe for the prompt.

## Train

`run_training.sh` installs its Python packages, builds the dataset, trains an
mmBERT LoRA adapter, and runs the script's inference demo:

```bash
bash run_training.sh
```

Environment variables such as `MODEL`, `EPOCHS`, `BATCH_SIZE`, `MAX_SAMPLES`,
and `LEARNING_RATE` override its defaults. If `VLLM_ENDPOINT` is set, the script
can synthesize examples for the `BOTH` class; review those examples before
using them as labels.

For direct control:

```bash
python modality_routing_bert_finetuning_lora.py \
  --mode train \
  --model mmbert-32k \
  --max-samples 6000 \
  --output-dir models/modality-router
```

Use `--help` for current LoRA, GPU, and synthesis options.

## Export a Reviewable Dataset

The exporter writes deterministic train, validation, and test JSONL files,
label mappings, dataset statistics, export configuration, a dataset card, and a
Hugging Face `DatasetDict`:

```bash
python export_modality_dataset.py \
  --output-dir modality-routing-dataset \
  --max-samples 6000 \
  --overwrite
```

To add model-generated `BOTH` examples, pass `--vllm-endpoint`,
`--vllm-model`, and `--synthesize-both`. Publishing with `--push-to-hub`
requires `--repo-id` and an `HF_TOKEN`.

The training script currently rebuilds and internally splits its dataset,
whereas the exporter preserves the split returned by `prepare_datasets()`.
Use the exported split for dataset review and reproducible evaluation.

## Data and Evaluation

The data loader draws text-only prompts from instruction datasets,
image-generation prompts from DiffusionDB, and mixed-modality prompts from
curated templates or optional synthesis. Check dataset revisions and class
balance before every training run.

Report per-class precision and recall, the confusion matrix, multilingual
coverage, and failure cases such as requests that mention an image without
asking to create one. Validate the exported adapter through the router's actual
modality signal path before treating it as supported.

## Router-Native Evaluation & Graduation Gate (Issue #3198)

Per Decision Record [Routing-Native Model Experiments](../../../../../website/docs/proposals/routing-native-model-experiments.md) (Issue #3198 / Epic #2974), evaluating candidate model families beyond BERT requires two empirical protocols:

### 1. Routing Agreement and Fixed-Policy Controls

Standard accuracy alone is insufficient. Candidates must report per-request routing agreement with the baseline and outperform static cost-matched policy controls:

```bash
python evaluate_routing_agreement_and_controls.py \
  --eval-file path/to/test_predictions.jsonl \
  --output-json eval_report.json
```

Or pass separate files:

```bash
python evaluate_routing_agreement_and_controls.py \
  --ground-truth test_dataset.jsonl \
  --baseline-preds baseline_preds.jsonl \
  --candidate-preds candidate_preds.jsonl \
  --output-json eval_report.json
```

The script evaluates:
- **Routing Agreement Rate:** Per-request decision agreement $\mathbb{I}(\hat{y}_{\text{cand}} = \hat{y}_{\text{base}})$.
- **Disagreement Attribution:** Which model matched ground truth when decisions diverge.
- **Fixed-Policy Controls:** Compares candidate against `always-cheapest` (100% AR), `always-strongest` (100% BOTH), and `best-fixed-split-matched-cost` (optimal static probability mixture under the candidate's average cost budget).
- **Graduation Gate:** Verifies Macro F1 parity (within 1%), agreement rate ($\ge 90\%$), positive disagreement attribution, and outperforming the static fixed split.

### 2. Same-Run Latency & Profiling Harness

To avoid runtime drift (where batch shape or warm state changes between runs causing false wins):

```bash
python same_run_profile_harness.py \
  --backend torch \
  --baseline-model models/mmbert32k-modality-classifier \
  --candidate-model models/candidate-student-classifier \
  --batch-sizes 1 4 8 16 \
  --warmup-runs 10 \
  --eval-runs 50 \
  --output-json latency_report.json
```

For quick local or CI verification with simulated models:

```bash
python same_run_profile_harness.py --backend dummy
```

The harness executes interleaved trials in the same process with fixed batch shapes and sequence lengths, reporting p50/p90/p95/p99/p99.9 latency, peak RSS memory, and enforcing the $\ge 20\%$ p99 latency reduction gate.
