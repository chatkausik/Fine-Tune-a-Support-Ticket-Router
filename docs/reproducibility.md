# Reproducing the Support Ticket Routing Experiment

This guide describes the current notebook. Exact recreation of the historical score requires missing artifacts: original training YAML, adapter weights and configuration, model revision, a complete environment lock, and per-ticket predictions. The [project paper](project-paper.md) separates recorded facts from inferences.

## Architecture overview

![Support ticket router architecture](images/Architecture.png)

The supplied diagram summarizes training and the intended routing workflow. Email/chat/portal ingestion, queue dispatch, and confidence-based human triage are proposed extensions, not implemented integrations. The 4,000 rows form a training pool; the internal validation setting and reconstructed three-epoch schedule are unresolved. “No leakage” means the held-out rows are excluded from TRAIN.json; semantic overlap has not been audited. Confidence is an uncalibrated first-token score, and the two reported accuracies use different decision rules.

## 1 Runtime and installation

Open `Finetune_Support_Ticket_Classifier_Qwen3.ipynb` in Google Colab and select a T4 GPU runtime. The workflow downloads LLaMA-Factory and model weights and starts a web interface for training. The inference code has CUDA, MPS, and CPU branches, but the full notebook assumes Colab paths and `google.colab.files`.

The installation cell changes into `/content`, **deletes an existing `LLaMA-Factory` directory**, clones the repository, and runs `pip install -e ".[torch,metrics]"`. Preserve previous adapters and logs before rerunning it. The clone is not pinned to a commit.

The GPU-check cell only prints a warning when CUDA is unavailable; it does not stop execution. Confirm `torch.cuda.is_available()` is `True` before training. The optional identity-dataset cell modifies an unrelated upstream example and is not required for this project.

Historical installation output contains dependency conflicts. A final “Successfully installed” message does not establish compatibility. Check imports, application startup, and `pip check` output. Resolve failures in a compatible environment instead of treating every warning as harmless. If a restart is needed, rerun the cells that establish Python variables.

| Package | Version printed in historical installation output |
|---|---|
| LLaMA-Factory | 0.9.6.dev0 |
| Transformers | 5.8.0 |
| PEFT | 0.18.1 |
| Accelerate | 1.11.0 |
| Datasets | 4.0.0 |
| TRL | 0.24.0 |
| Gradio | 5.50.0 |
| Tokenizers | 0.22.2 |

This is a partial record, not an installation prescription or tested lockfile. For a new run, retain the full package list, Python/PyTorch/CUDA versions, GPU details, and LLaMA-Factory commit.

## 2 Prepare the dataset

Upload only `support_tickets.csv`. The printed prompt mistakenly says `support_ticket.csv`; the code takes the first uploaded filename. Verify columns `category_truth` and `text`.

The preparation cell filters unknown labels, shuffles with seed 42, and creates a stratified 80/20 split with seed 42. Filtering is silent apart from the resulting counts. Expect 5,000 accepted rows, 4,000 training-pool records, and 1,000 held-out records.

Training records go to `/content/LLaMA-Factory/data/TRAIN.json`, registration updates `/content/LLaMA-Factory/data/dataset_info.json`, and the external evaluation set goes to `/content/val_split.csv`. The cell does not deduplicate, redact, or normalize text; placeholders already exist in the input CSV.

The source CSV SHA-256 is:

```text
e2badb2350901e22df542a43160b5e994638df3db8cee0021d8e62ac5b63884b
```

Retain the input hash, exported records, and row identifiers for each partition. Keep the final held-out partition separate from training-setting selection and confidence calibration.

## 3 Configure training in LLaMA Board

Run the web UI cell and open its newly generated public URL. Select `Qwen/Qwen3-1.7B-Base`, supervised fine-tuning, method `lora`, and dataset `support_tickets`. LoRA is the fine-tuning method, not the training stage. Verify and save the training prompt template.

Set explicit epochs, microbatch size, gradient accumulation, learning rate, scheduler, cutoff length, compute precision, LoRA rank, alpha, dropout, target modules, and seed. Defaults can change and several original settings are unconfirmed. Use these historical reconstructions as starting points to investigate, not as a verified original configuration:

| Field | Historical reconstruction |
|---|---|
| Epochs | 3 |
| Effective batch size | 16, reportedly 2 × 8 on one device |
| Initial learning rate | Approximately 5e-5 |
| Scheduler | Consistent with cosine decay |
| Training budget | Roughly 2–3 hours on the reported T4 |

Choose internal validation explicitly. With `val_size=0.1`, the pool yields approximately 3,600 training and 400 validation rows. At effective batch 16, this is about 225 optimizer steps per epoch and 675 over three epochs, subject to trainer batching behavior. Using all 4,000 rows instead gives 250 steps per epoch and 750 over three epochs. The latter matches the saved plot; the original YAML is needed to resolve the discrepancy.

Save the resolved configuration before starting. Monitor logs and checkpoints. A UI parse error alone does not prove training failed or succeeded: verify that optimizer steps continue and inspect backend logs. Wait for completion and final adapter files, then copy the complete output path.

Stop the web UI cell manually after completion. Closing a UI tunnel is separate from deleting files; restarting or recycling a runtime can remove local storage.

## 4 Set paths and inspect the loss

Set `ADAPTER_DIR` to the directory containing `adapter_config.json`, adapter weights, and `trainer_log.jsonl`. For the reference run:

```python
ADAPTER_DIR = (
    "/content/LLaMA-Factory/saves/Qwen3-1.7B-Base/lora/"
    "train_2026-09-11-17-48-28"
)
MERGED_DIR = "/content/qwen3_merged"
BASE_MODEL_NAME = "Qwen/Qwen3-1.7B-Base"
```

Replace the timestamped directory with your run. Do not prepend a base directory to an already absolute path. The loss cell reads JSON lines containing `current_steps` and `loss`, plots them, and writes `/content/training_curve.png`. Low training loss is not an evaluation result.

## 5 Evaluate the baseline and merge the adapter

Run the combined baseline-and-merge cell after the path and data cells. It loads the untouched base model, performs letter-constrained classification on held-out tickets, and prints the baseline report. Only then does it attach the adapter and merge weights.

Supported weight names are `adapter_model.safetensors`, `adapters.safetensors`, or `adapter_model.bin`. The model and tokenizer are saved to `MERGED_DIR` and retained in `_model` and `_tokenizer`. A fresh session requires rerunning the loading workflow or explicitly reloading these saved artifacts.

If `apply_chat_template` fails, check the saved tokenizer and training template. An arbitrary replacement changes evaluation conditions. If memory is exhausted, inspect device allocation and release unused models; changes in precision or quantization must be recorded as a different configuration.

## 6 Smoke test and final evaluation

Run the `classify()` cell, full evaluation, report, confusion matrix, and comparison plot in order. Five smoke-test examples check basic behavior; they do not estimate accuracy. The historical run gets four right and misroutes an account-creation request.

`classify()` generates at most 10 tokens, accepts a case-insensitive label prefix, and assigns `Support general` when none matches. Confidence is calculated from only the first token of each label. Full evaluation discards confidence and does not export raw generations or fallback indicators.

Current reports pass matching `labels` and `target_names`. Historical saved reports and the embedded comparison plot predate that correction. Rerunning the cells generates results for your model. Compare them with the [corrected reference tables](project-paper.md#7-results-and-metric-provenance), retaining any differences instead of substituting reference values.

## 7 Preserve artifacts

| Artifact | Runtime location | Included in repository |
|---|---|---|
| Input CSV | Repository root | Yes |
| Training records and registration | `/content/LLaMA-Factory/data/` | No |
| Held-out split | `/content/val_split.csv` | No |
| Training YAML | Export from UI or run directory | No |
| Adapter and configuration | `ADAPTER_DIR` | No |
| Full trainer log | `ADAPTER_DIR/trainer_log.jsonl` | No |
| Merged model and tokenizer | `/content/qwen3_merged/` | No |
| Training curve | `/content/training_curve.png` | Historical copy in `docs/images/` |
| Confusion matrix | `/content/confusion_matrix.png` | Historical copy in `docs/images/` |
| Comparison chart | `/content/baseline_vs_finetuned.png` | Corrected aggregate reconstruction in `docs/images/` |
| Per-ticket predictions | No export implemented | No |

Download or persist artifacts before the runtime recycles. Prediction exports, calibration measurements, partition manifests, and environment capture require additional code; these are proposed improvements, not current notebook outputs.

## 8 Documentation verification

This revision checked schema, label counts, empty texts, exact and normalized uniqueness, and character lengths. Fine-tuned metrics were recalculated from the visible saved matrix; baseline rows were remapped from the historical report. Matrix row totals sum to 1,000 and all 19 errors reconcile with headline accuracy. No GPU training or model inference was performed during this review.
