# Fine Tune a Support Ticket Router

This project adapts `Qwen/Qwen3-1.7B-Base` with Low-Rank Adaptation (LoRA) to classify IT helpdesk tickets into seven routing queues. The Google Colab notebook covers dataset preparation, training through LLaMA Board, adapter merging, inference, and evaluation.

The saved reference run reports **98.1% accuracy on 1,000 held-out tickets**, compared with **20.9% for the base model**. The **77.2 percentage point improvement** compares two pipelines with different prompts and decoding methods; it does not isolate the effect of changing model weights alone.

## Project documentation

- [Project paper](docs/project-paper.md): abstract, problem statement, data audit, architecture, methodology, experimental setup, results, error analysis, limitations, and references.
- [Word project paper](docs/Support_Ticket_Router_Submission.docx): the paper in a shareable document format.
- [Reproduction guide](docs/reproducibility.md): execution order, configuration checks, troubleshooting, and artifact retention.
- [Reference training log](docs/llama-board-training-log.md): observations, configuration inferences, and the unresolved validation-split discrepancy.

## Task and scope

| Queue | Intended routing scope |
|---|---|
| `Active Directory` | Accounts, identity, and access workflows |
| `Fileservice` | Shared folders and network drive access |
| `O365` | Outlook, Teams, OneDrive, and collaboration issues |
| `EOL` | Server retirement and decommissioning |
| `Software` | Application installation, updates, and access |
| `Computer-Services` | Printers, scanners, drivers, and device support |
| `Support general` | General requests requiring triage |

These are working descriptions, not a formal annotation policy. The implementation returns a label and an optional confidence score. Ticket-system integration, webhook delivery, a serving API, and automatic human-review thresholds are future work.

## Architecture

![Support ticket router architecture](docs/images/Architecture.png)

The supplied diagram summarizes training and the intended routing workflow. Email/chat/portal ingestion, queue dispatch, and confidence-based human triage are proposed extensions, not implemented integrations. The 4,000 rows form a training pool; the internal validation setting and reconstructed three-epoch schedule are unresolved. “No leakage” means the held-out rows are excluded from TRAIN.json; semantic overlap has not been audited. Confidence is an uncalibrated first-token score, and the two reported accuracies use different decision rules.

## Results at a glance

| Metric | Base model | Fine-tuned model |
|---|---:|---:|
| Accuracy | 20.9% | 98.1% |
| Macro F1 | 0.151 | 0.981 |
| Correct predictions | 209 / 1,000 | 981 / 1,000 |
| Errors | 791 | 19 |
| Active Directory recall | 0.7% | 97.2% |

![Corrected baseline and fine-tuned comparison](docs/images/baseline_vs_finetuned.png)

The chart uses corrected class mappings. Historical report outputs and the embedded comparison chart in the notebook still contain an earlier class-name ordering error; the notebook source already fixes all three `classification_report` calls. See [metric provenance](docs/project-paper.md#7-results-and-metric-provenance).

## Dataset

`support_tickets.csv` contains **5,000 rows** with columns `category_truth` and `text`. Each class has 714 or 715 examples. The review found no empty ticket text, exact duplicates, or duplicates after lowercasing and collapsing whitespace. Ticket length averages 175.8 characters, with a range of 72–492.

The notebook shuffles and creates a stratified **4,000 / 1,000** split using `random_state=42`. The 1,000-row file is named `val_split.csv`, but serves as the final evaluation set. Dataset origin, annotation process, licensing, and semantic overlap between splits are not established by the repository.

## Run the notebook

1. Open [the notebook](Finetune_Support_Ticket_Classifier_Qwen3.ipynb) in Google Colab and select a T4 GPU runtime.
2. Run installation and GPU checks in a fresh session. The installation cell deletes an existing `/content/LLaMA-Factory` directory; preserve earlier runs before executing it.
3. Upload `support_tickets.csv`. Verify 4,000 training-pool rows and 1,000 held-out rows.
4. Launch LLaMA Board, select `Qwen/Qwen3-1.7B-Base`, supervised fine-tuning, LoRA, and dataset `support_tickets`. Save explicit training parameters; current UI defaults may differ from the reference run.
5. Choose internal validation deliberately. The notebook suggests `0.1`, leaving about 3,600 training examples and 400 internal validation examples. The recorded 750-step run instead fits all 4,000 examples at effective batch size 16 for three epochs. The original YAML is needed to resolve this difference.
6. Wait for training to finish, copy the complete adapter path into `ADAPTER_DIR`, then stop the web UI cell manually.
7. Run loss visualization, baseline evaluation and merge, smoke tests, held-out evaluation, and charts in order.
8. Save the adapter, tokenizer, training configuration, logs, split files, and outputs before Colab recycles the runtime.

The historical log suggests roughly **2–3 hours for training on one T4**; the exact completion time is not retained. The [reproduction guide](docs/reproducibility.md) explains each step and its expected artifacts.

## Repository map

| Path | Purpose |
|---|---|
| `Finetune_Support_Ticket_Classifier_Qwen3.ipynb` | Executable workflow and historical outputs |
| `support_tickets.csv` | Labeled input dataset |
| `docs/project-paper.md` | Detailed technical paper |
| `docs/Support_Ticket_Router_Submission.docx` | Word version of the paper |
| `docs/reproducibility.md` | Execution and artifact guide |
| `docs/llama-board-training-log.md` | Reference-run evidence |
| `docs/images/` | Training, confusion-matrix, and corrected comparison figures |
| `docs/reference-results.json` | Transcribed aggregate evidence and dataset audit |

Model weights, training YAML, the full trainer log, and per-ticket predictions are not included. Recorded scores were reviewed against saved outputs and aggregate counts; training and model inference were not rerun for this documentation update.

## Main limitations

The two models use different decision rules. Fine-tuned inference accepts case-insensitive label prefixes and silently assigns `Support general` when generation matches no label. Its confidence uses only the **first token of each label**, not the probability of the complete label, and has not been calibrated. A five-ticket smoke test includes an account-creation error despite the high held-out score.

Before deployment, preserve complete experiment artifacts, use a matched evaluation protocol, audit similar tickets across splits, inspect the 19 errors, and measure performance on representative operational traffic. The repository does not establish latency, serving cost, production accuracy, or an adequate human-review threshold.
