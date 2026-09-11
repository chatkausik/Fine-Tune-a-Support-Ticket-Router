# Fine-Tune a Support Ticket Router

A LoRA fine-tune of `Qwen/Qwen3-1.7B-Base` that reads a human-written IT helpdesk
ticket and routes it to one of seven queues. The model never writes a reply — it
classifies intent so the rest of the pipeline can take over.

| Route | Handles |
|---|---|
| `Active Directory` | identity and access, account and login workflows |
| `Fileservice` | file share and network drive permissions |
| `O365` | Outlook, Skype, Teams, OneDrive, mailbox issues |
| `EOL` | server retirement and decommissioning |
| `Software` | application installs, updates, app access |
| `Computer-Services` | printers, scanners, drivers, device support |
| `Support general` | everything else needing a human triage pass |

## Contents

- `Finetune_Support_Ticket_Classifier_Qwen3.ipynb` — the whole project: data prep,
  training via the [LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory) LLaMA Board
  UI, and evaluation against a zero-shot baseline.
- `support_tickets.csv` — 5,000 labelled tickets (`category_truth`, `text`),
  balanced at ~714 per class. Names, locations and IDs are redacted as
  `[NAME]`, `[LOCATION]`, `[TICKET ID]`, etc.

## Running it

Open the notebook in **Google Colab** with a **T4 GPU** runtime
(*Runtime → Change runtime type → T4 GPU*), then run the cells in order.

1. **Install / GPU check** — clones LLaMA-Factory and installs it.
2. **Configuration** — every constant lives here. Nothing to edit on the first pass.
3. **Prepare dataset** — finds the CSV (local file, else this repo over HTTPS, else
   prompts for upload), makes a stratified 80/20 split, writes the training portion as
   ShareGPT JSON, and registers it as `support_tickets`.
4. **Fine-tune via LLaMA Board** — opens a Gradio UI on a public URL. Pick
   `Qwen/Qwen3-1.7B-Base`, dataset `support_tickets`, stage **lora**, and set
   **Validation size** to `0.1`. Stop the cell manually when training finishes
   (~30–60 min on a T4).
5. Copy the **Output Dir** LLaMA Board reports into `ADAPTER_DIR` in the
   Configuration cell and re-run that cell.
6. Run the remaining cells: loss curve → baseline → merge → evaluation → charts.

## How evaluation works

Both the untouched base model and the fine-tuned one are scored the same way:
for each ticket, every candidate label is scored as a continuation by its
teacher-forced token log-likelihood in one forward pass, and the highest wins.
Nothing is generated and nothing is parsed, so a prediction is always one of the
seven classes and the reported gap reflects learned routing ability rather than
one model guessing the output format better than the other.

The base model is offered the classes as lettered options `A`–`G`, since it was
never taught our exact label strings; the fine-tuned model is scored over the label
strings it was trained to emit.

`test_split.csv` is held out before training and is untouched until the final
evaluation. LLaMA Board carves its own validation slice out of the training data,
so the headline accuracy is a genuine held-out estimate.

Outputs land in `/content`: `training_curve.png`, `confusion_matrix.png`,
`baseline_vs_finetuned.png`.

## Reading the results

Overall accuracy hides what matters for a router. A misrouted `Active Directory`
ticket means a new hire cannot log in on day one; a misrouted `Support general`
ticket wastes a specialist's afternoon. Look at **per-class recall on the
high-urgency queues** and at the confusion-pair breakdown the notebook prints
under the matrix — a model at 83% overall with 95% `Active Directory` recall is
safer in production than one at 90% overall with 60%.
