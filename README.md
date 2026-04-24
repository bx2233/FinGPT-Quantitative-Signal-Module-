# FinGPT Quantitative Signal Module

A reusable quantitative-signal module built on top of a LoRA-fine-tuned FinGPT
adapter. Turns unstructured financial news into a typed, validated, calibrated
trading signal (direction, magnitude bucket, continuous score, confidence,
rationale) that a downstream quant pipeline can consume directly.

**Author:** Beibei Xian
**Course:** STAT-GR5398 Spring 2026 · FinGPT LLM Track · Columbia University
**Status:** Active development — deadline extended, still tuning and expanding
the evaluation panel.

---

## What the module produces

One call in, one typed signal out:

```python
from signal_module import FinGPTSignalModule

fingpt = FinGPTSignalModule(
    base_model_name="deepseek-ai/DeepSeek-R1-Distill-Llama-8B",
    adapter_path="checkpoint-50",
    load_in_8bit=True,
    mode="simplified",
)

signal = fingpt.generate_signal(news_text="Apple beat Q3 earnings and raised guidance ...")
print(signal.to_dict())
```

```json
{
  "bucket": "Up 1-3%",
  "direction": "Bullish",
  "sentiment_score": 0.75,
  "confidence": 0.80,
  "event_type": "other",
  "urgency": 0.5,
  "rationale": "AXP beat consensus and guided higher ..."
}
```

Self-consistency calibrated confidence (majority vote over N samples):

```python
signal_cal = fingpt.generate_with_calibration(news_text=..., n_samples=5)
```

The module supports two output schemas: a **simplified** two-field schema
(`prediction`, `analysis`) that the adapter was retrained to emit reliably, and
an **original** six-field structured schema (`sentiment_score`, `direction`,
`confidence`, `event_type`, `urgency`, `rationale`). In both modes the module
validates, clips, and canonicalises every field before returning it.

## Why a signal module (and not just a fine-tuned model)

The fine-tuned adapter is just one replaceable component. The value of this
repo is the **interface around it**: schema, parsing, validation, mapping,
calibration, and an evaluation panel. Any future adapter — retrained on more
data, with different features, or with a different base model — drops in behind
the same interface and gets the same evaluation battery for free.

## Repo layout

```
.
├── README.md                                # this file
├── requirements.txt
├── signal_module.py                         # FinGPTSignalModule (dual-mode, calibration)
├── evaluate_signal.py                       # direction / bucket / MSE / ROUGE / ECE / IC
├── FinGPT_Assignment2_Checkpoint50.ipynb    # end-to-end notebook (retrained adapter)
├── outputs/                                 # evaluation artefacts from the latest run
│   ├── signals_greedy_100.jsonl
│   ├── signals_selfconsistency_100.jsonl
│   ├── metrics_greedy_100.json
│   ├── metrics_selfconsistency_100.json
│   ├── metrics_selfconsistency_with_ic.json
│   ├── comparison_greedy_vs_sc.csv
│   ├── confusion_direction.csv
│   ├── confusion_bucket.csv
│   ├── reliability_diagram.png
│   └── A2_full_summary.json
└── docs/
    └── medium_blog_draft.md                 # long-form writeup (draft)
```

## Module design

| Stage       | Component                    | Purpose                                                           |
|-------------|------------------------------|-------------------------------------------------------------------|
| Input       | news + company context       | strips Llama-2 `[INST]`/`<<SYS>>` wrapping, builds prompt         |
| Prompting   | `SIMPLIFIED_EVAL_PROMPT`     | primes the model with `Output: {"prediction":"` to force JSON    |
| Inference   | base (8-bit) + LoRA adapter  | DeepSeek-R1-Distill-Llama-8B + `checkpoint-50`                    |
| Parsing     | `_parse_simplified`          | robust JSON reconstruction with regex fallback                    |
| Mapping     | `BUCKET_TO_SIGNAL`           | 5-class bucket → (direction, sentiment_score in [-2, +2])         |
| Validation  | `_validate`                  | clips ranges, canonicalises direction, enforces schema            |
| Calibration | `generate_with_calibration`  | N-sample majority vote → agreement-rate confidence                |
| Aggregation | `composite_score`            | single scalar quant signal for backtesting                        |

## Evaluation panel

`evaluate_signal.py` returns everything needed to audit the signal end-to-end:

- **Direction accuracy** — binary (Up vs Down), ternary (Bullish / Neutral / Bearish), and 5-class bucket accuracy.
- **Same-direction accuracy** — 5-class predictions collapsed to Up / Down / Neutral.
- **MSE** — `sentiment_score` (scaled) vs. signed bucket midpoint %.
- **ROUGE-1 / 2 / L** — model `analysis` vs. teacher `analysis`.
- **Inference time** — mean and median per sample.
- **Expected Calibration Error (ECE)** + reliability diagram.
- **Information Coefficient (IC)** — Spearman between `composite_score` and realized returns.
- **Long–short decile return** — top-decile minus bottom-decile mean return.

## Extensions implemented

- **Structured output design** — strict typed JSON schema with post-parse validation (100% parse rate in the latest run on 100 test samples).
- **Confidence calibration** — self-consistency (Wang et al., 2022) with N-sample majority vote + agreement-rate confidence, scored with ECE and a reliability diagram.
- **Multi-signal design** — direction, 5-class magnitude bucket, continuous sentiment score, confidence, all emitted from the same call.
- **Error analysis** — confusion matrices (direction and bucket), bucket-distribution comparison, and signal quality via Spearman IC and long–short decile return.

## Quickstart (Google Colab · A100 recommended)

1. Place the retrained LoRA adapter at
   `/content/drive/MyDrive/fingpt_simplified_retrain/dow30_simplified_v1/checkpoint-50`.
2. Upload `signal_module.py` and `evaluate_signal.py` into the runtime (the
   notebook's first cell handles either `files.upload()` or a copy from Drive).
3. Open `FinGPT_Assignment2_Checkpoint50.ipynb`, attach an A100 runtime, and
   Runtime → Run all.

## Background: why a retrained adapter

The first iteration LoRA adapter (from Assignment 1) generated only end-of-sequence
tokens when loaded inside the Assignment 2 stack. Diagnosis and fix:

1. **Toolchain.** A `torchao 0.10.0` / `peft ≥ 0.17` interaction on Colab
   silently broke adapter dispatch on DeepSeek-R1-Distill-Llama-8B.
2. **Training signal.** The original adapter was trained with a full Llama-2
   `[INST]<<SYS>>...<</SYS>>[/INST]` wrapping plus a six-field JSON target on
   ~1230 Dow30 examples for 1 epoch — not enough supervision per field, so the
   adapter collapsed to EOS.

**Fix.** Retrained on the same data with a simplified `{"prediction": "...",
"analysis": "..."}` target. The resulting `checkpoint-50` generates valid JSON
100% of the time and underwrites all evaluation numbers in `outputs/`.

## References

- FinGPT · AI4Finance Foundation — https://github.com/AI4Finance-Foundation/FinGPT
- Course repo — https://github.com/AI4Finance-Foundation/STAT-GR5398-Spring-2026
- LoRA · Hu et al., 2021 — https://arxiv.org/abs/2106.09685
- Self-consistency for LLMs · Wang et al., 2022 — https://arxiv.org/abs/2203.11171
- DeepSeek-R1-Distill-Llama-8B — https://huggingface.co/deepseek-ai/DeepSeek-R1-Distill-Llama-8B
- FinGPT Dow30 forecaster dataset — https://huggingface.co/datasets/FinGPT/fingpt-forecaster-dow30-202305-202405
