# Democritus Causal-Extraction Benchmark

This directory supports the frozen upstream extraction comparison reported in
Table 8 of:

> Sridhar Mahadevan. “Democritus: Homotopy-Localized Causal Discourse
> Extraction from Language.” *Entropy* **2026**, *28*(9), 986.
> [https://doi.org/10.3390/e28090986](https://doi.org/10.3390/e28090986)

The comparison uses 404 unique passages from UniCausal's grouped AltLex test
split: 115 passages contain at least one annotated causal relation, with 127
directed cause-effect pairs in total. Every method is scored against the same
frozen split.

## Redistribution Boundary

No benchmark dataset is included in this repository. The UniCausal maintainers
reported that they could not locate licensing information for AltLex, so users
must obtain the grouped split from its original source and assess its terms.

Frozen prediction JSONL files are also excluded because they contain spans
copied from that source text. The aggregate metric files are included, together
with the SHA-256 hashes of the exact prediction files used for the paper. This
preserves result identity without redistributing the underlying passages or
text-bearing derivatives.

## Reported Results

| Method | Classification P/R/F1 | Cause F1 | Effect F1 | Span macro F1 | Exact/relaxed pair F1 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Cue baseline | 50.0 / 22.6 / 31.1 | — | — | — | — |
| UniCausal public baseline | 52.0 / 68.7 / 59.2 | 40.8 | 46.5 | 43.6 | 20.5 / 37.3 |
| Restricted Democritus (Kimi K2.5) | 44.1 / 80.9 / 57.1 | 37.1 | 41.1 | 39.1 | 0.0 / 35.8 |

Prediction hashes recorded by the metric files:

| Method | SHA-256 |
| --- | --- |
| Cue baseline | `74176742e0d4b31266740196f98a879456f1a23b4913a01e38f50d625757872c` |
| UniCausal | `19fbb48eed3ed064aad348a7dfcb809358cedcfb02ffad4ac0486c19935183c0` |
| Restricted Democritus | `b446ae4acb6a5f4c85d2d58a2fc3c6aad4fcc4a27fd8d789210fea8412b752db` |

The restricted Democritus run completed all 404 requests without a request or
JSON-parsing failure. Six proposed relation records failed strict verbatim-span
validation. Its zero exact-pair score and 35.8% relaxed-pair F1 must be read
together: the extracted direction often overlaps the annotation while using a
broader clause boundary.

## Reproduce the Comparison

Create an environment with CLIFF's optional UniCausal dependencies:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e '.[unicausal]'
```

Then supply a legally obtained grouped UniCausal/AltLex CSV:

```bash
cd experiments/causal_extraction_benchmark
DATA=/path/to/UniCausal/data/grouped/splits/altlex_test.csv

python benchmark.py --dataset "$DATA" manifest --output outputs/altlex_manifest.json
python benchmark.py --dataset "$DATA" cue --output outputs/cue_predictions.jsonl
python benchmark.py --dataset "$DATA" score \
  --predictions outputs/cue_predictions.jsonl \
  --output outputs/cue_metrics.json

python benchmark.py --dataset "$DATA" unicausal \
  --output outputs/unicausal_predictions.jsonl
python benchmark.py --dataset "$DATA" score \
  --predictions outputs/unicausal_predictions.jsonl \
  --output outputs/unicausal_metrics.json
```

The paper's restricted Democritus comparison used Kimi K2.5 at temperature
zero, with reasoning disabled, through an OpenAI-compatible chat-completions
endpoint. The harness is endpoint-agnostic:

```bash
python benchmark.py --dataset "$DATA" llm \
  --base-url http://HOST:PORT \
  --model MODEL_ID \
  --workers 4 \
  --output outputs/democritus_predictions.jsonl
python benchmark.py --dataset "$DATA" score \
  --predictions outputs/democritus_predictions.jsonl \
  --output outputs/democritus_metrics.json
```

The LLM command is resumable. It retains raw responses, validation failures,
runtime, usage, model identity, and errors in the local output file.

## Tests

```bash
python3 -m unittest test_benchmark.py
```
