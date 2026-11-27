# Benchmarking the matching funnel

Compare pipeline configurations (rules only, rules + LLM, rules + decision model + LLM, ...)
on synthetic datasets with known answers, and record accuracy, cost, latency and manual effort.

## How the funnel works

1. **Rules** (always first): reference ID, exact amount and date, close-match score, grouped amounts.
2. **Decision model** (optional): the Strands Decider picks one candidate or "none" with a
   probability. Picks at or above `threshold` are matched; the rest move on.
3. **LLM** (optional): a chat model picks one candidate or "none" with a confidence level.
   Only levels in `accept_confidence` are matched; the rest go to a person.

Models only see open bank lines whose candidate book entries **already agree on amount**
(or groups the rules found that sum exactly). Swapped digits, fees and unknown charges never
reach a model; arithmetic stays with the rules. Items are handled largest amount first, so a
budget (`max_calls`, `max_cost_usd`) is spent where the money is.

## One-time setup on a Mac (Apple silicon)

1. Install Ollama 0.40 or later from https://ollama.com and open it once.
2. Download the models (about 2 GB and 6 GB):
   ```bash
   ollama pull strands-decider
   ollama pull qwen3.5:9b        # check the exact tag with `ollama list` or on ollama.com
   ```
   If your Qwen tag differs, change `"model"` in the configs that use it.
3. Check both answer:
   ```bash
   curl -s http://localhost:11434/v1/systemone -H 'Content-Type: application/json' -d '{
     "model": "strands-decider", "state": "Bank line: -500.00 ACH DEBIT ULINE",
     "questions": {"match": {"type": "choice", "instructions": "Which entry is the same payment?",
       "criteria": {"A": "Bill payment - Uline -500.00", "none": "None of these"}}}}'
   ollama run qwen3.5:9b "Reply with the word ready"
   ```
4. From the project folder, run the tests: `python3 -m pytest -q`.

## Running benchmarks

Start small to see timings on your machine:

```bash
python3 benchmarks/run_benchmark.py \
  --configs benchmarks/configs/rules_only.json \
            benchmarks/configs/rules_decider_local.json \
            benchmarks/configs/rules_decider_llm_local.json \
  --events 300 --seeds 1
```

Then scale up. A full run like this is an overnight job on a laptop, because LLM calls run
one at a time (keep the Mac awake with `caffeinate -i` in front of the command):

```bash
caffeinate -i python3 benchmarks/run_benchmark.py \
  --configs benchmarks/configs/rules_only.json \
            benchmarks/configs/rules_llm_local.json \
            benchmarks/configs/rules_decider_llm_local.json \
  --profiles realistic hard --events 10000 --seeds 1 2 3
```

Datasets are generated once into `benchmarks/data/` and reused. Results go to
`benchmarks/results/<timestamp>/`:

| File | Contents |
|---|---|
| `summary.md` | Mean ± standard deviation per config and profile, ready for the article |
| `results.csv` | Every metric for every run, including per-stage calls, tokens, cost and latency percentiles |
| `runs/*.json` | Per-run detail: every model call's choice, confidence, tokens, latency and cost |

### Options

| Option | Meaning |
|---|---|
| `--configs` | One or more config files |
| `--profiles` | `easy`, `realistic`, `hard` (difficulty mix of the synthetic data) |
| `--events` | Transactions per dataset; several values allowed, e.g. `--events 1000 10000` |
| `--seeds` | Datasets per profile; more seeds give tighter error bars |
| `--minutes-per-item` | Assumed minutes a person spends per open item (default 2) |

## Configuration files

```json
{
  "name": "rules + decider + LLM (local)",
  "rules": {"auto_accept": 90},
  "stages": [
    {"kind": "decider", "provider": "ollama", "model": "strands-decider",
     "threshold": 0.9, "max_candidates": 4,
     "price_in_per_mtok": 0.05, "price_out_per_mtok": 0.05},
    {"kind": "llm", "provider": "ollama", "model": "qwen3.5:9b",
     "accept_confidence": ["high"], "max_calls": 2000,
     "price_in_per_mtok": 0.10, "price_out_per_mtok": 0.40, "timeout_s": 300}
  ]
}
```

| Key | Applies to | Meaning |
|---|---|---|
| `rules` | all | Overrides of the rule settings (`MatchConfig`): windows, thresholds, weights |
| `kind` | stage | `decider` or `llm` |
| `provider` | stage | `ollama`, `anthropic` (Claude; needs `ANTHROPIC_API_KEY`) or `simulated` |
| `model` | stage | Model name in that provider |
| `threshold` | decider | Minimum probability to accept a pick |
| `accept_confidence` | llm | Confidence levels to accept: any of `high`, `medium`, `low` |
| `max_candidates` | stage | Candidates shown per question |
| `max_calls`, `max_cost_usd` | stage | Budget per run; items beyond it go to a person |
| `price_in_per_mtok`, `price_out_per_mtok` | stage | Dollars per million tokens, for cost figures |
| `name` | stage | Label in results (default: the kind); needed if two stages share a kind |
| `base_url`, `timeout_s`, `think` | stage | Provider settings |

Keys starting with `_` are comments.

**About cost figures for local runs.** Local models cost nothing per token. The configured
prices let the report show what the same run would cost on a hosted model, which is the
production question. Replace them with your provider's current rates before publishing.
The Strands Decider doesn't report token counts, so its tokens are estimated at about 4
characters per token (flagged as `decider_tokens_estimated` in `results.csv`).

**Simulated configs** (`sim_*.json`) answer from the ground truth at a set accuracy, to test
the pipeline, budgets and reports without any model. They're marked *sim* in the summary.
Never report their numbers as model results.

## Reading the summary

- **Precision**: share of automatic matches that are right. The number to protect.
- **Recall (all)**: share of all real matches made automatically, including look-alike pairs
  no rule can separate.
- **Left after rules**: share of lines the rules couldn't settle; model stages are billed on these.
- **Items for a person / person-hours**: what's left for manual review.
- **Cost / 1k lines**: from token counts and configured prices.
- **Est. time**: rule time plus every model call's latency, one after another. Laptop
  latency is not production latency; label it as local in the article.
