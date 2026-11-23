# Bank reconciliation

Upload a bank statement and an export of the same account from your books. The
app matches them, shows what doesn't line up and why, lets you resolve each
item, and produces a reconciliation report with suggested journal entries.

Nothing is stored: files and decisions live in the browser session's memory
and disappear when the tab closes.

## Run the app

```bash
pip install -r requirements.txt
streamlit run app.py
```

Click **Try it with sample data** to see it work without your own files.

The app has four screens:

| Screen | What you do there |
|---|---|
| Upload | Add the two CSVs; column mapping is guessed, you confirm or correct it |
| Summary | Match rate, how matches were made, items by reason, value still open |
| Needs attention | Accept suggested matches, or mark items resolved with a reason and note |
| Report and download | Bank vs book balance with reconciling items, journal entries, downloads (HTML report, CSVs, zip) |

## Deploy to Streamlit Community Cloud (free)

1. Push this folder to a GitHub repo (public for the free tier).
2. Sign in at share.streamlit.io with GitHub.
3. Create app → pick the repo, branch `main`, main file `app.py` → Deploy.

## Engine only (command line)

```bash
python synth/generate.py --seed 42 --out data            # synthetic bank + ledger + truth
python -m recon run --bank data/bank.csv --ledger data/ledger.csv \
    --period-end 2026-11-30 --out results --truth data/truth.csv
python -m pytest -q
```

`--truth` is optional. With it, the run is scored against the generator's ground truth.
The engine itself uses only the Python standard library; the app adds Streamlit and pandas.

## How matching works

| Layer | Matches when | Confidence |
|---|---|---|
| 1. Reference ID | bank and ledger share a reference and amount | 100 |
| 2. Exact | same amount, bank posts 0-5 days later, exactly one candidate each way | 95 |
| 3. Fuzzy | score on amount, date and description is 90+ and clearly beats the runner-up | score |
| 4. Combination | one line equals a group of 2-5 lines on the other side, and no other group fits | 90 |

Rules that hold across all layers:

- **No guessing.** If two candidates fit equally well, nothing is matched. Both become suggestions for a person.
- **Every match records its layer, confidence and a plain-language reason**, e.g. `Score 93: same amount, bank posted 7 days later, description 100% similar`.
- **Leftovers become exceptions with a suggested bucket**: timing, missing_from_books, error or real_problem. When the system can't tell, the bucket is blank and a person decides.

Settings such as windows, thresholds, weights and which layers run live in
`MatchConfig`, one per firm. Each run saves a copy in `config.json`.

## Adding a layer (plugin slot)

```python
from recon import Layer, register_layer

@register_layer
class MyLayer(Layer):
    name = "my_layer"
    def run(self, state):
        for b in state.open_bank():
            ...  # state.add_match(...) or state.suggest(...)
```

Then put `"my_layer"` in `MatchConfig.layers` where it should run.

## Outputs of a run

| File | Contents |
|---|---|
| transactions.csv | normalized transactions from both sides |
| matches.csv | match ID, layer, type, confidence, rows, reason |
| exceptions.csv | unmatched items, suggested bucket, reason, top suggestion |
| suggestions.json | every suggestion, including rival candidates |
| summary.json | counts, match rate, share by layer, unreconciled amounts, ID coverage (feeds the summary view) |
| config.json | the matching settings this run used |
| evaluation.json | precision, recall and per-scenario scores (with `--truth`) |

## Layout

```
app.py        Streamlit app
recon/        engine: models, normalize, text, layers, engine, evaluate, CLI,
              plus autodetect (column guessing), review (decisions), report
synth/        synthetic data generator and an independent dataset checker
data/         sample dataset (seed 42) used by "Try it with sample data"
mappings/     column mappings for the generator's bank and ledger formats
tests/        unit tests per layer, review/report tests, end-to-end app test
```
