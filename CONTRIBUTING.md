# Contributing

Thanks for your interest. Bug reports, ideas and code are all welcome.

## Reporting a problem or idea

Open an issue on GitHub. For a matching problem, the most useful report says:

- what you expected to match and what happened instead
- the shape of your files: column names and two or three example rows

**Never attach real bank statements or books.** Replace names, amounts and
reference numbers with made-up ones that still show the problem, or reproduce
it with the synthetic generator (`python synth/generate.py`).

## Setting up

```bash
git clone https://github.com/Tamalika101/recon-pipeline
cd recon-pipeline
pip install -r requirements.txt pytest
python -m pytest -q          # all tests should pass
streamlit run app.py         # the app, at http://localhost:8501
```

## Making a change

1. Fork the repo and create a branch.
2. Make your change, with a test that shows it working. Matching changes
   should keep the end-to-end tests in `tests/test_engine.py` passing,
   especially the rule that automatic matches are never wrong.
3. Run `python -m pytest -q`.
4. Open a pull request explaining what changed and why.

## Ground rules for the matching engine

- **Never guess.** If two candidates fit equally well, the engine suggests
  both and lets a person decide. A missed match costs a click; a wrong match
  can hide a real problem.
- **Explain every match.** Each match records its layer, confidence and a
  plain-language reason.
- **Keep the engine dependency-free.** `recon/` uses only the Python standard
  library; Streamlit and pandas are for the app only.
- **Keep "How it works" honest.** If you change a rule, update its text in
  `recon/explain.py`, which is generated from the real settings.

## Adding a matching layer

Layers are plug-ins. Subclass `Layer` in `recon/layers.py`, give it a unique
`name`, decorate it with `@register_layer`, and add the name to
`MatchConfig.layers`. See the README for an example.

## License

By contributing, you agree that your contributions are licensed under the
MIT License in `LICENSE`.
