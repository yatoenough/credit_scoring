# Credit Scoring

A small credit card application demo. The [Streamlit](https://github.com/streamlit/streamlit) interface collects an applicant’s age, income, education, employment, and car ownership, then sends those details to a scoring API to determine approval.

The project includes a logistic regression training routine and scoring data SQLite format. Training reads `data/scoring.db`, runs 5-fold stratified cross-validation against a dummy baseline, picks the decision threshold that maximizes F1 on out-of-fold predictions, and evaluates once on a held-out test set (ROC-AUC, PR-AUC, precision, recall, F1, approval rate, confusion matrix). It saves the pipeline and threshold to `target/model.pkl` and the metrics to `metrics/metrics.json`.

## Run

Install dependencies with [uv](https://docs.astral.sh/uv/):

```sh
uv sync
```

Train the model:

```sh
uv run train
```

Run scoring service:

```sh
uv run serve
```

Run the application interface:

```sh
uv run streamlit run src/credit_scoring/app.py
```

User interface expects a scoring API that accepts the applicant fields om `/score` and returns an `approved` result.

## Roadmap

- [ ] Dockerize the API and Streamlit app.
- [X] Explore the data with basic EDA: distributions, missing values, class balance, and feature relationships.
- [X] Add a preprocessing pipeline and save it together with the model.
- [X] Improve evaluation with reproducible splits and metrics such as precision, recall, and a confusion matrix.
- [ ] Add tests and CI checks, and document the dataset and the demo's limitations.
