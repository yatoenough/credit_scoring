# Credit Scoring

A small credit card application demo. The Streamlit interface collects an applicant’s age, income, education, employment, and car ownership, then sends those details to a local scoring API to determine approval.

The project includes a logistic regression training routine and scoring data in CSV and SQLite formats. Training reads `data/scoring.db`, evaluates precision and recall, and saves the model to `models/model.pkl`.

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

The interface expects a scoring API that accepts the applicant fields and returns an `approved` result.
