# Credit Scoring

A small credit card application demo. The [Streamlit](https://github.com/streamlit/streamlit) interface collects an applicant’s age, income, education, employment, and car ownership, then sends those details to a scoring API to determine approval.

The project includes a logistic regression training routine and scoring data SQLite format. Training reads `data/scoring.db`, evaluates precision and recall, and saves the model with preprocessing pipeline.

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
- [ ] Improve evaluation with reproducible splits and metrics such as precision, recall, and a confusion matrix.
- [ ] Add tests and CI checks, and document the dataset and the demo's limitations.
