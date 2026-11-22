# Credit Scoring

A small credit card application demo. The [Streamlit](https://github.com/streamlit/streamlit) interface collects an applicant’s age, income, education, employment, and car ownership, then sends those details to a FastAPI scoring service that decides whether to approve the card.

## Dataset

`data/scoring.db` (SQLite, table `scoring`) has 180,816 applicants and no missing values:

| Column | Type | Range |
|---|---|---|
| `age` | int | 21–72 |
| `income` | int, thousands | 1–1000 |
| `education` | 0/1 | higher education |
| `work` | 0/1 | stable job |
| `car` | 0/1 | owns a car |
| `default` | 0/1 | target |

The default rate is **11.5%**, so the classes are imbalanced. Many rows share the same feature values, because there are only five coarse features. These rows are real applicants, not copies, so they are kept. See `notebooks/eda.ipynb` for the full exploration.

## Model

`src/credit_scoring/pipeline.py` defines one scikit-learn `Pipeline`, used for both training and serving:

- `income` → `log10` → `StandardScaler`
- `age` → `StandardScaler`
- `education`, `work`, `car` → passthrough
- `LogisticRegression(class_weight="balanced")`

`uv run train` does the following:

1. Makes a stratified 80/20 train/test split with `random_state=42`.
2. Runs 5-fold stratified cross-validation and compares the model with a `DummyClassifier` baseline.
3. Picks the decision threshold that gives the best F1 on out-of-fold predictions.
4. Evaluates once on the test set.

The pipeline and threshold are saved to `target/model.pkl` and the metrics to `metrics/metrics.json`.

Latest test-set results (threshold 0.517):

| Metric | Model | Dummy |
|---|---|---|
| ROC-AUC | 0.590 | 0.497 |
| PR-AUC | 0.147 | 0.115 |
| Precision | 0.143 | 0.110 |
| Recall | 0.584 | 0.109 |
| F1 | 0.230 | 0.110 |
| Approval rate | 52.9% | 88.5% |

## Run locally

Install dependencies with [uv](https://docs.astral.sh/uv/):

```sh
uv sync
```

Train the model:

```sh
uv run train
```

Run the scoring service on `127.0.0.1:3333`. Use `HOST` and `PORT` to change the address, and `TARGET_PATH` to load a different model:

```sh
uv run serve
```

Score an applicant:

```sh
curl -X POST localhost:3333/score -H 'content-type: application/json' \
  -d '{"age":30,"income":50,"education":true,"work":true,"car":false}'
```

The response looks like `{"approved": true, "default_probability": 0.35}`. Inputs outside the training range, such as `income: 0` or `age: 10`, return 422.

Run the application interface. Set `API_URL` if the API is not at `http://127.0.0.1:3333`:

```sh
uv run streamlit run src/credit_scoring/ui.py
```

## Run with Docker

```sh
docker compose up --build
```

The image trains the model at build time. Compose starts:

- `api` on `localhost:3333`, with a healthcheck on `/health`
- `ui` on `localhost:8501`, which starts once the API is healthy

## Development

```sh
uv run pytest
uv run ruff check
uv run ruff format --check
```

CI runs the same checks on every push to `main` and on every pull request.

## Limitations

- **Small feature set.** Five coarse features carry little signal: ROC-AUC is about 0.59. At the chosen threshold the model turns down about 47% of applicants and still misses about 42% of defaults.
- **No calibration check.** `default_probability` comes from a class-balanced model, so it is not a true probability of default.
- **Age is a protected attribute.** Using it in credit decisions raises fair-lending concerns.
- **No monitoring.** There is no drift detection, logging, or retraining.
- **Not for real credit decisions.** This is a learning demo.
