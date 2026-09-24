# Jubilee

Jubilee trains models from a conversation. A prep agent loads, cleans, and splits data. A training agent then engineers features, compares estimators, trains, and writes a report, pausing for human review after each step. A separate analysis agent searches the catalog, runs statistics, and calls pretrained models.

## Where things live

| | |
|---|---|
| [`agents/`](agents/README.md) | Orchestrators, the training graph, skills, and subagents |
| [`ui/`](ui/README.md) | Chat and progress UI with step confirmation |
| `backend/` | FastAPI (catalog, chat, training, experiments), plus Redis and Celery for long jobs |
| `tools/` | Data retrieval, transforms, analysis, and pretrained models |
| `datasets/` | Local CSV and SQL files, indexed by `catalog.json` |

## Run

```bash
docker compose up         # Redis, API on :8000, Celery worker
uvicorn app:app --reload  # or run the API alone, without Docker
```

UI setup and modes are in [`ui/README.md`](ui/README.md).

## Design

### Two agents, one graph

[`agents/training/agent_simple.py`](agents/training/agent_simple.py) runs a single LangGraph graph with two agents:

- **Prep agent:** collects the data, picks the model family, cleans, and defines the label and splits.
- **Training agent:** builds features, compares models, gets the training plan approved, trains, and writes the report.

Between them, a handoff node passes only the goal, target, model family, and split references. Neither agent sees the other's tools or transcript. Calling a tool whose step already finished returns `SKIP`.

### Library inspection instead of hand-written tool descriptions

`extract_params` reads constructor parameters from whatever versions of PyTorch and scikit-learn are installed, so upgrading a library never requires editing a prompt or tool schema.

- It resolves a class through registered namespaces: `torch.nn`, `torch.optim`, the learning-rate schedulers, and sklearn's `all_estimators`.
- Parameter names and defaults come from `inspect.signature`. Types, allowed choices, and log or linear scale come from the docstring (numpydoc, or PyTorch's `Args:` block).
- When the docstring is too thin, a small model reads the `__init__` source instead. Results are cached against a hash of that source.

### Skills make new capabilities a folder, not a code change

- A training skill is a directory holding a `SKILL.md`, which is injected into the agent's context, and a `train.py` exposing `run(params)`, which the agent calls through `train_with_skill`.
- A new model family or search protocol is a new folder. The graph stays the same.
- The subagents (explainer, evaluator, deployer, and others) follow the same pattern: each loads its own `SKILL.md` as its system prompt.
- **Fixed-parameter families** (supervised, unsupervised) take an estimator name. Hyperparameters come from `extract_params`, followed by a cross-validated search.
- **Code-writing families** (neural networks) run in a sandbox. The agent writes PyTorch, with helpers already loaded for loading, encoding, preprocessing, metrics, saving, and `extract_params`. The skill document sets the experiment protocol: change one thing at a time, keep or discard it, and log every run.

### A small, fixed set of data tools

The model never gets raw SQL access or a pandas session. It works through tools that hide SQL and pandas behind stable arguments.

- **`get_sql_schema`** lists the tables and columns.
- **`sql_query`** runs a parameterized `SELECT` only. It refuses writes and multi-statement queries, caps the rows returned, and registers the result as a dataset reference. A misspelled table or column name gets back the closest match.
- **Transforms** (filter, impute, cast, lag, rolling windows, group-by, pivot, and more) each take a dataset reference and return a new one, plus a one-line summary of rows, columns, and types. `apply_transformations` chains a list of them.

Because tools exchange references rather than data, the model only ever sees schemas, metrics, and feature rankings. That keeps the context small, and parallel jobs can share the same underlying tables.

### The model writes a feature spec; code executes it

- Feature engineering outputs a JSON spec: a name and a formula `op` for each feature.
- A deterministic executor applies the spec. Stateful operations such as binning are fit on the training split only. `as_of_constraint` blanks out values that would not be known at prediction time. Execution errors go back to the model as a bad spec for it to fix.
- A scout grid then tests several feature sets (all features, the top features by mutual information, a set with correlated features removed, and ablations). It trains HistGradientBoosting and RandomForest on each in a process pool, then returns the winning set and a feature-importance ranking across all variants.

### Review is a graph interrupt

- After each step, the graph calls LangGraph's `interrupt()` with a summary. Approving continues the run. Asking for a redo re-runs that step with your comment attached as feedback.
- A redo clears the results of every later step, so nothing downstream is built on stale output.
- The UI's Accept, Redo, and Accept All buttons map directly onto this protocol.

### Budgets

- **Feature engineering:** the system prompt limits it to 3 iterations. This is an instruction to the model, not a check in code.
- **Trained models:** capped at 10 per run in code. After that, `train_with_skill` refuses and tells the agent to finish.
- **Parallelism:** scout workers, scout variants, batch training, and model evaluation are separate settings in [`backend/shared/settings.py`](backend/shared/settings.py). All default low so small workers stay within memory, and the scout grid is skipped for datasets over 200k rows.

### Execution and persistence

- **Routing:** `POST /api/chat` sends a message to the training graph when datasets are linked or a run is being resumed. Otherwise, a small model decides between training and the analysis orchestrator.
- **Jobs:** training runs as a Celery task on Redis. If Celery is unavailable, it falls back to a thread in the API process, unless `TRAINING_REQUIRE_CELERY` is set.
- **Storage:** the model registry lives in Postgres (SQLite locally), and weights live in Cloudflare R2 (local disk when R2 isn't configured). Training-state snapshots larger than `TRAINING_STATE_BLOB_MIN_BYTES` go to object storage, and the database keeps a pointer to them.
