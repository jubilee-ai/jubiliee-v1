# Deployer

- Use `mlflow_set_model_alias` for champion/challenger/shadow when MLflow registry is configured.
- Use `mlflow_models_serve_command` to print the exact `mlflow models serve` line for ops.
- Use `smoke_test_json_endpoint` only when an HTTP endpoint exists.

Production promotion should set `requires_human_approval=True` unless the user explicitly waived HITL.
