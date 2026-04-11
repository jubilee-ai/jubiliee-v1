"""Celery tasks for long-running training jobs.

Each task updates its state in Redis via Celery's result backend,
which the service layer polls for SSE updates.
"""

import traceback
from datetime import datetime
from typing import Optional

from backend.celery_app import app
from backend.shared.serialization import serialize_state
from backend.shared.training_artifacts import prepare_job_state_for_persistence
from backend.training import repository


@app.task(bind=True, name="training.run_training")
def run_training_task(
    self,
    job_id: str,
    goal: str,
    linked_datasets: Optional[list[str]],
    model_pref: Optional[str],
    use_external_sources: bool = False,
):
    """Execute the full training pipeline as a Celery task.

    Updates task meta with progress so the API can stream status to the frontend.
    """
    from utils import training_dataset_scope

    from backend.training.service import load_and_register_dataset

    with training_dataset_scope(job_id):
        try:
            self.update_state(
                state="PROGRESS",
                meta={"status": "running", "current_step": "data_collection", "job_id": job_id},
            )

            registered_refs = []
            if linked_datasets:
                for dataset_path in linked_datasets:
                    ref = load_and_register_dataset(dataset_path)
                    if ref:
                        registered_refs.append(ref)

            self.update_state(
                state="PROGRESS",
                meta={"status": "running", "current_step": "select_model", "job_id": job_id},
            )

            from agents.training.agent_simple import invoke_simple_training_agent

            final_linked = registered_refs if registered_refs else linked_datasets
            result = invoke_simple_training_agent(
                goal=goal,
                linked_datasets=final_linked,
                user_model_preference=model_pref,
                use_external_sources=use_external_sources,
            )

            try:
                repository.save_run_dataset_links(job_id, result)
            except Exception:
                pass

            serialized = serialize_state(result)
            stored_state = prepare_job_state_for_persistence(job_id, serialized)
            completed_at = datetime.now().isoformat()
            repository.update_training_status(
                job_id,
                {
                    "status": "completed",
                    "progress": 100,
                    "current_step": "completed",
                    "state": stored_state,
                    "completed_at": completed_at,
                },
            )

            return {
                "status": "completed",
                "job_id": job_id,
                "state": stored_state,
                "completed_at": completed_at,
            }

        except Exception as e:
            traceback.print_exc()
            err_at = datetime.now().isoformat()
            try:
                repository.update_training_status(
                    job_id,
                    {
                        "status": "error",
                        "error": str(e),
                        "completed_at": err_at,
                    },
                )
            except Exception:
                pass
            self.update_state(
                state="FAILURE",
                meta={
                    "status": "error",
                    "job_id": job_id,
                    "error": str(e),
                    "completed_at": err_at,
                },
            )
            raise


@app.task(name="training.run_experiment_graph")
def run_experiment_graph_task(
    experiment_id: str,
    goal: str,
    linked_datasets: Optional[list[str]],
    model_pref: Optional[str],
    conversation: Optional[list[dict]] = None,
):
    """Run async experiment graph training on a worker (not in the API process)."""
    from backend.training.service import _run_experiment_graph_task_worker

    _run_experiment_graph_task_worker(
        experiment_id,
        goal,
        linked_datasets,
        model_pref,
        conversation,
    )
