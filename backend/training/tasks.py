"""Celery tasks for long-running training jobs.

Each task updates its state in Redis via Celery's result backend,
which the service layer polls for SSE updates.
"""

import traceback
from datetime import datetime
from typing import Optional

from backend.celery_app import app
from backend.shared.serialization import serialize_state


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
    from backend.training.service import load_and_register_dataset

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
            from backend.training.repository import save_run_dataset_links
            save_run_dataset_links(job_id, result)
        except Exception:
            pass

        return {
            "status": "completed",
            "job_id": job_id,
            "state": serialize_state(result),
            "completed_at": datetime.now().isoformat(),
        }

    except Exception as e:
        traceback.print_exc()
        self.update_state(
            state="FAILURE",
            meta={
                "status": "error",
                "job_id": job_id,
                "error": str(e),
                "completed_at": datetime.now().isoformat(),
            },
        )
        raise
