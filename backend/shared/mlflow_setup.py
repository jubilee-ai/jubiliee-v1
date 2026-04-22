"""
MLflow bootstrap: tracking URI + optional autolog for LangChain / sklearn / H2O.

All autolog calls are best-effort (missing integrations or old MLflow versions are ignored).
"""

from __future__ import annotations

import logging
from pathlib import Path

logger = logging.getLogger(__name__)


def register_prompt_from_skill_md(name: str, skill_path: str, tags: dict | None = None) -> str | None:
    """
    Best-effort: register a local SKILL.md as an MLflow prompt (Phase 6).
    Returns prompt version URI or None.
    """
    try:
        from backend.shared.settings import get_settings

        if not (getattr(get_settings(), "MLFLOW_TRACKING_URI", None) or "").strip():
            return None
        import mlflow

        p = Path(skill_path)
        if not p.is_file():
            return None
        text = p.read_text(encoding="utf-8")
        try:
            from mlflow.entities import Prompt  # type: ignore

            pr = Prompt(name=name, template=text, tags=tags or {})
            return mlflow.genai.register_prompt(pr)  # type: ignore[attr-defined]
        except Exception:
            mlflow.set_experiment("jubilee_prompts")
            with mlflow.start_run(run_name=f"prompt_{name}"):
                mlflow.log_text(text, artifact_file=f"prompts/{name}.md")
            return None
    except Exception as e:
        logger.debug("register_prompt_from_skill_md: %s", e)
        return None


def _register_jubilee_prompts_best_effort() -> None:
    """Phase 6: best-effort register SKILL.md files as MLflow prompts / artifacts."""
    try:
        from backend.shared.settings import get_settings

        root = get_settings().project_root
    except Exception:
        return
    skill_paths = [
        root / "agents/subagents/evaluator/SKILL.md",
        root / "agents/subagents/explainer/SKILL.md",
        root / "agents/subagents/deployer/SKILL.md",
        root / "agents/subagents/monitor/SKILL.md",
        root / "agents/training/skills/h2o_automl/SKILL.md",
    ]
    for p in skill_paths:
        try:
            register_prompt_from_skill_md(
                name=f"jubilee_{p.parent.name}_{p.stem}",
                skill_path=str(p),
                tags={"source": "jubilee", "path": str(p.relative_to(root))},
            )
        except Exception:
            continue


def bootstrap_mlflow() -> None:
    """Set tracking URI and enable framework autolog when configured."""
    try:
        from backend.shared.settings import get_settings
    except Exception:
        return

    settings = get_settings()
    uri = getattr(settings, "MLFLOW_TRACKING_URI", None) or ""
    if not uri.strip():
        return

    try:
        import mlflow
    except ImportError:
        logger.warning("MLflow not installed; skipping MLflow bootstrap")
        return

    mlflow.set_tracking_uri(uri.strip())

    if not getattr(settings, "MLFLOW_ENABLE_AUTOLOG", True):
        _register_jubilee_prompts_best_effort()
        return

    _safe_autolog("mlflow.sklearn", "autolog", log_models=False, log_input_examples=False)
    _safe_autolog("mlflow.xgboost", "autolog", log_models=False)
    _safe_autolog("mlflow.h2o", "autolog", log_models=False)

    # LangChain / LangGraph (names vary by MLflow version)
    try:
        import mlflow.langchain as mlc  # type: ignore

        if hasattr(mlc, "autolog"):
            mlc.autolog()
    except Exception as e:
        logger.debug("mlflow.langchain.autolog skipped: %s", e)

    try:
        import mlflow.langgraph as mlg  # type: ignore

        if hasattr(mlg, "autolog"):
            mlg.autolog()
    except Exception as e:
        logger.debug("mlflow.langgraph.autolog skipped: %s", e)

    # Anthropic / OpenAI tracing (GenAI stack)
    for mod in ("mlflow.anthropic", "mlflow.openai"):
        _safe_autolog(mod, "autolog")

    # Phase 6: optional GenAI hooks (no-op if unavailable)
    _maybe_enable_genai_eval_hooks()
    _register_jubilee_prompts_best_effort()


def _safe_autolog(module_name: str, attr: str = "autolog", **kwargs) -> None:
    try:
        mod = __import__(module_name, fromlist=[attr])
        fn = getattr(mod, attr, None)
        if callable(fn):
            fn(**kwargs) if kwargs else fn()
    except Exception as e:
        logger.debug("%s.%s skipped: %s", module_name, attr, e)


def _maybe_enable_genai_eval_hooks() -> None:
    """Register optional Agent GPA / trace-aware eval when MLflow GenAI extras exist."""
    try:
        import mlflow.genai as genai  # type: ignore

        if hasattr(genai, "configure"):
            genai.configure()  # type: ignore[call-arg]
    except Exception:
        pass
