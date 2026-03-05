import importlib.util as ilu
from functools import lru_cache

from backend.shared.settings import get_settings


@lru_cache(maxsize=1)
def get_orchestrator_module():
    settings = get_settings()
    spec = ilu.spec_from_file_location(
        "orchestrator_agent_mod", settings.project_root / "agent.py"
    )
    mod = ilu.module_from_spec(spec)
    spec.loader.exec_module(mod)  # type: ignore[union-attr]
    return mod


def get_orchestrator_agent():
    return get_orchestrator_module().agent
