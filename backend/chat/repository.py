import importlib.util as ilu

from backend.shared.settings import get_settings


def get_orchestrator_module():
    global _CACHED_MOD, _CACHED_MTIME_NS
    settings = get_settings()
    agent_path = settings.project_root / "agent.py"
    try:
        mtime_ns = agent_path.stat().st_mtime_ns
    except OSError:
        mtime_ns = None

    if _CACHED_MOD is not None and _CACHED_MTIME_NS == mtime_ns:
        return _CACHED_MOD

    spec = ilu.spec_from_file_location(
        "orchestrator_agent_mod", agent_path
    )
    mod = ilu.module_from_spec(spec)
    spec.loader.exec_module(mod)  # type: ignore[union-attr]
    _CACHED_MOD = mod
    _CACHED_MTIME_NS = mtime_ns
    return _CACHED_MOD


def get_orchestrator_agent():
    return get_orchestrator_module().agent


_CACHED_MOD = None
_CACHED_MTIME_NS = None
