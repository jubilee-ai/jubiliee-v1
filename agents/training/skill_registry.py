"""
Skill Registry for Training Models

Auto-discovers model training skills from the skills/ directory.
Provides tools for the training agent to list, load, and execute skills.
"""

import importlib.util
import json
import sys
from pathlib import Path
from typing import Optional

from langchain_core.tools import tool
from pydantic import BaseModel, Field

SKILLS_DIR = Path(__file__).parent / "skills"

_ROOT_DIR = Path(__file__).parent.parent.parent
_TOOLS_DIR = _ROOT_DIR / "tools" / "models-tools" / "training"
_DATA_TOOLS_DIR = _ROOT_DIR / "tools" / "data-tools"


def _ensure_paths():
    for p in [str(_TOOLS_DIR), str(_DATA_TOOLS_DIR)]:
        if p not in sys.path:
            sys.path.insert(0, p)


def _discover_skills() -> dict[str, dict]:
    """Auto-discover all skills in the skills/ directory."""
    skills = {}
    if not SKILLS_DIR.exists():
        return skills

    for skill_dir in sorted(SKILLS_DIR.iterdir()):
        if not skill_dir.is_dir():
            continue
        meta_path = skill_dir / "meta.json"
        if not meta_path.exists():
            continue

        with open(meta_path) as f:
            meta = json.load(f)
        meta["_dir"] = str(skill_dir)
        skills[meta["name"]] = meta

    return skills


def _load_skill_prompt(skill_name: str) -> str:
    """Load a skill's SKILL.md content."""
    skills = _discover_skills()
    if skill_name not in skills:
        available = list(skills.keys())
        raise ValueError(f"Skill '{skill_name}' not found. Available: {available}")

    skill_md = Path(skills[skill_name]["_dir"]) / "SKILL.md"
    if not skill_md.exists():
        raise ValueError(f"SKILL.md not found for skill '{skill_name}'")

    return skill_md.read_text()


def _run_skill(skill_name: str, params: dict) -> str:
    """Execute a skill's training script."""
    _ensure_paths()

    skills = _discover_skills()
    if skill_name not in skills:
        available = list(skills.keys())
        raise ValueError(f"Skill '{skill_name}' not found. Available: {available}")

    train_py = Path(skills[skill_name]["_dir"]) / "train.py"
    if not train_py.exists():
        raise ValueError(f"train.py not found for skill '{skill_name}'")

    spec = importlib.util.spec_from_file_location(
        f"skill_{skill_name}_train", str(train_py)
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    if not hasattr(module, "run"):
        raise ValueError(
            f"Skill '{skill_name}' train.py must define a run(params) function"
        )

    return module.run(params)


# ── Public helpers for use by training.py and agent_simple.py ────────────


def get_available_models_for_task(task_type: str) -> list[dict]:
    """Get skills available for a given task type."""
    skills = _discover_skills()
    return [
        {"skill": name, "label": meta["label"]}
        for name, meta in skills.items()
        if task_type in meta.get("task_types", [])
    ]


def get_all_skill_names() -> set[str]:
    """Get all registered skill names."""
    return set(_discover_skills().keys())


def get_all_skills_for_selection() -> dict[str, dict]:
    """Return enriched metadata for every skill, used by the model-selection step.

    Each value contains: name, label, brief, task_types,
    when_to_use, when_not_to_use, and aliases.
    """
    return _discover_skills()


def build_alias_map() -> dict[str, str]:
    """Build a mapping from every alias (lowercased) to its canonical skill name.

    Each skill's meta.json may contain an ``aliases`` list.  The canonical
    ``name`` is always included as an alias automatically.
    """
    alias_map: dict[str, str] = {}
    for name, meta in _discover_skills().items():
        alias_map[name] = name
        for alias in meta.get("aliases", []):
            alias_map[alias.lower()] = name
    return alias_map


# ── LangChain Tools ─────────────────────────────────────────────────────


class ListSkillsInput(BaseModel):
    task_type: Optional[str] = Field(
        default=None,
        description="Filter by task type: 'classification', 'regression', or 'survival'. "
        "If None, lists all available skills.",
    )


@tool("list_training_skills", args_schema=ListSkillsInput)
def list_training_skills_tool(task_type: Optional[str] = None) -> str:
    """List all available model training skills.

    Returns the name, supported task types, and brief description for each skill.
    Use get_skill_prompt to read the full documentation for a specific skill
    before training.
    """
    skills = _discover_skills()

    if task_type:
        skills = {
            k: v for k, v in skills.items() if task_type in v.get("task_types", [])
        }

    if not skills:
        suffix = f" for task type '{task_type}'" if task_type else ""
        return f"No training skills found{suffix}."

    lines = [f"Available Training Skills ({len(skills)}):", ""]
    for name, meta in skills.items():
        tasks = ", ".join(meta.get("task_types", []))
        lines.append(f"- **{meta['label']}** (skill: `{name}`)")
        lines.append(f"  Tasks: {tasks}")
        lines.append(f"  {meta.get('brief', '')}")
        lines.append("")

    lines.append(
        "Use get_skill_prompt(skill_name) to see full details and parameters."
    )
    return "\n".join(lines)


class GetSkillPromptInput(BaseModel):
    skill_name: str = Field(
        description="Name of the skill to load (from list_training_skills)"
    )


@tool("get_skill_prompt", args_schema=GetSkillPromptInput)
def get_skill_prompt_tool(skill_name: str) -> str:
    """Load the full prompt for a training skill.

    Returns the skill's documentation including when to use,
    required/optional parameters, hyperparameter tuning guidance,
    feature engineering guidance, and examples.

    Read this before calling train_with_skill.
    """
    try:
        return _load_skill_prompt(skill_name)
    except ValueError as e:
        return f"Error: {e}"


class TrainWithSkillInput(BaseModel):
    model_config = {"extra": "allow"}

    skill_name: str = Field(
        description="Name of the skill to use (from list_training_skills)"
    )
    params: Optional[dict] = Field(
        default=None,
        description="Training parameters as a JSON object. Must include "
        "'model_name' and 'train_dataset_ref'. See the skill documentation for "
        "required and optional parameters for each skill.",
    )


@tool("train_with_skill", args_schema=TrainWithSkillInput)
def train_with_skill_tool(skill_name: str, params: Optional[dict] = None, **kwargs) -> str:
    """Train a model using a specific skill.

    Pass skill_name and a params dict with model_name, train_dataset_ref,
    target_column, and any model-specific hyperparameters.
    """
    if params is None:
        params = {}
    # If the LLM passed training parameters as top-level keys instead of
    # nesting them inside params, merge them in.
    if kwargs:
        params = {**kwargs, **params}
    try:
        return _run_skill(skill_name, params)
    except Exception as e:
        return f"Training failed: {e}"


# ── Convenience exports ─────────────────────────────────────────────────

SKILL_TOOLS = [
    list_training_skills_tool,
    get_skill_prompt_tool,
    train_with_skill_tool,
]
