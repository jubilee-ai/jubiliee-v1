"""Parameter Extraction Tool

Dynamically extracts constructor parameters from any Python class —
sklearn estimators, PyTorch modules/optimizers/schedulers, or any other
framework registered via `register_search_namespace()`.

Uses inspect + docstring parsing for well-documented classes, and falls back
to a small LLM to interpret raw source code when documentation is sparse.

Results are cached per (class_name, source_hash) so repeated calls are free.
"""

import hashlib
import importlib
import inspect
import json
import re
from pathlib import Path
from typing import Optional

_CACHE_DIR = Path(__file__).parent / ".param_cache"

_SKIP_PARAMS = frozenset({
    "random_state", "n_jobs", "verbose", "warm_start", "copy_X",
    "fit_intercept", "normalize", "copy", "n_features_in_",
    "device", "dtype", "factory_kwargs",
})

_LOG_SCALE_HINTS = frozenset({
    "alpha", "C", "gamma", "learning_rate", "lr", "epsilon",
    "lambda", "reg_alpha", "reg_lambda", "l2_regularization",
    "weight_decay", "eps", "momentum",
})


# ── Namespace registry ──────────────────────────────────────────────────

_SEARCH_REGISTRIES: list[dict] = []


def register_search_namespace(
    module_path: str,
    base_class: str | None = None,
    discovery_fn: str | None = None,
):
    """Register a namespace for automatic class resolution.

    Args:
        module_path: Python module to search (e.g. "torch.nn", "sklearn")
        base_class: Fully-qualified base class to filter by
                    (e.g. "torch.nn.Module", "torch.optim.Optimizer")
        discovery_fn: Special discovery function name in the module
                      (e.g. "all_estimators" for sklearn.utils)
    """
    _SEARCH_REGISTRIES.append({
        "module": module_path,
        "base_class": base_class,
        "discovery_fn": discovery_fn,
    })


def _auto_register_defaults():
    """Auto-register available frameworks at import time."""
    try:
        from sklearn.utils import all_estimators  # noqa: F401
        register_search_namespace("sklearn", discovery_fn="all_estimators")
    except ImportError:
        pass

    try:
        import torch  # noqa: F401
        register_search_namespace("torch.nn", base_class="torch.nn.Module")
        register_search_namespace("torch.optim", base_class="torch.optim.Optimizer")
        register_search_namespace(
            "torch.optim.lr_scheduler",
            base_class="torch.optim.lr_scheduler.LRScheduler",
        )
    except ImportError:
        pass


_auto_register_defaults()


# ── Docstring parsing ───────────────────────────────────────────────────

def _parse_numpydoc_params(docstring: str) -> dict[str, dict]:
    """Parse sklearn-style numpydoc Parameters section."""
    match = re.search(
        r"Parameters\s*\n\s*-{3,}\s*\n(.*?)(?:\n\s*(?:Returns|Attributes|Notes|Examples|See Also|References|Raises)\s*\n\s*-{3,}|\Z)",
        docstring,
        re.DOTALL,
    )
    if not match:
        return {}

    block = match.group(1)
    params: dict[str, dict] = {}

    entries = re.split(r"\n(?=\s{0,4}\w+\s*:)", "\n" + block)
    for entry in entries:
        entry = entry.strip()
        if not entry:
            continue

        header = re.match(r"^(\w+)\s*:\s*(.+?)$", entry, re.MULTILINE)
        if not header:
            continue

        name = header.group(1)
        type_str = header.group(2).strip()
        desc_lines = [ln.strip() for ln in entry.split("\n")[1:] if ln.strip()]
        description = " ".join(desc_lines)

        info: dict = {"type_str": type_str, "description": description}

        choice_match = re.search(r"\{([^}]+)\}", type_str)
        if choice_match:
            raw = choice_match.group(1)
            choices = [c.strip().strip("'\"") for c in raw.split(",")]
            info["choices"] = [None if c == "None" else c for c in choices]
            info["param_kind"] = "categorical"
        elif re.match(r"^float\b", type_str):
            info["param_kind"] = "float"
        elif re.match(r"^int\b", type_str):
            info["param_kind"] = "int"
        elif re.match(r"^bool\b", type_str):
            info["param_kind"] = "bool"
        else:
            info["param_kind"] = "other"

        params[name] = info

    return params


def _parse_pytorch_args(docstring: str) -> dict[str, dict]:
    """Parse PyTorch-style 'Args:' docstring sections.

    Handles formats like:
        Args:
            in_features (int): size of each input sample.
            bias (bool): If ``False``, no additive bias. Default: ``True``.
    """
    match = re.search(
        r"(?:^|\n)\s*Args:\s*\n(.*?)(?:\n\s*(?:Returns|Attributes|Note|Examples|Shape|Keyword|Raises|See|Warning)|\Z)",
        docstring,
        re.DOTALL,
    )
    if not match:
        return {}

    block = match.group(1)
    params: dict[str, dict] = {}

    # Detect indentation level from first parameter line
    indent_match = re.search(r"\n(\s+)\w+\s*[\(:]", "\n" + block)
    indent = len(indent_match.group(1)) if indent_match else 4
    # Split only at lines with the parameter indentation level (not deeper continuations)
    # Exclude URL-like entries (https:, http:) via negative lookahead
    entries = re.split(rf"\n(?=\s{{{indent}}}(?!https?:)\w+\s*[\(:])", "\n" + block)

    for entry in entries:
        entry = entry.strip()
        if not entry:
            continue

        header = re.match(
            r"^(\w+)\s*(?:\(([^)]*)\))?\s*[:\u2013\u2014-]\s*(.*)$",
            entry,
            re.DOTALL,
        )
        if not header:
            continue

        name = header.group(1)
        type_str = (header.group(2) or "").strip()
        raw_desc = header.group(3).strip()
        desc_lines = [ln.strip() for ln in raw_desc.split("\n") if ln.strip()]
        description = " ".join(desc_lines)

        info: dict = {"type_str": type_str, "description": description}

        choice_match = re.search(r"\{([^}]+)\}", type_str + " " + description)
        if choice_match:
            raw = choice_match.group(1)
            choices = [c.strip().strip("'\"") for c in raw.split(",")]
            info["choices"] = [None if c == "None" else c for c in choices]
            info["param_kind"] = "categorical"
        elif re.match(r"^float\b", type_str):
            info["param_kind"] = "float"
        elif re.match(r"^int\b", type_str):
            info["param_kind"] = "int"
        elif re.match(r"^bool\b", type_str):
            info["param_kind"] = "bool"
        else:
            info["param_kind"] = "other"

        params[name] = info

    return params


def _parse_docstring(docstring: str) -> dict[str, dict]:
    """Try numpydoc first, then PyTorch Args-style."""
    result = _parse_numpydoc_params(docstring)
    if result:
        return result
    return _parse_pytorch_args(docstring)


# ── LLM fallback ────────────────────────────────────────────────────────

def _llm_describe_params(
    class_name: str,
    init_source: str,
    signature_info: dict[str, dict],
) -> dict[str, dict]:
    """Use a small LLM to generate parameter descriptions from source code."""
    from openai import OpenAI

    client = OpenAI()

    sig_lines = "\n".join(
        f"  {name}: default={info['default']}"
        for name, info in signature_info.items()
    )

    prompt = (
        f"Analyze this Python class constructor and describe each parameter "
        f"for hyperparameter tuning.\n\n"
        f"Class: {class_name}\n\n"
        f"Signature:\n{sig_lines}\n\n"
        f"Source code:\n{init_source[:3000]}\n\n"
        f"For each parameter, return a JSON object:\n"
        f'{{\n'
        f'  "param_name": {{\n'
        f'    "type": "float" | "int" | "categorical" | "bool" | "other",\n'
        f'    "description": "one-line description",\n'
        f'    "tunable": true | false,\n'
        f'    "choices": ["a", "b"]  // only if categorical\n'
        f'    "range": {{"low": number, "high": number}}  // only if numeric\n'
        f'    "scale": "log" | "linear"\n'
        f"  }}\n"
        f"}}\n\n"
        f"Return ONLY valid JSON, no markdown fences."
    )

    response = client.chat.completions.create(
        model="gpt-4.1-nano",
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
    )

    text = response.choices[0].message.content.strip()
    if text.startswith("```"):
        text = re.sub(r"^```\w*\n?", "", text)
        text = re.sub(r"\n?```$", "", text)

    return json.loads(text)


# ── Cache helpers ────────────────────────────────────────────────────────

def _cache_key(cls: type) -> str:
    try:
        source = inspect.getsource(cls)
        source_hash = hashlib.sha256(source.encode()).hexdigest()[:12]
    except (OSError, TypeError):
        source_hash = "nosource"
    return f"{cls.__name__}_{source_hash}"


def _load_cache(key: str) -> dict | None:
    path = _CACHE_DIR / f"{key}.json"
    if path.exists():
        try:
            return json.loads(path.read_text())
        except (json.JSONDecodeError, OSError):
            return None
    return None


def _save_cache(key: str, data: dict):
    try:
        _CACHE_DIR.mkdir(parents=True, exist_ok=True)
        (_CACHE_DIR / f"{key}.json").write_text(
            json.dumps(data, indent=2, default=str)
        )
    except OSError:
        pass


# ── Builder helpers ──────────────────────────────────────────────────────

def _infer_kind(default) -> str:
    if isinstance(default, bool):
        return "bool"
    if isinstance(default, float):
        return "float"
    if isinstance(default, int):
        return "int"
    if isinstance(default, str):
        return "categorical"
    return "other"


def _build_from_docstring(
    signature_info: dict[str, dict],
    doc_params: dict[str, dict],
) -> dict[str, dict]:
    """Merge signature defaults with parsed docstring info."""
    result: dict[str, dict] = {}
    for name, sig_info in signature_info.items():
        default = sig_info["default"]
        doc = doc_params.get(name, {})

        kind = doc.get("param_kind", _infer_kind(default))
        tunable = name not in _SKIP_PARAMS and kind in ("float", "int", "categorical")

        entry: dict = {
            "type": kind,
            "default": default,
            "tunable": tunable,
            "description": doc.get("description", ""),
        }

        if "choices" in doc:
            entry["choices"] = [c for c in doc["choices"] if c is not None]

        if kind == "float" and isinstance(default, (int, float)) and default > 0:
            entry["scale"] = "log" if name in _LOG_SCALE_HINTS else "linear"

        result[name] = entry
    return result


def _build_from_signature_only(signature_info: dict[str, dict]) -> dict[str, dict]:
    """Minimal extraction using only signature defaults."""
    result: dict[str, dict] = {}
    for name, sig_info in signature_info.items():
        default = sig_info["default"]
        kind = _infer_kind(default)
        result[name] = {
            "type": kind,
            "default": default,
            "tunable": name not in _SKIP_PARAMS and kind in ("float", "int", "categorical"),
            "description": "",
        }
    return result


# ── Class resolution ─────────────────────────────────────────────────────

def _resolve_from_registry(class_name: str, registry: dict):
    """Search a single registry entry for a class by name."""
    module_path = registry["module"]
    base_class_path = registry.get("base_class")
    discovery_fn = registry.get("discovery_fn")

    if discovery_fn == "all_estimators":
        try:
            from sklearn.utils import all_estimators
            for name, cls in all_estimators():
                if name == class_name:
                    return cls
        except ImportError:
            pass
        return None

    try:
        mod = importlib.import_module(module_path)
    except (ImportError, ModuleNotFoundError):
        return None

    candidate = getattr(mod, class_name, None)
    if candidate is None or not inspect.isclass(candidate):
        return None

    if base_class_path:
        parts = base_class_path.rsplit(".", 1)
        try:
            base_mod = importlib.import_module(parts[0])
            base_cls = getattr(base_mod, parts[1])
            if not issubclass(candidate, base_cls):
                return None
        except (ImportError, AttributeError, TypeError):
            return None

    return candidate


def _resolve_class(class_name: str, module_hint: str | None = None):
    """Import and return the class object.

    Resolution order:
    1. Fully-qualified path (e.g. "torch.nn.Linear")
    2. module_hint (e.g. module_hint="torch.optim")
    3. Registered namespace scan (sklearn, torch.nn, torch.optim, ...)
    """
    if "." in class_name:
        module_path, cls_name = class_name.rsplit(".", 1)
        try:
            return getattr(importlib.import_module(module_path), cls_name, None)
        except (ImportError, ModuleNotFoundError):
            return None

    if module_hint:
        try:
            cls = getattr(importlib.import_module(module_hint), class_name, None)
            if cls and inspect.isclass(cls):
                return cls
        except (ImportError, ModuleNotFoundError):
            pass

    for registry in _SEARCH_REGISTRIES:
        cls = _resolve_from_registry(class_name, registry)
        if cls is not None:
            return cls

    return None


# ── Core extraction function ─────────────────────────────────────────────

def extract_estimator_params(
    class_name: str,
    module_hint: str | None = None,
) -> dict:
    """Extract parameter metadata from any Python estimator class.

    Uses inspect.signature for names/defaults, numpydoc parsing for types
    and choices when docs are rich, and a small LLM (gpt-4.1-nano) to
    interpret source code when documentation is sparse or absent.

    Results are cached to disk keyed on (class_name, source_hash).
    """
    cls = _resolve_class(class_name, module_hint)
    if cls is None:
        return {"_error": f"Could not resolve class '{class_name}'"}

    cache_key = _cache_key(cls)
    cached = _load_cache(cache_key)
    if cached is not None:
        return cached

    sig = inspect.signature(cls.__init__)
    signature_info: dict[str, dict] = {}
    for name, p in sig.parameters.items():
        if name == "self":
            continue
        signature_info[name] = {
            "default": p.default if p.default is not inspect.Parameter.empty else "__REQUIRED__",
        }

    docstring = cls.__doc__ or ""
    doc_params = _parse_docstring(docstring) if docstring else {}

    try:
        init_source = inspect.getsource(cls.__init__)
    except (OSError, TypeError):
        init_source = None

    has_rich_docs = len(doc_params) >= len(signature_info) * 0.5

    if has_rich_docs:
        result = _build_from_docstring(signature_info, doc_params)
    elif init_source:
        try:
            result = _llm_describe_params(cls.__name__, init_source, signature_info)
        except Exception:
            result = _build_from_signature_only(signature_info)
    else:
        result = _build_from_signature_only(signature_info)

    for name in list(result.keys()):
        if name.startswith("_"):
            continue
        if name in _SKIP_PARAMS and isinstance(result[name], dict):
            result[name]["tunable"] = False

    result["_class"] = f"{cls.__module__}.{cls.__name__}"
    result["_source"] = "docstring" if has_rich_docs else ("llm" if init_source else "signature")

    _save_cache(cache_key, result)
    return result


# ── LangChain tool wrapper ───────────────────────────────────────────────


def _build_tool():
    """Lazy-build the LangChain tool to avoid import errors when langchain
    is not installed (e.g. during standalone testing of the core function)."""
    from langchain.tools import tool as lc_tool
    from pydantic import BaseModel, Field

    class ExtractParamsInput(BaseModel):
        model_config = {"extra": "forbid"}

        class_name: str = Field(
            description=(
                "Class name to inspect. Can be a simple name like "
                "'RandomForestClassifier', 'Linear', or 'AdamW', or a fully "
                "qualified path like 'torch.nn.Linear'. Auto-resolved via "
                "registered namespaces (sklearn, torch.nn, torch.optim, etc.)."
            )
        )
        module_hint: Optional[str] = Field(
            default=None,
            description=(
                "Optional module path for resolution when class_name is not "
                "fully qualified (e.g. 'torch.optim', 'torch.nn', 'xgboost')."
            ),
        )

    @lc_tool("extract_estimator_params", args_schema=ExtractParamsInput)
    def extract_estimator_params_tool(
        class_name: str,
        module_hint: Optional[str] = None,
    ) -> str:
        """Extract constructor parameters and tuning metadata for any class.

        Works with sklearn estimators, PyTorch nn.Module layers, optimizers,
        LR schedulers, and any class registered via register_search_namespace().
        Returns each parameter's type, default, valid choices (if categorical),
        suggested scale (log/linear), and a description.
        """
        result = extract_estimator_params(class_name, module_hint)
        return json.dumps(result, indent=2, default=str)

    return extract_estimator_params_tool


try:
    extract_estimator_params_tool = _build_tool()
except ImportError:
    extract_estimator_params_tool = None  # type: ignore[assignment]


# ── CLI entrypoint ───────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Extract constructor parameter metadata for any class "
        "(sklearn estimators, PyTorch modules/optimizers/schedulers, etc.).",
    )
    parser.add_argument(
        "class_name",
        help=(
            "Class name, e.g. 'RandomForestClassifier', 'torch.nn.Linear', "
            "'AdamW', or 'ReduceLROnPlateau'."
        ),
    )
    parser.add_argument(
        "--module",
        default=None,
        help="Module hint for resolution (e.g. 'torch.optim', 'torch.nn', 'xgboost').",
    )
    args = parser.parse_args()

    result = extract_estimator_params(args.class_name, module_hint=args.module)
    print(json.dumps(result, indent=2, default=str))
