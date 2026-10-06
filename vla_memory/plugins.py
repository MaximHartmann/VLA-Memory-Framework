"""Component plug-ins: which class implements the VLM backend or the policy transport, chosen by name on the command line.

A component is named in one of three ways, tried in this order:
    a registered name     the built-ins (vlm.VLM_BACKENDS: openai; transport.TRANSPORTS: openpi) and names added at run time
                          with vlm.register_vlm_backend / transport.register_transport (a launcher that imports its class first)
    an entry point        a name an installed package declares in its pyproject.toml, in the group vla_memory.vlm_backends or
                          vla_memory.transports:
                              [project.entry-points."vla_memory.vlm_backends"]
                              gemini = "my_package.gemini:GeminiVLM"
    an import path        "module:attribute" or "module.attribute", e.g. my_package.policy:MyTransport
construct() calls the class with the keyword arguments the proxy knows (a VLM backend: base_url, model, api_key,
chat_template_kwargs; a transport: host, port) as far as its signature declares them (all of them if it takes **kwargs), plus
the options given with --vlm-arg / --transport-arg KEY=VALUE, which are always passed. An option's value is read as JSON when it
parses (60, true, [10, 8], {"seed": 3}), else as a string.
"""
from __future__ import annotations

import argparse
import importlib
import inspect
import json
from typing import Any, Callable


def key_value(text: str) -> tuple[str, Any]:
    """'KEY=VALUE' -> (KEY, VALUE), the value parsed as JSON when it parses, else the string (the argparse type of the options)."""
    key, sep, value = text.partition("=")
    if not sep or not key.strip():
        raise argparse.ArgumentTypeError(f"expected KEY=VALUE, got {text!r}")
    try:
        return key.strip(), json.loads(value)
    except json.JSONDecodeError:
        return key.strip(), value


def import_object(path: str):
    """'module:attribute' (the attribute may be dotted) or 'module.attribute' -> the object."""
    mod, _, attr = path.partition(":") if ":" in path else path.rpartition(".")
    if not mod or not attr:
        raise ValueError(f"not an import path (module:attribute): {path!r}")
    obj = importlib.import_module(mod)
    for part in attr.split("."):
        obj = getattr(obj, part)
    return obj


def entry_points(group: str) -> dict:
    """The entry points installed packages declare in `group`, by name (not loaded)."""
    try:
        from importlib.metadata import entry_points as _eps
        return {ep.name: ep for ep in _eps(group=group)}
    except Exception:   # broken package metadata must not stop the proxy
        return {}


def resolve(spec, registry: dict, group: str, kind: str):
    """The class (or factory) `spec` names: a registered name, an entry point of `group`, or an import path. A class or a
    callable passes through unchanged (programmatic use). Raises ValueError with the known names."""
    if not isinstance(spec, str):
        return spec
    if spec in registry:
        entry = registry[spec]
        return import_object(entry) if isinstance(entry, str) else entry
    eps = entry_points(group)
    if spec in eps:
        return eps[spec].load()
    if ":" in spec or "." in spec:
        try:
            return import_object(spec)
        except (ImportError, AttributeError, ValueError) as e:
            raise ValueError(f"{kind} {spec!r}: cannot import it ({e})") from e
    known = ", ".join(sorted(set(registry) | set(eps)))
    raise ValueError(f"unknown {kind} {spec!r} (known: {known}; or an import path module:Class)")


def construct(factory: Callable, standard: dict, options: dict | None = None, kind: str = "component"):
    """factory(**kwargs): the `standard` arguments its signature declares (all of them if it takes **kwargs) and every one of
    `options` (an option the signature does not take is a ValueError naming the ones it takes)."""
    options = dict(options or {})
    try:
        params = inspect.signature(factory).parameters
    except (TypeError, ValueError):   # no signature (a builtin): pass everything
        params = None
    if params is None or any(p.kind is inspect.Parameter.VAR_KEYWORD for p in params.values()):
        return factory(**{**standard, **options})
    names = [n for n, p in params.items() if p.kind in (inspect.Parameter.POSITIONAL_OR_KEYWORD, inspect.Parameter.KEYWORD_ONLY)]
    unknown = [k for k in options if k not in names]
    if unknown:
        raise ValueError(f"{kind} takes no option(s) {unknown}; it takes {names}")
    return factory(**{**{k: v for k, v in standard.items() if k in names}, **options})
