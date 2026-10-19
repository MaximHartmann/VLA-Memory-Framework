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
    key, separator, value = text.partition("=")
    if not separator or not key.strip():
        raise argparse.ArgumentTypeError(f"expected KEY=VALUE, got {text!r}")
    try:
        return key.strip(), json.loads(value)
    except json.JSONDecodeError:
        return key.strip(), value


def import_object(path: str):
    """'module:attribute' (the attribute may be dotted) or 'module.attribute' -> the object."""
    if ":" in path:
        module_name, _, attribute_path = path.partition(":")
    else:
        module_name, _, attribute_path = path.rpartition(".")
    if not module_name or not attribute_path:
        raise ValueError(f"not an import path (module:attribute): {path!r}")
    target = importlib.import_module(module_name)
    for attribute in attribute_path.split("."):
        target = getattr(target, attribute)
    return target


def entry_points(group: str) -> dict:
    """The entry points installed packages declare in `group`, by name (not loaded)."""
    try:
        from importlib.metadata import entry_points as installed_entry_points
        return {entry_point.name: entry_point for entry_point in installed_entry_points(group=group)}
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
    installed = entry_points(group)
    if spec in installed:
        return installed[spec].load()
    if ":" in spec or "." in spec:
        try:
            return import_object(spec)
        except (ImportError, AttributeError, ValueError) as error:
            raise ValueError(f"{kind} {spec!r}: cannot import it ({error})") from error
    known = ", ".join(sorted(set(registry) | set(installed)))
    raise ValueError(f"unknown {kind} {spec!r} (known: {known}; or an import path module:Class)")


def _keyword_parameters(factory: Callable) -> list[str] | None:
    """The names a factory accepts as keyword arguments; None when it accepts any (it takes **kwargs, or has no signature)."""
    try:
        parameters = inspect.signature(factory).parameters
    except (TypeError, ValueError):   # no signature (a builtin): pass everything
        return None
    if any(parameter.kind is inspect.Parameter.VAR_KEYWORD for parameter in parameters.values()):
        return None
    keyword_kinds = (inspect.Parameter.POSITIONAL_OR_KEYWORD, inspect.Parameter.KEYWORD_ONLY)
    return [name for name, parameter in parameters.items() if parameter.kind in keyword_kinds]


def construct(factory: Callable, standard: dict, options: dict | None = None, kind: str = "component"):
    """factory(**kwargs): the `standard` arguments its signature declares (all of them if it takes **kwargs) and every one of
    `options` (an option the signature does not take is a ValueError naming the ones it takes)."""
    options = dict(options or {})
    accepted = _keyword_parameters(factory)
    if accepted is None:
        return factory(**{**standard, **options})
    unknown = [name for name in options if name not in accepted]
    if unknown:
        raise ValueError(f"{kind} takes no option(s) {unknown}; it takes {accepted}")
    declared_standard = {name: value for name, value in standard.items() if name in accepted}
    return factory(**{**declared_standard, **options})
