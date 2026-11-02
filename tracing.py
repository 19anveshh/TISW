"""LangSmith spans for the stages LangChain does not instrument.

The LangGraph run and the Groq call already emit their own runs. Retrieval,
parsing, chunking and embedding are plain functions, so they need spans of their
own to appear in the same trace.

Two rules hold everywhere here: an inactive trace never imports LangSmith, and a
LangSmith problem can never fail a chat turn, an upload, or an index build.
"""

from __future__ import annotations

import contextlib
import contextvars
import functools
import hashlib
import os
import re
import sys
from typing import Any, Callable, Iterator, Optional

from runtime_config import is_offline_mode

MAX_TEXT = 220

_SECRET_PATTERN = re.compile(r"(?i)(?:gsk_[A-Za-z0-9_-]+|lsv2_[A-Za-z0-9_-]+)")

# Set only while a real root run is open, so nested stage spans know they have a
# trace to attach to. Threads started with threading.Thread begin with the
# default (False) and must open their own root.
_SPANS_ENABLED = contextvars.ContextVar("tisw_spans_enabled", default=False)


def key_configured() -> bool:
    return bool(os.getenv("LANGSMITH_API_KEY", "").strip())


def can_trace() -> bool:
    """Tracing is possible: a server-side key exists and we are not offline."""
    return not is_offline_mode() and key_configured()


def spans_enabled() -> bool:
    return bool(_SPANS_ENABLED.get())


def _secret_free(value: Any) -> str:
    return _SECRET_PATTERN.sub("[REDACTED]", str(value))


def redact(value: Any, *, limit: int = MAX_TEXT) -> str:
    """Secret-free, length-bounded text safe to store as a span payload."""
    text = _secret_free(value)
    return text if len(text) <= limit else text[:limit] + "…"


def preview(value: Any, *, limit: int = 160) -> str:
    """The same guarantee with whitespace collapsed, for one-line span payloads."""
    text = " ".join(_secret_free(value).split())
    return text if len(text) <= limit else text[:limit] + "…"


def hash_id(value: Any) -> str:
    """A stable, non-reversible identifier, so a session token stays local."""
    return hashlib.sha256(str(value).encode("utf-8")).hexdigest()[:16]


def arg_at(args: tuple, kwargs: dict[str, Any], index: int, name: str, default: Any = None) -> Any:
    """Positional-or-keyword access for span payload projections."""
    if len(args) > index:
        return args[index]
    return kwargs.get(name, default)


def add_outputs(run: Any, outputs: dict[str, Any]) -> None:
    if run is None:
        return
    try:
        run.add_outputs(dict(outputs))
    except Exception as exc:
        _warn("TRACE OUTPUT ERROR", exc)


def add_metadata(run: Any, metadata: dict[str, Any]) -> None:
    if run is None:
        return
    try:
        run.add_metadata(dict(metadata))
    except Exception as exc:
        _warn("TRACE METADATA ERROR", exc)


def add_tags(run: Any, tags: list[str]) -> None:
    if run is None:
        return
    try:
        run.add_tags([str(tag) for tag in tags if tag])
    except Exception as exc:
        _warn("TRACE TAG ERROR", exc)


def _warn(label: str, exc: BaseException) -> None:
    try:
        print(f"[{label}] {type(exc).__name__}: {redact(exc, limit=180)}", file=sys.stderr)
    except Exception:
        pass


@contextlib.contextmanager
def _nonfatal(manager: Any) -> Iterator[None]:
    """Enter and leave a LangSmith context without ever failing the block."""
    try:
        manager.__enter__()
    except Exception as exc:
        _warn("TRACE CONTEXT ERROR", exc)
        yield
        return

    try:
        yield
    except BaseException:
        try:
            manager.__exit__(*sys.exc_info())
        except Exception as exc:
            _warn("TRACE CONTEXT ERROR", exc)
        raise
    else:
        try:
            manager.__exit__(None, None, None)
        except Exception as exc:
            _warn("TRACE CONTEXT ERROR", exc)


@contextlib.contextmanager
def _open_run(
    name: str,
    *,
    run_type: str,
    inputs: Optional[dict[str, Any]],
    tags: Optional[list[str]] = None,
    metadata: Optional[dict[str, Any]] = None,
    project_name: Optional[str] = None,
    activate: bool,
) -> Iterator[Any]:
    """Enter a LangSmith run in context, yielding None if it cannot be opened."""
    # A root decides for itself whether it is sent: the ambient environment can
    # have tracing switched off, and an unposted root would leave every stage
    # span underneath it orphaned into separate traces.
    outer: Any = None
    manager: Any = None
    try:
        from langsmith import trace, tracing_context

        kwargs: dict[str, Any] = {"run_type": run_type, "inputs": dict(inputs or {})}
        if tags is not None:
            kwargs["tags"] = list(tags)
        if metadata is not None:
            kwargs["metadata"] = dict(metadata)
        if project_name:
            kwargs["project_name"] = project_name
        manager = trace(name, **kwargs)
        outer = (
            _nonfatal(tracing_context(enabled=True)) if activate else contextlib.nullcontext()
        )
    except Exception as exc:
        _warn("TRACE START ERROR", exc)
        yield None
        return

    token = _SPANS_ENABLED.set(True) if activate else None
    try:
        with outer:
            try:
                run = manager.__enter__()
            except Exception as exc:
                _warn("TRACE START ERROR", exc)
                yield None
                return
            try:
                yield run
            except BaseException:
                try:
                    manager.__exit__(*sys.exc_info())
                except Exception as exc:
                    _warn("TRACE END ERROR", exc)
                raise
            else:
                try:
                    manager.__exit__(None, None, None)
                except Exception as exc:
                    _warn("TRACE END ERROR", exc)
    finally:
        if token is not None:
            _SPANS_ENABLED.reset(token)


@contextlib.contextmanager
def root(
    name: str,
    *,
    enabled: bool,
    run_type: str = "chain",
    inputs: Optional[dict[str, Any]] = None,
    tags: Optional[list[str]] = None,
    metadata: Optional[dict[str, Any]] = None,
    project_name: Optional[str] = None,
) -> Iterator[Any]:
    """Open one trace that the whole unit of work nests under, or nothing at all.

    ``enabled`` is the caller's own decision (the browser's Trace switch for a
    chat turn, the flag captured at upload time for a background index build);
    ``can_trace()`` adds the server-side key and offline checks on top of it.
    """
    if not enabled or not can_trace():
        yield None
        return
    with _open_run(
        name,
        run_type=run_type,
        inputs=inputs,
        tags=tags,
        metadata=metadata,
        project_name=project_name,
        activate=True,
    ) as run:
        yield run


@contextlib.contextmanager
def child(
    name: str,
    *,
    run_type: str = "tool",
    inputs: Optional[dict[str, Any]] = None,
) -> Iterator[Any]:
    """Open a stage span inside the active root; a no-op when there is none."""
    if not spans_enabled():
        yield None
        return
    with _open_run(name, run_type=run_type, inputs=inputs, activate=False) as run:
        yield run


def traced(
    name: str,
    *,
    run_type: str = "tool",
    inputs: Optional[Callable[[tuple, dict], dict[str, Any]]] = None,
    outputs: Optional[Callable[[Any], dict[str, Any]]] = None,
) -> Callable:
    """Wrap a function in a named span with caller-chosen safe payloads.

    The projections receive the bound arguments and the return value, so nothing
    that is not named here — file bytes, chunk bodies, keys — ever reaches
    LangSmith. With no open root the function runs completely unwrapped.
    """
    def decorate(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            if not spans_enabled():
                return func(*args, **kwargs)
            span_inputs = dict(inputs(args, kwargs)) if inputs else {}
            with child(name, run_type=run_type, inputs=span_inputs) as run:
                result = func(*args, **kwargs)
                if outputs is not None:
                    add_outputs(run, outputs(result))
                return result
        return wrapper
    return decorate
