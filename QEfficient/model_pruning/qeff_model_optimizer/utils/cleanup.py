"""Shared cleanup helpers for reversible model transforms."""

from __future__ import annotations


def register_model_cleanup(target, callback) -> None:
    """Register a reversible cleanup callback on the root model."""

    callbacks = getattr(target, "_nas_cleanup_callbacks", None)
    if callbacks is None:
        callbacks = []
        setattr(target, "_nas_cleanup_callbacks", callbacks)
    callbacks.append(callback)


def run_model_cleanup(target) -> None:
    """Run and clear all registered cleanup callbacks on the root model."""

    callbacks = getattr(target, "_nas_cleanup_callbacks", None)
    if not callbacks:
        return

    for callback in reversed(callbacks):
        callback(target)
    delattr(target, "_nas_cleanup_callbacks")
