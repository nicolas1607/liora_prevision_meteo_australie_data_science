"""Load the EXP-007 bundles even when XGBoost refuses to unpickle its own booster.

On xgboost 3.4.1 - the version this app pins - every bundle in ``models/`` fails
with ``XGBoostError('input stream corrupted')``.  ``Booster.__setstate__`` hands
the pickled blob straight to ``XGBoosterUnserializeFromBuffer``, which rejects it
even though the blob is well-formed UBJSON written by the same library.

The blob is an object ``{"Config": ..., "Model": ...}`` and the ``"Model"`` value
alone is byte-for-byte what ``save_model(*.ubj)`` produces.  So the recovery is:

1. unpickle with every ``xgboost`` class replaced by an inert stub, so nothing
   reaches the broken ``__setstate__`` (joblib's own unpickler has to be
   subclassed here - a plain ``pickle.Unpickler`` cannot read joblib's format);
2. pull the raw bytes out of the stubbed ``Booster``'s ``handle`` state;
3. slice the ``"Model"`` value out and feed just that to ``Booster.load_model``.

Only the training-time config is dropped, which inference does not need - the
recovered booster still carries its 800 rounds, its 44 features and their names.

Keys in this blob are length-prefixed with UBJSON's ``L`` (int64) marker; the
smaller markers are accepted too in case a different writer produced the file.
``load_bundle`` tries the ordinary path first, so on a build without the bug
nothing unusual happens.
"""

from __future__ import annotations

import struct
import warnings
from pathlib import Path
from typing import Any

import joblib
import numpy as np

_CORRUPT_HINTS = ("input stream corrupted", "corrupt", "unserialize")


def _key_markers(name: str) -> list[bytes]:
    """Every UBJSON length-prefix encoding of an object key."""
    raw = name.encode()
    return [
        b"L" + struct.pack(">q", len(raw)) + raw,   # int64  (what these files use)
        b"I" + struct.pack(">i", len(raw)) + raw,   # int32
        b"l" + struct.pack(">h", len(raw)) + raw,   # int16
        b"i" + bytes([len(raw)]) + raw,             # int8
        b"U" + bytes([len(raw)]) + raw,             # uint8
    ]


class _Stub:
    """Stands in for an XGBoost class so unpickling cannot reach its __setstate__."""

    def __init__(self, *args, **kwargs):
        self.__dict__["_state"] = {}

    def __setstate__(self, state):
        self.__dict__["_state"] = state


def _stubbed_load(path: Path) -> Any:
    """Unpickle a joblib bundle with the xgboost classes neutralised."""
    import joblib.numpy_pickle as numpy_pickle

    class _StubbingUnpickler(numpy_pickle.NumpyUnpickler):
        def find_class(self, module: str, name: str):
            if module.startswith("xgboost"):
                return type(f"Stub_{name}", (_Stub,), {"_xgb_name": name})
            return super().find_class(module, name)

    with open(path, "rb") as handle:
        return _StubbingUnpickler(str(path), handle, ensure_native_byte_order=True).load()


def _find_handle(obj, _seen: set | None = None) -> bytes | None:
    """Depth-first search for the booster's raw ``handle`` byte buffer."""
    _seen = set() if _seen is None else _seen
    if id(obj) in _seen:
        return None
    _seen.add(id(obj))

    if isinstance(obj, (bytes, bytearray)):
        return bytes(obj) if len(obj) > 1024 else None
    if isinstance(obj, np.ndarray) and obj.dtype == np.uint8:
        return obj.tobytes()

    children: list = []
    if isinstance(obj, dict):
        children = list(obj.values())
    elif isinstance(obj, (list, tuple, set)):
        children = list(obj)
    elif hasattr(obj, "__dict__"):
        children = list(vars(obj).values())

    for child in children:
        found = _find_handle(child, _seen)
        if found is not None:
            return found
    return None


def _booster_from_blob(blob: bytes):
    """Slice the ``"Model"`` value out of the UBJSON object and load only that."""
    import xgboost as xgb

    for marker in _key_markers("Model"):
        start = blob.find(marker)
        if start < 0:
            continue
        payload = bytes(blob[start + len(marker):])
        # "Model" is the final key, so its value runs to the outer object's '}'
        if payload.endswith(b"}"):
            payload = payload[:-1]
        booster = xgb.Booster()
        # load_model needs a writable buffer; a read-only view is rejected
        booster.load_model(bytearray(payload))
        return booster
    raise ValueError("no UBJSON 'Model' key in the pickled booster")


def load_bundle(path: str | Path) -> dict:
    """Return the bundle dict, repairing the booster if the normal load fails."""
    path = Path(path)
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            return joblib.load(path)
    except Exception as original:
        if not any(hint in str(original).lower() for hint in _CORRUPT_HINTS):
            raise

        bundle = _stubbed_load(path)
        if not isinstance(bundle, dict):
            raise original

        blob = _find_handle(bundle.get("model"))
        if blob is None:
            raise original

        bundle["model"] = _booster_from_blob(blob)
        bundle["_recovered"] = True
        return bundle
