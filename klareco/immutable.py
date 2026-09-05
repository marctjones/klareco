"""Owned, recursively read-only JSON snapshots with explicit mutable copying."""

from copy import deepcopy
from typing import Any, NoReturn


def _readonly(*args: Any, **kwargs: Any) -> NoReturn:
    raise TypeError("Snapshot is immutable; deepcopy it before editing")


class FrozenDict(dict):
    __setitem__ = __delitem__ = __ior__ = _readonly
    clear = pop = popitem = setdefault = update = _readonly

    def __deepcopy__(self, memo):
        result = {}
        memo[id(self)] = result
        result.update((deepcopy(k, memo), deepcopy(v, memo)) for k, v in self.items())
        return result


class FrozenList(list):
    __setitem__ = __delitem__ = __iadd__ = __imul__ = _readonly
    append = clear = extend = insert = pop = remove = reverse = sort = _readonly

    def __deepcopy__(self, memo):
        result = []
        memo[id(self)] = result
        result.extend(deepcopy(v, memo) for v in self)
        return result


def freeze(value: Any, memo: dict[int, Any] | None = None) -> Any:
    if isinstance(value, (FrozenDict, FrozenList)):
        return value
    if memo is None:
        memo = {}
    if id(value) in memo:
        return memo[id(value)]
    if isinstance(value, dict):
        result = FrozenDict()
        memo[id(value)] = result
        for key, item in value.items():
            dict.__setitem__(result, key, freeze(item, memo))
        return result
    if isinstance(value, list):
        result = FrozenList()
        memo[id(value)] = result
        for item in value:
            list.append(result, freeze(item, memo))
        return result
    if isinstance(value, tuple):
        result = tuple(freeze(item, memo) for item in value)
        return value if all(a is b for a, b in zip(value, result)) else result
    return value
