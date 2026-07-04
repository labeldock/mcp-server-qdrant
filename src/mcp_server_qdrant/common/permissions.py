"""
Per-collection access control for the Qdrant MCP server.

Collections are configured through the ``COLLECTION_NAME`` environment variable as a
whitespace-separated list of directives in the form ``name[:perms]``, e.g.::

    COLLECTION_NAME="travel:ro place:rw pin:rwd"

``perms`` is any combination of the letters ``r`` (read / find), ``w`` (write /
store & update) and ``d`` (delete). Any other letter is ignored, so ``ro`` reads as
"read only" (just ``r``). A directive without a ``:perms`` suffix defaults to full
access (``rwd``). A bare ``COLLECTION_NAME="mycol"`` therefore keeps the previous
single-default-collection behaviour with full permissions.
"""

from __future__ import annotations

from dataclasses import dataclass

READ = "r"
WRITE = "w"
DELETE = "d"
ALL_PERMS = "rwd"

PERM_WORD = {READ: "read", WRITE: "write", DELETE: "delete"}


@dataclass(frozen=True)
class CollectionAccess:
    """The access level granted for a single collection."""

    name: str
    permissions: frozenset[str]  # subset of {"r", "w", "d"}

    def can(self, perm: str) -> bool:
        return perm in self.permissions

    def perm_string(self) -> str:
        """Fixed-width ``rwd`` style string, e.g. ``r--`` or ``rw-``."""
        return "".join(p if p in self.permissions else "-" for p in ALL_PERMS)

    def perm_words(self) -> list[str]:
        """Human-readable list, e.g. ``["read", "write"]``."""
        return [PERM_WORD[p] for p in ALL_PERMS if p in self.permissions]


def parse_collection_directives(
    raw: str | None, *, read_only: bool = False
) -> dict[str, CollectionAccess]:
    """
    Parse the ``COLLECTION_NAME`` directive string into an ordered mapping of
    collection name -> :class:`CollectionAccess`.

    :param raw: The raw ``COLLECTION_NAME`` value (may be ``None`` / empty).
    :param read_only: Global override (``QDRANT_READ_ONLY``); when ``True`` every
        collection is stripped down to read-only.
    :return: Ordered dict keyed by collection name. Empty when ``raw`` is empty,
        which the server treats as "no whitelist" (legacy open behaviour).
    """
    result: dict[str, CollectionAccess] = {}
    if not raw:
        return result

    for token in raw.split():
        name, sep, perms_raw = token.partition(":")
        if not name:
            continue
        if sep:
            perms = frozenset(c for c in perms_raw.lower() if c in ALL_PERMS)
            # A colon was given but nothing valid parsed out -> fall back to full.
            if not perms:
                perms = frozenset(ALL_PERMS)
        else:
            perms = frozenset(ALL_PERMS)

        if read_only:
            perms = perms & frozenset(READ)

        result[name] = CollectionAccess(name=name, permissions=perms)

    return result


def union_permissions(
    collections: dict[str, CollectionAccess],
) -> frozenset[str]:
    """The set of permissions granted by at least one collection."""
    granted: set[str] = set()
    for access in collections.values():
        granted |= access.permissions
    return frozenset(granted)
