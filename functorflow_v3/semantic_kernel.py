"""Minimal semantic kernel for FunctorFlow v2.

The v2 kernel stays intentionally small. It preserves the first-class
categorical vocabulary from v1 while making room for agent-native workflow
semantics in higher-level modules.
"""

from __future__ import annotations

from dataclasses import dataclass, field


def _same_category(*objects: "ModelObject") -> "Category":
    if not objects:
        raise ValueError("Expected at least one object.")
    category = objects[0].category
    for obj in objects[1:]:
        if obj.category != category:
            raise ValueError(
                "Objects belong to different categories: "
                f"{category.name!r} vs {obj.category.name!r}."
            )
    return category


@dataclass(frozen=True)
class Category:
    """A named ambient category."""

    name: str
    metadata: dict[str, object] = field(default_factory=dict)


@dataclass(frozen=True)
class Interface:
    """A named interface exposed by a semantic object."""

    name: str
    kind: str
    metadata: dict[str, object] = field(default_factory=dict)


@dataclass(frozen=True)
class ModelObject:
    """A first-class semantic object."""

    name: str
    category: Category
    interfaces: tuple[Interface, ...] = ()
    metadata: dict[str, object] = field(default_factory=dict)


@dataclass(frozen=True)
class Morphism:
    """A typed arrow between semantic objects."""

    name: str
    source: ModelObject
    target: ModelObject
    metadata: dict[str, object] = field(default_factory=dict)

    @property
    def category(self) -> Category:
        return _same_category(self.source, self.target)


@dataclass(frozen=True)
class Functor:
    """A lightweight functor record."""

    name: str
    source_category: Category
    target_category: Category
    metadata: dict[str, object] = field(default_factory=dict)
