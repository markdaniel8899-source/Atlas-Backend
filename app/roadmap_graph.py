"""Turn whatever Llama 3.1 returns into a well-formed roadmap graph.

The model is prompted for a strict shape, but real completions drift: keys get
renamed, numbers arrive as strings, prerequisites get written as titles instead
of slugs. This module parses leniently, enforces the schema, and guarantees the
result is a DAG with a connected, acyclic flow path the UI can draw.
"""

from __future__ import annotations

import json
import math
import re
from typing import Any, Iterable

from app.schemas import (
    RoadmapConnection,
    RoadmapNode,
    RoadmapPayload,
    RoadmapPhase,
    RoadmapResource,
)

RESOURCE_TYPES = frozenset({"video", "article", "book", "course", "practice", "docs"})

PHASE_KEYS = ("phases", "stages", "modules", "units", "sections", "steps")
NODE_KEYS = ("nodes", "levels", "topics", "lessons", "items", "chapters", "entries")
PREREQ_KEYS = (
    "prerequisites",
    "prereqs",
    "depends_on",
    "requires",
    "predecessors",
    "unlocks",
)
HOUR_KEYS = ("estimated_hours", "hours", "estimated_time", "total_hours")
WEEK_KEYS = ("duration_weeks", "weeks", "duration", "estimated_weeks", "timeline")
OBJECTIVE_KEYS = ("objective", "goal", "outcome", "aim")
MILESTONE_KEYS = ("milestones", "checks", "checkpoints", "criteria")
RESOURCE_KEYS = ("resources", "links", "references", "materials")
DAY_KEYS = ("estimated_days", "days", "duration_days")
OBJECTIVE_LIST_KEYS = ("objectives", "learning_objectives", "goals")

MAX_PHASES = 16
MAX_NODES_PER_PHASE = 24
MAX_RESOURCES = 12
MAX_PREREQS = 40

_NUMBER_RE = re.compile(r"-?\d+(?:\.\d+)?")


class RoadmapShapeError(ValueError):
    """The model returned JSON that could not be read as a roadmap."""


def _first(mapping: dict[str, Any], keys: Iterable[str], default: Any = None) -> Any:
    for key in keys:
        if key in mapping and mapping[key] is not None:
            return mapping[key]
    return default


def _text(value: Any, limit: int) -> str:
    if value is None or isinstance(value, (dict, bytes)):
        return ""
    if isinstance(value, list):
        parts = [_text(item, limit) for item in value]
        value = ", ".join(part for part in parts if part)
    elif not isinstance(value, str):
        value = str(value)
    value = re.sub(r"\s+", " ", value).strip()
    return value[:limit]


def _number(value: Any, default: float, low: float, high: float) -> float:
    if isinstance(value, bool):
        return default
    if isinstance(value, (int, float)):
        number = float(value)
    elif isinstance(value, str):
        match = _NUMBER_RE.search(value.replace(",", ""))
        if not match:
            return default
        number = float(match.group())
    else:
        return default
    if math.isnan(number) or math.isinf(number):
        return default
    return max(low, min(high, number))


def _string_list(value: Any, max_items: int, max_len: int) -> list[str]:
    if isinstance(value, str):
        raw: list[Any] = re.split(r"[\n;]+", value)
    elif isinstance(value, list):
        raw = value
    else:
        return []

    out: list[str] = []
    for item in raw:
        text = _text(item, max_len).rstrip(".;,:")
        if text and text not in out:
            out.append(text)
        if len(out) >= max_items:
            break
    return out


def _url_label(url: str) -> str:
    label = re.sub(r"^https?://", "", url)
    label = re.sub(r"^www\.", "", label)
    return label.rstrip("/")[:120]


def _resources(value: Any) -> list[RoadmapResource]:
    if not isinstance(value, list):
        return []

    out: list[RoadmapResource] = []
    for item in value:
        if len(out) >= MAX_RESOURCES:
            break
        if isinstance(item, str):
            text = _text(item, 600)
            if not text:
                continue
            if text.startswith(("http://", "https://", "www.")):
                out.append(RoadmapResource(title=_url_label(text), url=text))
            else:
                out.append(RoadmapResource(title=text[:300], url=""))
            continue
        if not isinstance(item, dict):
            continue

        title = _text(_first(item, ("title", "name", "label")), 300)
        url = _text(_first(item, ("url", "link", "href")), 600)
        if not title and not url:
            continue
        if not title:
            title = url

        raw_type = _text(_first(item, ("type", "kind", "format")), 40).lower()
        kind = raw_type if raw_type in RESOURCE_TYPES else "article"
        out.append(RoadmapResource(type=kind, title=title, url=url))
    return out  # type: ignore[return-value]


def normalize_roadmap(raw: Any, *, hours_per_week: int = 10) -> RoadmapPayload:
    """Parse a raw model payload into a validated, acyclic roadmap."""
    if not isinstance(raw, dict):
        raise RoadmapShapeError("expected a JSON object")

    phases_raw = _first(raw, PHASE_KEYS)
    if not isinstance(phases_raw, list) or not phases_raw:
        # Candy-Crush style payload: a flat, sequential "levels" trail.
        levels_raw = raw.get("levels")
        if isinstance(levels_raw, list) and levels_raw:
            phases_raw = _levels_as_phases(levels_raw, hours_per_week)
    if not isinstance(phases_raw, list) or not phases_raw:
        print(
            "PARSED OBJECT (invalid structure):",
            json.dumps(raw, ensure_ascii=False, default=str)[:500],
        )
        raise RoadmapShapeError(
            f"AI returned invalid structure. Raw output: {str(raw)[:100]}"
        )

    title = _text(_first(raw, ("title", "name", "goal")), 200) or "Learning roadmap"
    summary = _text(_first(raw, ("summary", "description", "overview")), 1000)

    phases: list[RoadmapPhase] = []
    order: dict[str, int] = {}
    seen_keys: set[str] = set()
    counter = 0

    phase_items = [p for p in phases_raw if isinstance(p, dict)]
    for phase_index, phase_raw in enumerate(phase_items, start=1):
        if len(phases) >= MAX_PHASES:
            break

        nodes_raw = _first(phase_raw, NODE_KEYS)
        node_items = _as_node_items(nodes_raw)
        if not node_items:
            continue

        nodes: list[RoadmapNode] = []
        for position, node_raw in enumerate(node_items):
            if len(nodes) >= MAX_NODES_PER_PHASE:
                break
            node_title = _text(_first(node_raw, ("title", "name", "topic", "label")), 200)
            if not node_title:
                continue

            key = _make_key(
                _text(_first(node_raw, ("key", "id", "slug")), 80),
                phase_index,
                len(nodes),
                seen_keys,
            )
            seen_keys.add(key)

            raw_kind = _text(_first(node_raw, ("type", "kind")), 24).lower()
            is_boss = (
                raw_kind in {"boss", "devil"}
                or "boss" in node_title.lower()
                or "devil" in node_title.lower()
            )
            node_hours = int(
                _number(_first(node_raw, HOUR_KEYS), 0, 0, 1000)
            )
            node_days = int(_number(_first(node_raw, DAY_KEYS), 0, 0, 365))
            if node_days <= 0 and node_hours > 0 and hours_per_week > 0:
                node_days = max(
                    1, min(365, round(node_hours * 7 / hours_per_week))
                )

            node = RoadmapNode(
                key=key,
                title=node_title,
                description=_text(
                    _first(node_raw, ("description", "detail", "why", "summary")), 800
                ),
                phase_number=phase_index,
                estimated_hours=node_hours,
                estimated_days=node_days,
                prerequisites=_string_list(
                    _first(node_raw, PREREQ_KEYS), MAX_PREREQS, 80
                ),
                resources=_resources(_first(node_raw, RESOURCE_KEYS)),
                type="boss" if is_boss else "regular",
                quiz_required=is_boss
                or bool(_first(node_raw, ("quiz_required",), False)),
                objectives=_string_list(
                    _first(node_raw, OBJECTIVE_LIST_KEYS), 6, 300
                ),
            )
            order[key] = counter
            counter += 1
            nodes.append(node)

        if not nodes:
            continue

        total_hours = sum(node.estimated_hours for node in nodes)
        hours = int(
            _number(_first(phase_raw, ("total_hours", "hours")), total_hours, 0, 5000)
        )
        if hours <= 0:
            hours = total_hours

        weeks = _number(_first(phase_raw, WEEK_KEYS), 0, 0, 104)
        if weeks <= 0:
            weeks = math.ceil(hours / hours_per_week) if hours_per_week else 0

        phases.append(
            RoadmapPhase(
                number=phase_index,
                title=_text(_first(phase_raw, ("title", "name", "label")), 160)
                or f"Phase {phase_index}",
                objective=_text(_first(phase_raw, OBJECTIVE_KEYS), 600),
                duration_weeks=float(weeks),
                hours=hours,
                milestones=_string_list(_first(phase_raw, MILESTONE_KEYS), 8, 300),
                nodes=nodes,
            )
        )

    if not phases:
        raise RoadmapShapeError("no usable phases were returned")

    # Phases that produced no usable nodes are dropped above, so renumber the
    # survivors to stay contiguous (1, 2, 3 ...) before we order the graph.
    for index, phase in enumerate(phases, start=1):
        phase.number = index
        for node in phase.nodes:
            node.phase_number = index

    _resolve_prerequisites(phases)
    connections = _derive_connections(phases, order)

    return RoadmapPayload(
        title=title,
        summary=summary,
        phases=phases,
        connections=connections,
    )


def _levels_as_phases(
    levels_raw: list[Any], hours_per_week: int
) -> list[dict[str, Any]]:
    """Turn a flat sequential "levels" array into phase dicts.

    Groups close at every boss level so each phase mirrors one boss-gated
    section of the trail; the resulting dicts then flow through the normal
    node parsing in `normalize_roadmap`.
    """
    phases: list[dict[str, Any]] = []
    group: list[dict[str, Any]] = []
    group_title = ""
    group_no = 1
    hours_per_day = max(1.0, hours_per_week / 7) if hours_per_week else 2.0

    def flush() -> None:
        nonlocal group, group_title, group_no
        if not group:
            return
        phases.append(
            {
                "number": group_no,
                "title": group_title or f"Phase {group_no}",
                "objective": "",
                "duration_weeks": 0,
                "hours": 0,
                "milestones": [],
                "nodes": group,
            }
        )
        group_no += 1
        group = []
        group_title = ""

    for item in levels_raw:
        if not isinstance(item, dict):
            continue
        if len(phases) >= MAX_PHASES:
            break

        title = _text(_first(item, ("title", "name", "topic", "label")), 200)
        if not title:
            continue

        section = _text(_first(item, ("section", "chapter", "phase")), 120)
        if group and section and group_title and section != group_title:
            flush()

        raw_kind = _text(_first(item, ("type", "kind")), 24).lower()
        is_boss = (
            raw_kind in {"boss", "devil"}
            or "boss" in title.lower()
            or "devil" in title.lower()
        )

        days = int(
            _number(_first(item, DAY_KEYS), 0, 0, 365)
        )
        hours = int(_number(_first(item, HOUR_KEYS), 0, 0, 1000))
        if hours <= 0:
            effective_days = days if days > 0 else 1
            hours = min(1000, max(1, round(effective_days * hours_per_day)))
        if days <= 0:
            days = 1 if is_boss else max(1, round(hours / hours_per_day))

        group.append(
            {
                "key": _text(_first(item, ("key", "id", "slug")), 80),
                "title": title,
                "description": _text(
                    _first(item, ("description", "detail", "why", "summary")), 800
                )
                or (
                    "Complete this quiz to unlock the next phase."
                    if is_boss
                    else ""
                ),
                "estimated_hours": hours,
                "estimated_days": min(365, days),
                "prerequisites": _string_list(
                    _first(item, PREREQ_KEYS), MAX_PREREQS, 80
                ),
                "resources": _first(item, RESOURCE_KEYS),
                "type": "boss" if is_boss else "regular",
                "quiz_required": is_boss
                or bool(_first(item, ("quiz_required",), False)),
                "objectives": _string_list(
                    _first(item, OBJECTIVE_LIST_KEYS), 6, 300
                ),
            }
        )
        if not group_title and section:
            group_title = section
        if is_boss:
            flush()

    flush()
    return phases


def _as_node_items(nodes_raw: Any) -> list[dict[str, Any]]:
    if isinstance(nodes_raw, dict):
        nodes_raw = list(nodes_raw.values())
    if not isinstance(nodes_raw, list):
        return []
    items = [n for n in nodes_raw if isinstance(n, dict)]
    if items:
        return items
    return [
        {"title": item}
        for item in nodes_raw
        if isinstance(item, str) and item.strip()
    ]


def _make_key(raw: str, phase_index: int, position: int, seen: set[str]) -> str:
    key = re.sub(r"[^a-z0-9]+", "-", raw.lower()).strip("-")[:70]
    if not key:
        key = f"p{phase_index}-{position + 1}"
    candidate = key
    suffix = 1
    while candidate in seen:
        suffix += 1
        candidate = f"{key}-{suffix}"[:70]
    return candidate


def _resolve_prerequisites(phases: list[RoadmapPhase]) -> None:
    """Map prerequisite strings (often titles) onto real node keys."""
    alias: dict[str, str] = {}
    for phase in phases:
        for node in phase.nodes:
            alias.setdefault(node.key.lower(), node.key)
            alias.setdefault(node.title.lower(), node.key)

    for phase in phases:
        for node in phase.nodes:
            resolved: list[str] = []
            for raw in node.prerequisites:
                lookup = raw.strip().lower().rstrip(".;,:")
                target = alias.get(lookup) or alias.get(
                    re.sub(r"[^a-z0-9]+", "-", lookup).strip("-")
                )
                if target and target != node.key and target not in resolved:
                    resolved.append(target)
            node.prerequisites = resolved[:MAX_PREREQS]


def _derive_connections(
    phases: list[RoadmapPhase], order: dict[str, int]
) -> list[RoadmapConnection]:
    """Build an acyclic edge list: the sequential flow path plus dependencies."""
    edges: list[tuple[str, str]] = []
    seen: set[tuple[str, str]] = set()

    def add(source: str, target: str) -> None:
        if source == target:
            return
        if order.get(source, -1) >= order.get(target, 1 << 30):
            return
        if (source, target) in seen:
            return
        seen.add((source, target))
        edges.append((source, target))

    previous_tail: str | None = None
    for phase in phases:
        chain = [node.key for node in phase.nodes]
        if previous_tail and chain:
            add(previous_tail, chain[0])
        for source, target in zip(chain, chain[1:]):
            add(source, target)
        if chain:
            previous_tail = chain[-1]

    for phase in phases:
        for node in phase.nodes:
            for prereq in node.prerequisites:
                add(prereq, node.key)

    return [RoadmapConnection(source=s, target=t) for s, t in edges]
