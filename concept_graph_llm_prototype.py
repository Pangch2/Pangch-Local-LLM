from __future__ import annotations

from dataclasses import dataclass, field
from collections import defaultdict
from typing import Dict, List, Optional, Iterable, Tuple
import heapq


# ============================================================
# Core data structures
# ============================================================

@dataclass(slots=True)
class Edge:
    target: int
    relation: str
    strength: float = 1.0
    confidence: float = 1.0
    inhibitory: bool = False
    cost: float = 0.0


@dataclass(slots=True)
class ConceptNode:
    id: int
    name: str
    activation: float = 0.0
    confidence: float = 1.0
    novelty: float = 0.0
    flags: int = 0
    edges: List[Edge] = field(default_factory=list)


@dataclass(slots=True)
class WorkingNode:
    concept_id: int
    name: str
    activation: float
    depth: int
    parent: Optional[int] = None
    via_relation: Optional[str] = None


# ============================================================
# Long-term concept graph
# ============================================================

class ConceptGraph:
    def __init__(self) -> None:
        self.nodes: Dict[int, ConceptNode] = {}
        self.name_to_id: Dict[str, int] = {}
        self.alias_to_id: Dict[str, int] = {}
        self._next_id = 0

    def add_concept(
        self,
        name: str,
        *,
        aliases: Iterable[str] = (),
        confidence: float = 1.0,
    ) -> int:
        key = name.strip().lower()

        if key in self.name_to_id:
            node_id = self.name_to_id[key]
        else:
            node_id = self._next_id
            self._next_id += 1

            self.nodes[node_id] = ConceptNode(
                id=node_id,
                name=name,
                confidence=confidence,
            )
            self.name_to_id[key] = node_id

        self.alias_to_id[key] = node_id

        for alias in aliases:
            self.alias_to_id[alias.strip().lower()] = node_id

        return node_id

    def resolve(self, text: str) -> Optional[int]:
        return self.alias_to_id.get(text.strip().lower())

    def connect(
        self,
        source: str | int,
        target: str | int,
        relation: str,
        *,
        strength: float = 1.0,
        confidence: float = 1.0,
        inhibitory: bool = False,
        cost: float = 0.0,
        bidirectional: bool = False,
    ) -> None:
        src_id = self._id(source)
        dst_id = self._id(target)

        self.nodes[src_id].edges.append(
            Edge(
                target=dst_id,
                relation=relation,
                strength=strength,
                confidence=confidence,
                inhibitory=inhibitory,
                cost=cost,
            )
        )

        if bidirectional:
            self.nodes[dst_id].edges.append(
                Edge(
                    target=src_id,
                    relation=relation,
                    strength=strength,
                    confidence=confidence,
                    inhibitory=inhibitory,
                    cost=cost,
                )
            )

    def _id(self, concept: str | int) -> int:
        if isinstance(concept, int):
            return concept

        node_id = self.resolve(concept)
        if node_id is None:
            raise KeyError(f"알 수 없는 개념: {concept!r}")

        return node_id

    def reset_activation(self) -> None:
        for node in self.nodes.values():
            node.activation = 0.0


# ============================================================
# Sparse activation router
# ============================================================

class ActivationRouter:
    """
    Only active nodes are traversed.

    This is intentionally NOT a dense matrix operation.
    Complexity is roughly proportional to the number of visited edges.
    """

    def __init__(
        self,
        graph: ConceptGraph,
        *,
        decay: float = 0.78,
        threshold: float = 0.08,
        top_k: int = 64,
        max_depth: int = 4,
    ) -> None:
        self.graph = graph
        self.decay = decay
        self.threshold = threshold
        self.top_k = top_k
        self.max_depth = max_depth

        # Relation-specific routing bias.
        self.relation_gain: Dict[str, float] = defaultdict(lambda: 1.0)

    def set_relation_gain(self, relation: str, gain: float) -> None:
        self.relation_gain[relation] = gain

    def activate(
        self,
        seeds: Dict[str | int, float],
    ) -> Dict[int, WorkingNode]:
        self.graph.reset_activation()

        working: Dict[int, WorkingNode] = {}
        frontier: List[Tuple[float, int, int]] = []

        for concept, activation in seeds.items():
            node_id = self.graph._id(concept)
            activation = max(0.0, min(1.0, activation))

            self.graph.nodes[node_id].activation = activation
            working[node_id] = WorkingNode(
                concept_id=node_id,
                name=self.graph.nodes[node_id].name,
                activation=activation,
                depth=0,
            )

            # max heap via negative activation
            heapq.heappush(frontier, (-activation, node_id, 0))

        best_seen: Dict[int, float] = {
            node_id: item.activation
            for node_id, item in working.items()
        }

        while frontier:
            neg_act, src_id, depth = heapq.heappop(frontier)
            src_act = -neg_act

            if depth >= self.max_depth:
                continue

            # Ignore stale heap entries.
            if src_act + 1e-12 < best_seen.get(src_id, 0.0):
                continue

            src = self.graph.nodes[src_id]

            candidates: List[Tuple[float, Edge]] = []

            for edge in src.edges:
                sign = -1.0 if edge.inhibitory else 1.0

                message = (
                    src_act
                    * edge.strength
                    * edge.confidence
                    * self.relation_gain[edge.relation]
                    * self.decay
                    * sign
                )

                # Cost lowers routing priority.
                if message > 0.0:
                    message *= max(0.0, 1.0 - edge.cost)

                if abs(message) >= self.threshold:
                    candidates.append((abs(message), edge))

            # Local top-k avoids exploding fan-out.
            candidates.sort(key=lambda x: x[0], reverse=True)
            candidates = candidates[: self.top_k]

            for magnitude, edge in candidates:
                target = self.graph.nodes[edge.target]

                sign = -1.0 if edge.inhibitory else 1.0
                contribution = (
                    src_act
                    * edge.strength
                    * edge.confidence
                    * self.relation_gain[edge.relation]
                    * self.decay
                    * sign
                    * max(0.0, 1.0 - edge.cost)
                )

                new_activation = max(
                    -1.0,
                    min(1.0, target.activation + contribution),
                )

                target.activation = new_activation

                if abs(new_activation) < self.threshold:
                    continue

                prev = best_seen.get(edge.target, 0.0)
                if abs(new_activation) <= abs(prev):
                    continue

                best_seen[edge.target] = new_activation

                working[edge.target] = WorkingNode(
                    concept_id=edge.target,
                    name=target.name,
                    activation=new_activation,
                    depth=depth + 1,
                    parent=src_id,
                    via_relation=edge.relation,
                )

                if new_activation > 0:
                    heapq.heappush(
                        frontier,
                        (-new_activation, edge.target, depth + 1),
                    )

        # Global top-k over the final working set.
        ordered = sorted(
            working.values(),
            key=lambda item: abs(item.activation),
            reverse=True,
        )[: self.top_k]

        return {item.concept_id: item for item in ordered}


# ============================================================
# Very small reasoning engine
# ============================================================

class ReasoningEngine:
    """
    Deliberately simple rule engine.

    Later this can be replaced with:
    - learned operators
    - graph rewrite rules
    - planning/search
    - probabilistic inference
    """

    def __init__(self, graph: ConceptGraph) -> None:
        self.graph = graph

    def strongest(
        self,
        working: Dict[int, WorkingNode],
        n: int = 10,
    ) -> List[WorkingNode]:
        return sorted(
            working.values(),
            key=lambda x: x.activation,
            reverse=True,
        )[:n]

    def explain_path(
        self,
        working: Dict[int, WorkingNode],
        concept: str | int,
    ) -> List[Tuple[str, Optional[str]]]:
        node_id = self.graph._id(concept)

        if node_id not in working:
            return []

        result: List[Tuple[str, Optional[str]]] = []
        current = node_id
        visited = set()

        while current in working and current not in visited:
            visited.add(current)
            item = working[current]
            result.append((item.name, item.via_relation))

            if item.parent is None:
                break

            current = item.parent

        result.reverse()
        return result

    def find_relation(
        self,
        working: Dict[int, WorkingNode],
        source: str | int,
        relation: str,
    ) -> List[Tuple[str, float]]:
        src_id = self.graph._id(source)

        if src_id not in working:
            return []

        results: List[Tuple[str, float]] = []

        for edge in self.graph.nodes[src_id].edges:
            if edge.relation != relation:
                continue

            if edge.target not in working:
                continue

            score = (
                working[src_id].activation
                * edge.strength
                * edge.confidence
                * working[edge.target].activation
            )

            results.append(
                (self.graph.nodes[edge.target].name, score)
            )

        results.sort(key=lambda x: x[1], reverse=True)
        return results


# ============================================================
# Extremely simple "perception" layer
# ============================================================

class KeywordPerception:
    """
    Prototype only:
    maps words/aliases in text to existing concepts.

    This is NOT the intended final architecture.
    A learned semantic parser should replace it later.
    """

    def __init__(self, graph: ConceptGraph) -> None:
        self.graph = graph

    def encode(self, text: str) -> Dict[int, float]:
        lowered = text.lower()
        found: Dict[int, float] = {}

        # Longer aliases first to reduce accidental overlap.
        aliases = sorted(
            self.graph.alias_to_id.items(),
            key=lambda x: len(x[0]),
            reverse=True,
        )

        for alias, node_id in aliases:
            if alias in lowered:
                found[node_id] = max(found.get(node_id, 0.0), 1.0)

        return found


# ============================================================
# Small text decoder for debugging
# ============================================================

class DebugDecoder:
    def __init__(self, graph: ConceptGraph) -> None:
        self.graph = graph

    def render(
        self,
        working: Dict[int, WorkingNode],
        *,
        limit: int = 15,
    ) -> str:
        items = sorted(
            working.values(),
            key=lambda x: x.activation,
            reverse=True,
        )[:limit]

        lines = ["활성화된 개념:"]

        for item in items:
            parent_name = (
                self.graph.nodes[item.parent].name
                if item.parent is not None
                else "-"
            )

            lines.append(
                f"  {item.name:20s} "
                f"활성도={item.activation: .3f} "
                f"깊이={item.depth} "
                f"관계={item.via_relation or '-':14s} "
                f"출발={parent_name}"
            )

        return "\n".join(lines)


# ============================================================
# Demo knowledge base
# ============================================================

def build_demo_graph() -> ConceptGraph:
    g = ConceptGraph()

    g.add_concept("사과", aliases=["apple"])
    g.add_concept("나무", aliases=["tree"])
    g.add_concept("낙하", aliases=["fall", "떨어지다", "떨어지는"])
    g.add_concept("물체", aliases=["object"])
    g.add_concept("질량", aliases=["mass"])
    g.add_concept("중력", aliases=["gravity"])
    g.add_concept("가속도", aliases=["acceleration"])
    g.add_concept("높이", aliases=["height"])
    g.add_concept("운동", aliases=["motion"])
    g.add_concept("지면", aliases=["ground", "땅"])

    g.connect("사과", "물체", "종류", strength=0.95)
    g.connect("물체", "질량", "속성", strength=0.88)
    g.connect("질량", "중력", "영향받음", strength=0.95)

    g.connect("중력", "가속도", "원인", strength=0.98)
    g.connect("가속도", "낙하", "원인", strength=0.92)

    g.connect("나무", "높이", "속성", strength=0.90)
    g.connect("높이", "낙하", "가능하게 함", strength=0.72)

    g.connect("낙하", "운동", "종류", strength=0.92)
    g.connect("낙하", "지면", "향함", strength=0.70)

    return g


# ============================================================
# Example
# ============================================================

def main() -> None:
    graph = build_demo_graph()

    perception = KeywordPerception(graph)

    router = ActivationRouter(
        graph,
        decay=0.82,
        threshold=0.10,
        top_k=32,
        max_depth=5,
    )

    # Relations can be selectively emphasized.
    router.set_relation_gain("원인", 1.15)
    router.set_relation_gain("영향받음", 1.10)

    reasoner = ReasoningEngine(graph)
    decoder = DebugDecoder(graph)

    query = input("질문: ").strip()
    if not query:
        return

    seeds = perception.encode(query)

    print("입력:")
    print(query)
    print()

    print("시드 개념:")
    for node_id, activation in seeds.items():
        print(
            f"  {graph.nodes[node_id].name}: "
            f"{activation:.2f}"
        )

    print()

    working = router.activate(seeds)

    print(decoder.render(working))
    print()

    print("중력까지의 추론 경로:")
    path = reasoner.explain_path(working, "중력")

    if not path:
        print("  중력이 활성화되지 않았습니다")
    else:
        for i, (name, relation) in enumerate(path):
            if i == 0:
                print(f"  {name}")
            else:
                print(f"    --{relation}--> {name}")

    print()

    print("낙하까지의 추론 경로:")
    path = reasoner.explain_path(working, "낙하")

    if not path:
        print("  낙하가 활성화되지 않았습니다")
    else:
        for i, (name, relation) in enumerate(path):
            if i == 0:
                print(f"  {name}")
            else:
                print(f"    --{relation}--> {name}")


if __name__ == "__main__":
    main()
