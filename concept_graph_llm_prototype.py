from __future__ import annotations

from dataclasses import dataclass, field
from collections import defaultdict
from typing import Dict, List, Optional, Iterable, Tuple
from pathlib import Path
import argparse
import re
from korean_event_input import KoreanEventInput, train_file
from event_reasoning import ReasoningSession
from korean_dialogue import KoreanDialogue


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
    confidence: float = 1.0
    novelty: float = 0.0
    flags: int = 0
    edges: List[Edge] = field(default_factory=list)
    domains: frozenset[str] = field(default_factory=frozenset)


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
        domains: Iterable[str] = (),
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
                domains=frozenset(domains),
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
            raise KeyError(f"Unknown concept: {concept!r}")

        return node_id

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
        self.relation_gain: Dict[str, float] = {}

    def set_relation_gain(self, relation: str, gain: float) -> None:
        self.relation_gain[relation] = gain

    def activate(
        self,
        seeds: Dict[str | int, float],
        *,
        disabled: Iterable[str | int] = (),
        active_domains: Optional[Iterable[str]] = None,
    ) -> Dict[int, WorkingNode]:
        blocked = {self.graph._id(concept) for concept in disabled}
        selected = None if active_domains is None else frozenset(active_domains)

        def allowed(node_id):
            domains = self.graph.nodes[node_id].domains
            return (node_id not in blocked and
                    (selected is None or not domains or bool(domains & selected)))

        activations: Dict[int, float] = {}
        working: Dict[int, WorkingNode] = {}
        frontier: Dict[int, float] = {}

        for concept, activation in seeds.items():
            node_id = self.graph._id(concept)
            if not allowed(node_id):
                continue
            activation = max(0.0, min(1.0, activation))
            if activation <= 0 or activation < self.threshold:
                continue
            activation = max(activations.get(node_id, 0.0), activation)
            activations[node_id] = activation
            frontier[node_id] = activation
            working[node_id] = WorkingNode(
                concept_id=node_id,
                name=self.graph.nodes[node_id].name,
                activation=activation,
                depth=0,
            )

        seed_ids = set(frontier)
        parent_strength: Dict[int, float] = {}

        for depth in range(self.max_depth):
            candidates: List[Tuple[float, int, Edge, float]] = []

            for src_id, src_act in frontier.items():
                for edge in self.graph.nodes[src_id].edges:
                    if not allowed(edge.target):
                        continue
                    contribution = (
                        src_act
                        * edge.strength
                        * edge.confidence
                        * self.graph.nodes[edge.target].confidence
                        * self.relation_gain.get(edge.relation, 1.0)
                        * self.decay
                        * (-1.0 if edge.inhibitory else 1.0)
                        * max(0.0, 1.0 - edge.cost)
                    )

                    if abs(contribution) >= self.threshold:
                        candidates.append(
                            (abs(contribution), src_id, edge, contribution)
                        )

            candidates.sort(key=lambda item: item[0], reverse=True)
            candidates = candidates[: self.top_k]
            if not candidates:
                break

            next_signals: Dict[int, float] = defaultdict(float)

            for magnitude, src_id, edge, contribution in candidates:
                next_signals[edge.target] += contribution

                if (
                    edge.target not in seed_ids
                    and magnitude > parent_strength.get(edge.target, -1.0)
                ):
                    parent_strength[edge.target] = magnitude
                    target = self.graph.nodes[edge.target]
                    working[edge.target] = WorkingNode(
                        concept_id=edge.target,
                        name=target.name,
                        activation=0.0,
                        depth=depth + 1,
                        parent=src_id,
                        via_relation=edge.relation,
                    )

            frontier = {}

            for node_id, signal in next_signals.items():
                signal = max(-1.0, min(1.0, signal))
                activation = max(
                    -1.0,
                    min(1.0, activations.get(node_id, 0.0) + signal),
                )
                activations[node_id] = activation

                item = working.get(node_id)
                if item is not None:
                    item.activation = activation
                elif abs(activation) >= self.threshold:
                    working[node_id] = WorkingNode(
                        concept_id=node_id,
                        name=self.graph.nodes[node_id].name,
                        activation=activation,
                        depth=depth + 1,
                    )

                # Inhibition affects the target but does not spread further.
                if signal >= self.threshold:
                    frontier[node_id] = signal

            if not frontier:
                break

        working = {
            node_id: item
            for node_id, item in working.items()
            if abs(item.activation) >= self.threshold or item.depth == 0
        }

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

    g.add_concept("apple", aliases=["사과"])
    g.add_concept("tree", aliases=["나무"])
    g.add_concept("fall", aliases=["떨어지다", "떨어지는", "낙하"], domains=["물리"])
    g.add_concept("object", aliases=["물체"], domains=["물리"])
    g.add_concept("mass", aliases=["질량"], domains=["물리"])
    g.add_concept("gravity", aliases=["중력"], domains=["물리"])
    g.add_concept("acceleration", aliases=["가속도"], domains=["물리"])
    g.add_concept("height", aliases=["높이"], domains=["물리"])
    g.add_concept("motion", aliases=["운동"], domains=["물리"])
    g.add_concept("ground", aliases=["땅", "지면"], domains=["물리"])

    g.connect("apple", "object", "is_a", strength=0.95)
    g.connect("object", "mass", "has_property", strength=0.88)
    g.connect("mass", "gravity", "affected_by", strength=0.95)

    g.connect("gravity", "acceleration", "causes", strength=0.98)
    g.connect("acceleration", "fall", "causes", strength=0.92)

    g.connect("tree", "height", "has_property", strength=0.90)
    g.connect("height", "fall", "enables", strength=0.72)

    g.connect("fall", "motion", "is_a", strength=0.92)
    g.connect("fall", "ground", "toward", strength=0.70)

    return g


def build_reasoning_graph(perception):
    graph = build_demo_graph()
    graph.add_concept("대화", domains=["대화"])
    # Build the index once; each request traverses only the selected subgraph.
    for predicate in perception.domains:
        domain = perception.domain_for(predicate)
        if domain is None:
            continue
        graph.add_concept(domain, domains=[domain])
        if graph.resolve(predicate) is None:
            graph.add_concept(predicate, domains=[domain])
        graph.connect(domain, predicate, "관련동작")
    return graph


# ============================================================
# Example
# ============================================================

def main() -> None:
    parser = argparse.ArgumentParser(description="한국어 사건과 희소 개념 활성화")
    parser.add_argument("--model", help="학습한 한국어 입력 모델 JSON")
    parser.add_argument("--dialogue-model", help="학습한 한국어 대화 모델 JSON")
    parser.add_argument("--disable", action="append", default=[], help="비활성화할 개념 이름 (반복 가능)")
    parser.add_argument("--disable-domain", action="append", default=[], help="비활성화할 분야 (소유, 위치, 물리 등)")
    parser.add_argument("--debug", action="store_true", help="활성화된 그래프 노드 출력")
    args = parser.parse_args()
    perception = (KoreanEventInput.load(args.model) if args.model
                  else train_file(Path(__file__).with_name("korean_events.jsonl")))
    dialogue = (KoreanDialogue.load(args.dialogue_model) if args.dialogue_model
                else KoreanDialogue.train_file(Path(__file__).with_name("korean_dialogues.jsonl")))
    graph = build_reasoning_graph(perception)

    router = ActivationRouter(
        graph,
        decay=0.82,
        threshold=0.10,
        top_k=32,
        max_depth=5,
    )

    # Relations can be selectively emphasized.
    router.set_relation_gain("causes", 1.15)
    router.set_relation_gain("affected_by", 1.10)

    decoder = DebugDecoder(graph)
    try:
        session = ReasoningSession(perception, router, disabled=args.disable,
                                   disabled_domains=args.disable_domain, dialogue=dialogue)
    except (KeyError, ValueError) as error:
        parser.error(str(error))
    print("문장이나 질문을 입력하세요. /초기화: 기억 삭제, /종료 또는 빈 입력: 종료")
    while True:
        try:
            text = input("입력: ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not text or text == "/종료":
            break
        if text == "/초기화":
            session.reset()
            print("작업 기억을 초기화했습니다.")
            continue
        for sentence in re.findall(r"[^.!?。！？\n]+[.!?。！？]*", text):
            if not sentence.strip():
                continue
            result = session.process(sentence)
            if args.debug:
                print("활성 분야: " + (", ".join(result.active_domains) or "없음"))
                if result.intent:
                    print("대화 의도: " + result.intent)
            if args.debug and result.event:
                event = result.event
                print(f"사건: {event.predicate} / {event.tense} / 부정: {'예' if event.negated else '아니오'}")
                print("  " + ", ".join(f"{role}={value}" for role, value in event.roles.items()))
            print(result.message)
            for evidence in result.evidence:
                print(f"  근거: {evidence}")
            if args.debug:
                print(decoder.render(result.working))


if __name__ == "__main__":
    main()
