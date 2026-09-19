"""Domain-gated event memory and small, explicit state transitions."""
from dataclasses import dataclass, field
import re

from korean_event_input import KoreanEvent
from korean_dialogue import introduced_name


# domain, entity role, new value role, previous owner role (if transferring).
# These operators are explicit semantics; the input/domain mapping is learned.
OPERATIONS = {
    "가지다": ("소유", "대상", "행위자", None),
    "주다": ("소유", "대상", "받는사람", "행위자"),
    "받다": ("소유", "대상", "행위자", "주는사람"),
    "가다": ("위치", "행위자", "장소", None),
    "오다": ("위치", "행위자", "장소", None),
    "이동하다": ("위치", "행위자", "장소", None),
    "있다": ("위치", "행위자", "장소", None),
}


@dataclass(frozen=True)
class Fact:
    value: str
    evidence: tuple[int, ...]


@dataclass
class TurnResult:
    status: str
    message: str
    active_domains: tuple[str, ...] = ()
    working: dict = field(default_factory=dict)
    event: KoreanEvent | None = None
    answer: str | None = None
    evidence: tuple[str, ...] = ()
    intent: str | None = None


def parse_question(text):
    # ponytail: explicit current-state questions; learn query structures when
    # historical, comparative, and multi-clause questions are introduced.
    text = text.strip().rstrip("?!.。！？").strip()
    if text.startswith("지금 "):
        text = text[3:]
    noun = r"([가-힣A-Za-z0-9_]+?)"
    endings = r"(?:누구|누구인가|누구야|누구입니까)"
    patterns = (
        ("소유", noun + r"(?:을|를) 가진 사람은 " + endings),
        ("소유", noun + r"(?:은|는|을|를) 누가 가지고 (?:있나|있니|있어|있는가|있습니까)"),
        ("소유", r"누가 " + noun + r"(?:을|를) 가지고 (?:있나|있니|있어|있는가|있습니까)"),
        ("소유", noun + r"의 소유자는 " + endings),
        ("위치", noun + r"(?:은|는|이|가) 어디에 (?:있나|있니|있어|있는가|있습니까)"),
        ("위치", noun + r"의 위치는 (?:어디|어디인가|어디야|어디입니까)"),
    )
    for domain, pattern in patterns:
        if match := re.fullmatch(pattern, text):
            return domain, match[1]
    return None


class ReasoningSession:
    def __init__(self, perception, router, *, disabled=(), disabled_domains=(), dialogue=None):
        self.perception = perception
        self.router = router
        self.graph = router.graph
        self.dialogue = dialogue
        self.disabled = {self.graph._id(name) for name in disabled}
        self.disabled_domains = set(disabled_domains)
        known_domains = {domain for counts in perception.domains.values() for domain in counts}
        unknown = self.disabled_domains - known_domains - {"소유", "위치", "대화"}
        if unknown:
            raise ValueError("등록되지 않은 분야: " + ", ".join(sorted(unknown)))
        self.reset()

    def reset(self):
        self.facts: dict[str, dict[str, Fact]] = {"소유": {}, "위치": {}}
        self.events: list[tuple[str, KoreanEvent]] = []
        self.user_name = None
        self.focus = {"소유": None, "위치": None}
        self.last_answer = None

    def _activate(self, domain, names):
        if domain in self.disabled_domains:
            return {}
        seeds = {node_id: 1.0 for name in [domain, *names]
                 if (node_id := self.graph.resolve(name)) is not None}
        # Blocking the relevant entity, operator, or domain also blocks state use.
        if self.disabled.intersection(seeds):
            return {}
        return self.router.activate(seeds, disabled=self.disabled, active_domains=[domain])

    def _answer(self, domain, entity):
        self.last_answer = None
        working = self._activate(domain, [entity])
        if not working:
            return TurnResult("비활성", "필요한 분야 또는 개념이 비활성 상태입니다.")
        self.focus[domain] = entity
        # Direct lookup in one domain. Other domain memories are not scanned.
        fact = self.facts[domain].get(entity)
        if fact is None:
            return TurnResult("모름", f"{entity}: 확인된 정보가 없습니다.", (domain,), working)
        label = "소유자" if domain == "소유" else "위치"
        result = TurnResult("답변", f"{entity}의 {label}: {fact.value}", (domain,), working,
                            answer=fact.value,
                            evidence=tuple(self.events[index][0] for index in fact.evidence))
        if domain == "소유":
            self.focus["위치"] = fact.value
        self.last_answer = (domain, entity, result)
        return result

    def _dialogue_turn(self, text):
        act = self.dialogue.classify(text) if self.dialogue else None
        if act is None or act.intent == "기타":
            return None
        followup = {"후속소유": "소유", "후속위치": "위치"}
        if act.intent in followup:
            domain = followup[act.intent]
            working = self._activate(domain, [])
            if not working:
                return TurnResult("비활성", "필요한 분야 또는 개념이 비활성 상태입니다.")
            if self.focus[domain] is None:
                return TurnResult("확인필요", act.response, (domain,), working, intent=act.intent)
            result = self._answer(domain, self.focus[domain])
            result.intent = act.intent
            return result
        if act.intent == "근거질문" and self.last_answer:
            domain, entity, answer = self.last_answer
            working = self._activate(domain, [entity])
            if not working:
                return TurnResult("비활성", "답의 근거가 속한 분야 또는 개념이 비활성 상태입니다.")
            return TurnResult("설명", "이전에 입력된 사건을 적용해 판단했어요.", (domain,), working,
                              evidence=answer.evidence, intent=act.intent)
        working = self._activate("대화", [])
        if not working:
            return TurnResult("비활성", "대화 분야가 비활성 상태입니다.")
        response = act.response
        if act.intent == "이름소개":
            name = introduced_name(text)
            if name is None:
                return TurnResult("확인필요", "'제 이름은 민수입니다'처럼 이름을 알려 주세요.",
                                  ("대화",), working, intent=act.intent)
            self.user_name = name
            response = response.replace("{이름}", name)
        elif act.intent == "이름질문" and self.user_name:
            response = f"이 대화에서 알려 주신 이름은 {self.user_name}님이에요."
        return TurnResult("대화", response, ("대화",), working, intent=act.intent)

    def process(self, text):
        # Explicit semantic queries take precedence over approximate dialogue matches.
        question = parse_question(text)
        if question and question[1] not in {"그", "그녀", "그것", "그거"}:
            return self._answer(*question)
        if result := self._dialogue_turn(text):
            return result

        if "?" in text or "？" in text:
            return TurnResult("미지원", "아직 이 질문은 이해하지 못했어요. 소유자나 위치를 구체적으로 물어봐 주세요.")
        event = self.perception.parse(text)
        if event is None:
            return TurnResult("미지원", "아직 이해하지 못한 표현이에요. 짧은 사건 문장으로 다시 말해 주시겠어요?")
        domain = self.perception.domain_for(event.predicate)
        if domain is None:
            return TurnResult("미지원", "해당 동작의 분야를 학습하지 못했거나 해석이 모호합니다.", event=event)
        working = self._activate(domain, [event.predicate, *event.roles.values()])
        predicate_id = self.graph.resolve(event.predicate)
        if not working or predicate_id not in working or working[predicate_id].activation <= 0:
            return TurnResult("비활성", "필요한 분야 또는 동작이 비활성 상태입니다.", event=event)
        result = TurnResult("보류", "사건을 보존했습니다. 상태 변경 연산은 아직 없습니다.",
                            (domain,), working, event)
        if event.unresolved:
            result.message = "해석되지 않은 어절이 있어 상태를 변경하지 않았습니다: " + ", ".join(event.unresolved)
            return result
        index = len(self.events)
        self.last_answer = None
        self.events.append((text.strip(), event))
        if event.negated or event.tense == "미래":
            result.message = "부정 또는 미래 사건을 보존했습니다. 현재 상태는 변경하지 않았습니다."
            return result
        operation = OPERATIONS.get(event.predicate)
        if operation is None or operation[0] != domain:
            return result
        _, entity_role, value_role, previous_role = operation
        required = {entity_role, value_role}
        if previous_role:
            required.add(previous_role)
        if not required.issubset(event.roles):
            result.message = "필수 역할이 부족하여 상태를 변경하지 않았습니다: " + ", ".join(sorted(required - event.roles.keys()))
            return result
        entity, value = event.roles[entity_role], event.roles[value_role]
        previous = self.facts[domain].get(entity)
        if previous_role and previous and previous.value != event.roles[previous_role]:
            result.status = "모순"
            result.message = f"기존 소유자({previous.value})와 주는 사람({event.roles[previous_role]})이 달라 변경을 보류했습니다."
            return result
        # ponytail: one named object and chronological input; add entity IDs and
        # explicit timestamps for multiple identical objects or out-of-order facts.
        evidence = previous.evidence + (index,) if previous_role and previous else (index,)
        self.facts[domain][entity] = Fact(value, evidence)
        self.focus[domain] = entity
        if domain == "소유":
            self.focus["위치"] = value
        result.status = "갱신"
        result.message = f"{domain} 기억 갱신: {entity} → {value}"
        return result
