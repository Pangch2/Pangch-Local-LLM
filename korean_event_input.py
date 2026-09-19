"""Small supervised Korean event encoder; no external dependencies."""
from collections import Counter, defaultdict
from dataclasses import dataclass
import argparse
import json
from pathlib import Path
import re


@dataclass
class KoreanEvent:
    predicate: str
    roles: dict[str, str]
    tense: str
    negated: bool
    unresolved: list[str]

    def seeds(self, graph):
        # Mention activation is not an assertion that the event happened.
        names = [self.predicate, *self.roles.values()]
        return {node_id: 1.0 for name in names
                if (node_id := graph.resolve(name)) is not None}


class KoreanEventInput:
    """Learn suffix-role and surface-predicate associations from labeled text.

    ponytail: single-clause, single-token arguments only; replace with a learned
    span encoder when compound noun phrases and embedded clauses are required.
    """
    def __init__(self):
        self.roles = defaultdict(Counter)
        self.predicates = defaultdict(Counter)
        self.domains = defaultdict(Counter)

    @staticmethod
    def tokens(text):
        return re.findall(r"[가-힣A-Za-z0-9_]+", text)

    def learn(self, example):
        text = example["text"]
        tokens = self.tokens(text)
        surface = example["surface"]
        predicate = example["predicate"]
        tense = example["tense"]
        negated = example["negated"]
        roles = example["roles"]
        domain = example.get("domain")
        if domain is not None and (not isinstance(domain, str) or not domain.strip()):
            raise ValueError("분야 주석은 빈 문자열이 아닌 이름이어야 합니다")
        if (not isinstance(negated, bool) or tense not in ("과거", "현재", "미래")
                or not isinstance(predicate, str) or not predicate.strip()
                or not isinstance(roles, dict)):
            raise ValueError("서술어, 시제, 부정, 역할 주석을 확인하세요")
        surface_tokens = self.tokens(surface)
        if not surface_tokens or tokens[-len(surface_tokens):] != surface_tokens:
            raise ValueError("서술어 표현은 문장 끝의 어절과 일치해야 합니다")
        arguments = tokens[:-len(surface_tokens)]
        updates = []
        for role, annotation in roles.items():
            word, value = annotation["surface"], annotation["value"]
            if (role not in ("행위자", "대상", "받는사람", "주는사람", "장소")
                    or not value or word not in arguments
                    or not word.startswith(value) or word == value):
                raise ValueError("역할 주석은 문장에 있는 '명사+조사'여야 합니다")
            updates.append((word[len(value):], role))
        for suffix, role in updates:
            self.roles[predicate + ":" + suffix][role] += 1
        self.predicates[" ".join(surface_tokens)][(predicate, tense, negated)] += 1
        if domain is not None:
            self.domains[predicate][domain] += 1

    def domain_for(self, predicate):
        return self.winner(self.domains.get(predicate, Counter()))

    @staticmethod
    def winner(counts):
        ranked = counts.most_common(2)
        if not ranked or (len(ranked) == 2 and ranked[0][1] == ranked[1][1]):
            return None
        return ranked[0][0]

    def parse(self, text):
        tokens = self.tokens(text)
        # Exact suffix lookup touches only expressions present in this input.
        for start in range(len(tokens)):
            counts = self.predicates.get(" ".join(tokens[start:]))
            if counts and (meaning := self.winner(counts)) is not None:
                break
        else:
            return None
        roles, unresolved = {}, []
        ambiguous = set()
        for word in tokens[:start]:
            for cut in range(1, len(word)):
                counts = self.roles.get(meaning[0] + ":" + word[cut:])
                role = self.winner(counts) if counts else None
                if role:
                    if role in roles or role in ambiguous:
                        roles.pop(role, None)
                        ambiguous.add(role)
                        unresolved.append(word)
                    else:
                        roles[role] = word[:cut]
                    break
            else:
                unresolved.append(word)
        return KoreanEvent(meaning[0], roles, meaning[1], meaning[2], unresolved)

    def save(self, path):
        payload = {"version": 2, "roles": dict(self.roles), "domains": dict(self.domains), "predicates": {
            key: [[*label, count] for label, count in counts.items()]
            for key, counts in self.predicates.items()}}
        Path(path).write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path):
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        if data.get("version") not in (1, 2):
            raise ValueError("지원하지 않는 입력 모델 버전입니다")
        model = cls()
        for suffix, counts in data["roles"].items():
            model.roles[suffix].update(counts)
        for predicate, counts in data.get("domains", {}).items():
            model.domains[predicate].update(counts)
        for surface, rows in data["predicates"].items():
            for predicate, tense, negated, count in rows:
                model.predicates[surface][(predicate, tense, negated)] = count
        return model


def train_file(path):
    model = KoreanEventInput()
    for line_number, line in enumerate(Path(path).read_text(encoding="utf-8").splitlines(), 1):
        if line.strip():
            try:
                model.learn(json.loads(line))
            except (ValueError, KeyError, TypeError) as error:
                raise ValueError(f"학습 자료 {line_number}행 오류: {error}") from error
    return model


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="한국어 사건 입력 계층 학습")
    parser.add_argument("data", help="정답 주석 JSONL 파일")
    parser.add_argument("output", help="학습 모델 JSON 파일")
    args = parser.parse_args()
    train_file(args.data).save(args.output)
    print(f"학습 모델 저장 완료: {args.output}")
