"""Supervised sparse character n-gram retrieval for Korean dialogue acts."""
from collections import Counter, defaultdict
from dataclasses import dataclass
import argparse
import json
from pathlib import Path
import re
import unicodedata


def introduced_name(text):
    text = unicodedata.normalize("NFKC", text).strip().rstrip(".!。！")
    match = re.fullmatch(
        r"(?:내 이름은|제 이름은|나는|저는)\s+([가-힣A-Za-z]{1,20}?)(?:입니다|이에요|예요|이야|야)", text)
    return match[1] if match else None


@dataclass(frozen=True)
class DialogueAct:
    intent: str
    response: str
    score: float


class KoreanDialogue:
    """Nearest labeled examples, indexed by sparse character features.

    ponytail: surface similarity cannot infer general semantics; use a learned
    semantic encoder when measured paraphrase coverage becomes insufficient.
    """
    def __init__(self):
        self.examples = []
        self.features = []
        self.index = defaultdict(set)

    @staticmethod
    def featurize(text):
        text = unicodedata.normalize("NFKC", text)
        if name := introduced_name(text):
            text = text.replace(name, "이름자리", 1)
        text = "".join(re.findall(r"[가-힣a-z0-9]+", text.lower()))
        text = "^" + text + "$"
        return {text[i:i + n] for n in (2, 3) for i in range(len(text) - n + 1)}

    def learn(self, example):
        if not isinstance(example, dict) or any(
            not isinstance(example.get(key), str) or not example[key].strip()
            for key in ("text", "intent", "response")
        ):
            raise ValueError("대화 예문에는 비어 있지 않은 text, intent, response가 필요합니다")
        features = self.featurize(example["text"])
        index = len(self.examples)
        self.examples.append({key: example[key] for key in ("text", "intent", "response")})
        self.features.append(features)
        for feature in features:
            self.index[feature].add(index)

    def classify(self, text, *, threshold=0.60, margin=0.08):
        features = self.featurize(text)
        hits = Counter(index for feature in features for index in self.index.get(feature, ()))
        best = {}
        for index, overlap in hits.items():
            example = self.examples[index]
            score = 2 * overlap / (len(features) + len(self.features[index]))
            intent = example["intent"]
            candidate = DialogueAct(intent, example["response"], score)
            if intent not in best or score > best[intent].score:
                best[intent] = candidate
        ranked = sorted(best.values(), key=lambda act: (-act.score, act.intent))
        if not ranked or ranked[0].score < threshold:
            return None
        if len(ranked) > 1 and ranked[0].score - ranked[1].score < margin:
            return None
        return ranked[0]

    @classmethod
    def train_file(cls, path):
        model = cls()
        for number, line in enumerate(Path(path).read_text(encoding="utf-8").splitlines(), 1):
            if line.strip():
                try:
                    model.learn(json.loads(line))
                except (ValueError, TypeError) as error:
                    raise ValueError(f"대화 학습 자료 {number}행 오류: {error}") from error
        return model

    def save(self, path):
        Path(path).write_text(json.dumps({"version": 1, "examples": self.examples},
                                        ensure_ascii=False, indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path):
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        if not isinstance(data, dict) or data.get("version") != 1 or not isinstance(data.get("examples"), list):
            raise ValueError("지원하지 않는 대화 모델 형식입니다")
        model = cls()
        for example in data["examples"]:
            model.learn(example)
        return model


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="한국어 대화 의도와 응답 예문 학습")
    parser.add_argument("data", help="대화 JSONL 학습 자료")
    parser.add_argument("output", help="저장할 대화 모델 JSON")
    args = parser.parse_args()
    KoreanDialogue.train_file(args.data).save(args.output)
    print(f"대화 모델 저장 완료: {args.output}")
