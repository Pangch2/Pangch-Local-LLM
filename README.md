# Pangch-Local-LLM

`concept_graph_llm_prototype.py`는 개념 그래프에서 활성화를 전파해 질의와 관련된 개념 및 추론 경로를 보여 주는 Python 프로토타입입니다. LLM 학습이나 문장 생성 기능은 없습니다.

## 실행

Python 3.10 이상과 표준 라이브러리만 필요합니다.

```sh
python concept_graph_llm_prototype.py
```

실행하면 `질문:` 프롬프트가 표시됩니다. 예를 들어 `사과가 나무에서 떨어지는 이유는 무엇일까?`를 입력하면 `사과`, `나무`, `떨어지는`을 찾고 `사과 → 물체 → 질량 → 중력` 경로 등을 출력합니다. 빈 입력은 종료합니다. 예제 그래프의 개념과 관계는 한국어를 기준으로 하며, 영어는 입력 별칭으로도 인식합니다.

## 동작 방식

1. `KeywordPerception`이 입력 문자열에서 등록된 개념명·별칭을 찾아 시드 활성도 `1.0`을 부여합니다.
2. `ActivationRouter`가 방향 그래프의 연결을 따라 활성도를 전파합니다. 연결 강도, 신뢰도, 관계별 가중치, 감쇠율, 비용 및 억제 연결을 반영합니다.
3. 임계값보다 작은 전파는 버리고, 각 노드의 상위 경로와 전역 `top_k`개 활성 노드만 유지합니다.
4. `ReasoningEngine`이 활성화된 개념의 강도순 목록, 선택된 경로, 활성화된 관계를 조회합니다.

전파값은 대략 다음과 같습니다.

```text
source activation × edge strength × edge confidence × relation gain × decay × (1 - cost)
```

억제 연결은 결과의 부호를 음수로 만듭니다. 최종 활성도는 `-1.0`~`1.0` 범위로 제한됩니다.

## 그래프 구성

`build_demo_graph()`에서 개념, 별칭, 관계를 정의합니다. 새 그래프는 같은 방식으로 만들 수 있습니다.

```python
from concept_graph_llm_prototype import ActivationRouter, ConceptGraph, KeywordPerception

graph = ConceptGraph()
graph.add_concept("비", aliases=["rain"])
graph.add_concept("젖은 땅", aliases=["wet ground"])
graph.connect("비", "젖은 땅", "원인", strength=0.9)

seeds = KeywordPerception(graph).encode("비가 온다")
working = ActivationRouter(graph).activate(seeds)
```

`connect()`는 `strength`, `confidence`, `cost`, `inhibitory`, `bidirectional` 옵션을 지원합니다. `ActivationRouter`에서는 `decay`, `threshold`, `top_k`, `max_depth`와 관계별 가중치를 조절할 수 있습니다.

## 한계

- 입력 인식은 학습된 언어 이해가 아닌 별칭의 부분 문자열 매칭입니다.
- 그래프와 지식은 실행 중 메모리에만 있으며 자동 학습·저장 기능이 없습니다.
- 활성도는 정답 확률이나 사실성 판정이 아닙니다.
