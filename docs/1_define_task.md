# Define Task 

**Motivation:** [Grouding DINO](https://github.com/IDEA-Research/GroundingDINO)는 텍스트 프롬프트가 주어졌을 때, 프롬프트에 맞는 객체를 탐지하는 모델이다. 그렇다면 **어떠한 맥락이 포함된 상황(situation)을 잘 감지하는지** 어떻게 평가할 수 있을까? 

## 1. Situation Prompt Test 

예를 들어, 
- "person"
- "knife"
- "person with knife"
- "a men is threatening someone with a knife" 

는 공통적으로 사람 또는 칼에 대한 상황을 나타내지만, 각각이 가지고 있는 상황적인 맥락이 다르다. 

특히 우리는 마지막 "a man is threatening someone with a knife" 와 같이 동일하게 "person with knife"가 포함되어 있더라도, 모델이 위협적인 상황(dangerous situation) 과 단순히 칼을 들고 있는 상황(non-dangerous situation)을 구분할 수 있는지 여부를 확인하고 싶다. 

또 다른 예시는, 

- "person"
- "person lying on the ground"
- "a person injured in a car accident"
- "a firefighter rescuing an unconscious person"

과 같이 동일한 객체 조합이더라도 상황적 의미와 관계성이 달라지는 경우이다.

따라서, 다음과 같이 **Situation Prompt Test**를 설계한다: 

- 동일하거나 유사한 객체 구성을 포함하지만, 서로 다른 상황적 의미를 가지는 프롬프트 쌍(prompt pair)을 구성한다.
- 각 프롬프트에 대해 모델이 생성한 bounding box와 confidence score를 평가한다.
- 단순 객체 존재 여부(object presence)가 아니라, 객체 간 관계(relation), 행위(action), 위험도(danger level), 맥락(context)을 얼마나 반영하는지를 측정한다.
- 특히 다음과 같은 능력을 중점적으로 평가한다:
    - **Situation Sensitivity**: 동일 객체 조합 내에서 상황 차이를 구분할 수 있는가?
    - **Relational Understanding**: 객체 간 상호작용과 관계를 이해하는가?
    - **Contextual Specificity**: 더 구체적인 상황 프롬프트에 대해 탐지 결과가 적절히 변화하는가?
    - **False Situation Activation**: 단순 객체 존재만으로 위험 상황을 과도하게 탐지하지 않는가?

이를 통해 모델이 단순 open-vocabulary object detection을 넘어, 상황적 의미를 어느 정도 이해하는지를 분석할 수 있다.

## 2. Baseline Test 

우리는 모델이 **Situation Prompt Test**를 잘 수행하면서도, 기본적인(**Baseline**) 객체 탐지 성능을 여전히 잘 수행하는지를 함께 확인해야 한다. 

따라서, 다음과 같이 **Baseline Test**를 설계한다: 

### 2-1. Single Object Detection

기본적인 open-vocabulary object detection 성능을 평가하기 위해, 단일 객체에 대한 탐지 성능을 측정한다.

예를 들어,

- "person"
- "car"
- "dog"
- "knife"
- "fire extinguisher"

와 같은 단일 객체 프롬프트를 입력하고, 모델이 해당 객체를 정확히 탐지하는지를 평가한다.

이를 통해 모델이 상황 이해 이전에, 기본적인 객체 grounding 능력을 충분히 유지하고 있는지를 확인한다.

평가 지표로는 다음을 사용할 수 있다:

- Bounding box IoU
- Detection accuracy
- Recall / Precision
- Mean Average Precision (mAP)

### 2-2. Compositional Generalization 

모델이 단순히 학습된 문장을 암기하는 것이 아니라, 여러 개의 객체와 관계를 조합하여 새로운 조합적 개념(compositional concepts)을 이해할 수 있는지를 평가한다.

예를 들어,

- "person riding horse"
- "person holding umbrella"
- "child sitting on suitcase"
- "doctor helping injured person"

과 같이 객체, 속성, 행동, 관계가 결합된 프롬프트를 사용한다.

특히 학습 데이터에서 자주 등장하지 않았을 가능성이 높은 조합(compositional combinations)을 포함하여, 모델의 compositional generalization 능력을 측정한다.

이를 통해 다음 능력을 평가할 수 있다:

- 객체 조합 이해 능력
- 관계 기반 grounding 능력
- unseen composition에 대한 일반화 성능
- 상황 프롬프트 확장에 대한 강건성

결과적으로, Baseline Test는 모델이 기본 객체 탐지 성능을 유지하면서도, Situation Prompt Test에서 요구되는 고차원적 이해 능력까지 확장할 수 있는지를 함께 검증하기 위한 기준 역할을 한다.

---

