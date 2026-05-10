# Design Benchmark  

## 1. Benchmark Objective

본 benchmark의 목적은 Grounding DINO가 단순히 prompt에 포함된 객체를 탐지하는 수준을 넘어, prompt에 포함된 **context**, **relation**, **action**, **dangerousness**를 어느 정도 반영하는지 평가하는 것이다.

특히 다음 두 가지 질문에 답하는 것을 목표로 한다.

1. 모델은 기본적인 object grounding 성능을 유지하는가?
2. 모델은 동일한 object composition 안에서도 서로 다른 semantic context를 구분할 수 있는가?

이를 위해 benchmark는 크게 두 축으로 구성한다.

- **Situation Prompt Test**
- **Baseline Test**

---

# 2. Dataset Construction Guide

## 2.1 Image Selection

각 test sample은 하나의 이미지와 여러 개의 prompt로 구성한다.

이미지는 다음 조건을 만족하도록 선택한다.

- 사람이 포함된 이미지
- 객체 간 interaction이 명확한 이미지
- 단순 object presence만으로는 상황을 판단하기 어려운 이미지
- dangerous / non-dangerous context를 비교할 수 있는 이미지
- 하나의 이미지 안에 여러 object가 동시에 존재하는 이미지

예를 들어, knife가 포함된 이미지를 사용할 경우 다음과 같이 구분한다.

| Image Type | Description |
|---|---|
| Dangerous | 사람이 칼로 다른 사람을 위협하는 장면 |
| Non-dangerous | 요리사가 칼로 음식을 자르는 장면 |
| Ambiguous | 사람이 칼을 들고 있지만 의도가 명확하지 않은 장면 |

---

## 2.2 Prompt Set Construction

각 이미지에 대해 prompt를 단계적으로 구성한다.

하나의 prompt set은 다음 네 단계로 구성한다.

### Level 1. Object-level Prompt

단일 객체만 포함하는 prompt이다.

예시:

- "person"
- "knife"
- "car"
- "dog"

목적은 모델이 기본 object grounding을 수행할 수 있는지 확인하는 것이다.

---

### Level 2. Object Composition Prompt

두 개 이상의 객체가 함께 등장하는 prompt이다.

예시:

- "person with knife"
- "person near car"
- "person and dog"
- "child with suitcase"

목적은 모델이 여러 객체의 조합을 grounding할 수 있는지 확인하는 것이다.

---

### Level 3. Relation / Action Prompt

객체 간 관계나 행동이 포함된 prompt이다.

예시:

- "person holding knife"
- "person riding horse"
- "doctor helping patient"
- "firefighter rescuing person"

목적은 단순 객체 조합이 아니라 object-action relation을 반영하는지 평가하는 것이다.

---

### Level 4. Situation Prompt

맥락, 의도, 위험성, 사회적 의미가 포함된 prompt이다.

예시:

- "a man is threatening someone with a knife"
- "a person injured in a car accident"
- "a firefighter rescuing an unconscious person"
- "a chef cooking with a knife"

목적은 모델이 high-level semantic context를 반영하는지 평가하는 것이다.

---

# 3. Situation Prompt Test

## 3.1 Goal

Situation Prompt Test는 모델이 동일하거나 유사한 object composition 안에서 서로 다른 contextual meaning을 구분할 수 있는지 평가한다.

즉, 모델이 단순히 "person"과 "knife"를 찾는 것이 아니라, 해당 장면이 threatening인지, cooking인지, rescuing인지, injured context인지 구분할 수 있는지를 본다.

---

## 3.2 Prompt Pair Design

Situation Prompt Test에서는 contrastive prompt pair를 구성한다.

하나의 이미지 또는 유사한 이미지 쌍에 대해 다음과 같은 prompt pair를 만든다.

### Case 1. Dangerous vs Non-dangerous

| Prompt Type | Prompt |
|---|---|
| Object | "person" |
| Object | "knife" |
| Composition | "person with knife" |
| Dangerous Situation | "a man is threatening someone with a knife" |
| Non-dangerous Situation | "a chef cooking with a knife" |

평가 포인트:

- dangerous image에서 dangerous prompt의 score가 높은가?
- non-dangerous image에서 dangerous prompt가 과도하게 activate되지 않는가?
- "person with knife"와 dangerous prompt의 결과가 동일하게 나오지 않는가?

---

### Case 2. Accident vs Resting

| Prompt Type | Prompt |
|---|---|
| Object | "person" |
| Pose | "person lying on the ground" |
| Dangerous Situation | "a person injured in a car accident" |
| Rescue Situation | "a firefighter rescuing an unconscious person" |
| Non-dangerous Situation | "a person lying on the grass" |

평가 포인트:

- 단순히 누워 있는 사람과 injured person을 구분하는가?
- car, road, firefighter 등 surrounding context를 반영하는가?
- rescue 상황에서 firefighter와 unconscious person의 relation을 반영하는가?

---

### Case 3. Helping vs Attacking

| Prompt Type | Prompt |
|---|---|
| Object | "person" |
| Composition | "two people" |
| Relation | "person holding another person" |
| Positive Situation | "person helping another person" |
| Negative Situation | "person attacking another person" |

평가 포인트:

- 같은 physical contact라도 helping과 attacking을 구분하는가?
- 모델이 실제 visual evidence 없이 prompt 의미만으로 과도하게 box를 생성하지 않는가?

---

# 4. Baseline Test

Baseline Test는 모델이 Situation Prompt Test를 수행하면서도 기본적인 object grounding 능력을 유지하는지 확인하기 위한 기준 평가이다.

Baseline Test는 다음 두 가지로 구성한다.

1. **Single Object Detection**
2. **Compositional Generalization**

---

## 4.1 Single Object Detection

### Goal

단일 객체 prompt에 대해 모델이 정확한 bounding box를 생성하는지 평가한다.

### Prompt Examples

- "person"
- "knife"
- "car"
- "dog"
- "fire extinguisher"
- "umbrella"
- "horse"
- "suitcase"

### Annotation

각 이미지에 대해 object-level bounding box를 annotation한다.

필수 annotation:

```json
{
  "image_id": "sample_001",
  "prompt": "knife",
  "target_objects": [
    {
      "label": "knife",
      "bbox": [x1, y1, x2, y2]
    }
  ]
}
````

### Metrics

* IoU
* Precision
* Recall
* mAP
* Confidence score

---

## 4.2 Compositional Generalization

### Goal

모델이 여러 object, action, relation이 결합된 새로운 조합적 개념(compositional concepts)을 grounding할 수 있는지 평가한다.

이 테스트는 Situation Prompt Test와 다르게 danger나 intention 같은 high-level context보다, object-action-relation 조합 자체에 집중한다.

### Prompt Examples

* "person riding horse"
* "person holding umbrella"
* "child sitting on suitcase"
* "doctor helping injured person"
* "dog sitting beside person"
* "person standing behind car"

### Evaluation Focus

다음 요소를 평가한다.

* object composition을 올바르게 찾는가?
* relation-aware grounding이 가능한가?
* seen phrase가 아닌 unseen composition에도 일반화하는가?
* prompt가 길어져도 detection이 안정적인가?

### Positive / Negative Prompt Pair

가능하면 다음과 같이 비슷하지만 관계가 다른 prompt를 함께 평가한다.

| Positive Prompt             | Negative Prompt                  |
| --------------------------- | -------------------------------- |
| "person riding horse"       | "person standing beside horse"   |
| "person holding umbrella"   | "umbrella beside person"         |
| "child sitting on suitcase" | "child standing near suitcase"   |
| "doctor helping patient"    | "doctor standing beside patient" |

---

# 5. Evaluation Protocol

## 5.1 Model Output

각 prompt에 대해 Grounding DINO는 다음 값을 출력한다.

* bounding box
* confidence score
* predicted phrase
* image id
* prompt id

결과는 다음 형식으로 저장한다.

```json
{
  "image_id": "sample_001",
  "prompt_id": "p001",
  "prompt": "a man is threatening someone with a knife",
  "predictions": [
    {
      "bbox": [x1, y1, x2, y2],
      "score": 0.73,
      "phrase": "man"
    }
  ]
}
```

---

## 5.2 Situation Score Comparison

Situation Prompt Test에서는 단일 prompt의 detection 성공 여부만 보지 않는다.

대신 같은 이미지에 대해 여러 prompt의 confidence score 변화를 비교한다.

예시:

| Prompt                                   | Expected Behavior                    |
| ---------------------------------------- | ------------------------------------ |
| "person"                                 | high score                           |
| "knife"                                  | high score                           |
| "person with knife"                      | high score                           |
| "a man threatening someone with a knife" | high score only in dangerous context |
| "a chef cooking with knife"              | high score only in cooking context   |

핵심은 dangerous prompt가 모든 knife image에서 높게 나오면 안 된다는 것이다.

---

## 5.3 False Situation Activation

False Situation Activation은 모델이 단순 object presence만 보고 high-level situation prompt를 잘못 활성화하는지를 측정한다.

예시:

* 이미지: chef cooking with knife
* prompt: "a man threatening someone with a knife"
* 기대 결과: low confidence 또는 no detection

이 경우 dangerous prompt가 높은 confidence를 가지면 false situation activation으로 기록한다.

---

## 5.4 Cross-prompt Consistency

Cross-prompt consistency는 prompt가 구체화될 때 detection 결과가 의미적으로 일관적인지 평가한다.

예시:

* "person"
* "person with knife"
* "a man threatening someone with a knife"

위 세 prompt의 bounding box가 모두 동일하게 person만 잡는다면, 모델이 situation-specific grounding을 하지 못했을 가능성이 있다.

반대로 threatening prompt에서 victim, aggressor, knife 주변 영역까지 더 적절하게 반영한다면 contextual grounding이 더 잘 된 것으로 볼 수 있다.

---

# 6. Suggested Metrics

## 6.1 Object-level Metrics

Baseline Test에서 사용한다.

* IoU
* Precision
* Recall
* mAP
* Detection Accuracy

---

## 6.2 Prompt-level Metrics

각 prompt에 대한 detection 품질을 평가한다.

* Prompt-specific confidence
* Prompt-conditioned localization accuracy
* Prompt-to-box consistency

---

## 6.3 Situation-level Metrics

Situation Prompt Test에서 사용한다.

### Situation Discrimination Score

dangerous prompt가 dangerous image에서 non-dangerous image보다 얼마나 높은 score를 가지는지 측정한다.

```text
SDS = score(dangerous prompt, dangerous image)
      - score(dangerous prompt, non-dangerous image)
```

SDS가 클수록 모델이 dangerous situation을 더 잘 구분한다고 볼 수 있다.

---

### False Situation Activation Rate

non-dangerous image에서 dangerous prompt가 잘못 activate되는 비율이다.

```text
FSAR = (# false activated dangerous prompts) / (# non-dangerous samples)
```

낮을수록 좋다.

---

### Prompt Specificity Gap

구체적인 situation prompt와 일반 object prompt 간 confidence 차이를 측정한다.

```text
PSG = score(situation prompt) - score(object composition prompt)
```

이 값은 단순히 높다고 좋은 것이 아니라, 올바른 context에서만 높아야 한다.

---

# 7. Annotation Guide

## 7.1 Required Annotation Fields

각 sample은 다음 정보를 포함해야 한다.

```json
{
  "image_id": "sample_001",
  "image_path": "images/sample_001.jpg",
  "category": "dangerous_knife",
  "prompts": [
    {
      "prompt_id": "p001",
      "prompt": "person",
      "type": "object",
      "expected": "positive"
    },
    {
      "prompt_id": "p002",
      "prompt": "a man threatening someone with a knife",
      "type": "situation",
      "expected": "positive"
    },
    {
      "prompt_id": "p003",
      "prompt": "a chef cooking with a knife",
      "type": "situation",
      "expected": "negative"
    }
  ],
  "annotations": [
    {
      "label": "person",
      "bbox": [x1, y1, x2, y2]
    },
    {
      "label": "knife",
      "bbox": [x1, y1, x2, y2]
    }
  ]
}
```

---

## 7.2 Prompt Type

각 prompt는 다음 중 하나로 분류한다.

| Type        | Description               |
| ----------- | ------------------------- |
| object      | 단일 객체 prompt              |
| composition | 여러 객체가 결합된 prompt         |
| relation    | 객체 간 관계 또는 행동이 포함된 prompt |
| situation   | 맥락, 의도, 위험성 등이 포함된 prompt |

---

## 7.3 Expected Label

각 prompt는 해당 이미지에 대해 positive 또는 negative로 표시한다.

| Expected  | Meaning                |
| --------- | ---------------------- |
| positive  | 이미지와 prompt가 의미적으로 일치  |
| negative  | 이미지와 prompt가 의미적으로 불일치 |
| ambiguous | 사람마다 판단이 다를 수 있음       |

---

# 8. Recommended Benchmark Categories

## 8.1 Dangerous Object Context

* threatening with knife
* chef cooking with knife
* person holding knife
* knife on table

## 8.2 Accident / Injury Context

* person lying on ground
* injured person
* car accident
* firefighter rescuing person

## 8.3 Human Interaction Context

* person helping another person
* person attacking another person
* doctor treating patient
* police arresting person

## 8.4 Object-Action Composition

* person riding horse
* person holding umbrella
* child sitting on suitcase
* person carrying box

---

# 9. Expected Analysis

최종 분석에서는 다음 질문에 답해야 한다.

1. 모델은 single object detection에서 충분한 성능을 보이는가?
2. 모델은 object composition을 안정적으로 grounding하는가?
3. 모델은 relation-aware prompt에 대해 다른 box 또는 confidence 변화를 보이는가?
4. 모델은 dangerous situation과 non-dangerous situation을 구분하는가?
5. 모델은 단순 object presence만으로 false situation activation을 발생시키는가?
6. 모델은 prompt가 구체화될수록 semantic specificity를 반영하는가?

---

# 10. Summary

본 benchmark는 Grounding DINO가 단순 object detector로 동작하는지, 아니면 prompt에 포함된 relation, action, context, dangerousness까지 반영하는지를 평가하기 위해 설계되었다.

Baseline Test는 기본 object grounding 능력을 확인하고, Situation Prompt Test는 모델의 high-level semantic grounding 능력을 분석한다.

따라서 두 테스트를 함께 수행함으로써, 모델의 object-level grounding 성능과 situation-aware grounding 성능을 동시에 평가할 수 있다.


