# TracIn Milestone Plan

## Goal
`TracIn`을 `src/gif/influence/`에 repo-native 방식으로 구현하고,  
`scripts/train/`에서 필요한 checkpoint trajectory를 저장한 뒤,  
저장된 checkpoint들을 사용해 **model edit용 influence/update**를 계산할 수 있게 만든다.

이 milestone의 목적은 두 가지다.
- TracIn을 현재 GIF benchmark에 올릴 수 있는 최소 구현을 만든다.
- TracIn이 원래 data attribution 방법이라는 점을 유지하면서도, model edit용 update proxy로 강제 적용할 수 있는 경로를 만든다.

---

## Scope

이번 milestone에서 다룰 범위:
- `src/gif/influence/tracin.py` 실제 구현
- training 중 checkpoint trajectory 저장
- checkpoint trajectory를 읽어 TracIn score / update 계산
- MNIST 실험 경로에 연결

이번 milestone에서 제외:
- 대규모 per-step checkpointing
- Captum / 외부 repo integration
- distributed training 지원
- 논문용 figure 생성

---

## Design Decision

### `repo-native`의 의미
여기서 `repo-native`는 외부 오픈소스 코드를 그대로 vendoring/submodule로 붙이는 것이 아니라,
**현재 이 레포의 구조와 API에 맞게 재구현한다**는 뜻이다.

즉:
- 구현 위치는 `src/gif/influence/tracin.py`
- 공통 미분 연산은 기존 `gif.influence.common` 재사용
- solver/selection/script 구조와 충돌하지 않게 연결
- benchmark script가 동일한 방식으로 method를 호출할 수 있게 맞춤

목표는 외부 구현 재사용이 아니라
**이 레포의 method family 안에 자연스럽게 들어가는 TracIn baseline**을 만드는 것이다.

### TracIn의 이 레포 내 역할
이 레포에서 TracIn은 원래 목적 그대로의 data attribution method가 아니라,
**model edit benchmark에 강제로 올리는 baseline**이다.

따라서 이 구현의 출력은 두 단계로 나눈다.

1. `score` 단계
- training examples 또는 target examples에 대한 TracIn influence score 계산

2. `update` 단계
- score 또는 gradient accumulation을 바탕으로
- parameter-space edit vector를 구성

즉 최종적으로는 `compute_update(...)` 경로가 필요하다.

---

## Target API

## `src/gif/influence/tracin.py`

예상 공개 API:

```python
class TracIn:
    def compute_scores(...)
    def compute_update(...)
```

또는 함수형 API:

```python
def tracin_scores(...)
def tracin_update(...)
```

최소 구현에서는 아래가 필요하다.

- `load_checkpoints(...)`
- `compute_gradient(...)` 재사용
- `tracin_score_from_checkpoints(...)`
- `tracin_update_from_checkpoints(...)`

---

## Checkpoint Policy

TracIn은 training trajectory가 필요하므로, train script가 checkpoint를 하나만 저장하면 안 된다.

최소 checkpoint 정책:
- epoch 끝마다 checkpoint 저장
- optional: best checkpoint도 별도 저장

저장 형식:

```python
{
    "net": model.state_dict(),
    "epoch": int,
    "model": str,
    "optimizer": optimizer.state_dict(),   # optional
    "lr": float,                           # optional but recommended
}
```

checkpoint 디렉토리 예시:

```text
checkpoints/
  mnist_resnet34/
    epoch_001.pth
    epoch_002.pth
    ...
    best.pth
```

같은 규칙을 deep FCN에도 적용한다.

---

## Milestones

## Progress Checklist

- [x] M1. TracIn Checkpoint Format
- [x] M2. TracIn Score Implementation
- [x] M3. TracIn Update for Model Edit
- [x] M4. Script Integration
- [x] M5. Benchmark Readiness
- [ ] Final end-to-end validation

진행 중인 항목은 작업 시점에 하나만 활성 상태로 본다.
각 milestone은 **구현 직후 최소 테스트를 바로 추가하고 바로 실행**하는 방식으로 진행한다.

## M1. TracIn Checkpoint Format

목표:
- train script가 trajectory checkpoint를 저장하게 만들기

작업:
- [x] `scripts/train/_mnist_train_common.py` 확장
- [x] epoch별 checkpoint 저장 옵션 추가
- [x] 저장 디렉토리 규칙 고정
- [x] `best checkpoint only`와 `trajectory checkpoint`를 분리

완료 기준:
- ResNet34 학습 시 epoch별 checkpoint가 저장됨
- deep FCN 학습 시 epoch별 checkpoint가 저장됨

즉시 테스트:
- [x] checkpoint naming/unit test 추가
- [x] trajectory directory 생성/unit test 추가
- [x] `scripts/train/train_resnet34_mnist.py` 짧은 smoke train 실행
- [x] `scripts/train/train_deep_fcn_mnist.py` 짧은 smoke train 실행
- [x] checkpoint 파일 개수와 naming 확인

---

## M2. TracIn Score Implementation

목표:
- 저장된 checkpoint trajectory를 읽어서 TracIn score를 계산

작업:
- [x] `src/gif/influence/tracin.py`에 checkpoint loader 추가
- [x] target batch gradient 계산
- [x] train/retain batch gradient 계산
- [x] checkpoint별 gradient inner product accumulation

최소 방식:
- epoch checkpoint 기준
- batch 단위 gradient
- learning-rate weighting은 optional

완료 기준:
- 동일 target/query pair에 대해 deterministic한 score 출력
- ResNet34와 FCN 둘 다 score 계산 가능

즉시 테스트:
- [x] checkpoint loader unit test 추가
- [x] checkpoint ordering unit test 추가
- [x] toy model score determinism test 추가
- [x] MNIST small checkpoint trajectory integration test 추가

---

## M3. TracIn Update for Model Edit

목표:
- TracIn을 model edit benchmark에 올리기 위한 update proxy 구현

작업:
- [x] score-only API와 별도로 update API 설계
- [x] target examples에 대한 gradient accumulation을 checkpoint trajectory로 weighted sum
- [x] parameter-space update vector 생성
- [x] GIF와 동일한 selector/update pipeline에 연결 가능한 형태로 출력

핵심 결정:
- full-parameter update를 먼저 만들고
- 필요하면 selector가 subset 적용

완료 기준:
- `compute_update(...) -> vector` 경로가 존재
- selector와 결합 가능

즉시 테스트:
- [x] toy model update direction test 추가
- [x] update tensor shape/unit test 추가
- [x] selector 결합 smoke test 추가
- [x] MNIST small integration test 추가

---

## M4. Script Integration

목표:
- 기존 search / unlearning script에서 TracIn을 method option으로 선택 가능하게 만들기

작업:
- [x] `scripts/search/_mnist_unlearning_common.py`에 method branch 추가
- [x] `caps`, `gif`, `tracin` 같은 방식으로 method 선택
- [x] checkpoint trajectory path 입력 추가

완료 기준:
- CLI에서 TracIn 선택 가능
- checkpoint trajectory를 읽어 influence/update 계산 가능

즉시 테스트:
- [x] CLI parse test 또는 smoke test 추가
- [x] `--help` 점검
- [x] small smoke run 실행

---

## M5. Benchmark Readiness

목표:
- GIF vs TracIn 비교 실험이 가능한 상태 만들기

작업:
- [x] 공통 metric 출력 유지
- [ ] retain/self_acc 기준 비교 가능하게 정리
- [ ] residual은 TracIn에는 직접 대응되지 않으므로 별도 표기 규칙 정리

완료 기준:
- 같은 모델/데이터에서 GIF와 TracIn을 같은 script framework로 실행 가능

즉시 테스트:
- [x] ResNet18 benchmark smoke run (same framework, same schema)
- [x] deep FCN baseline smoke run
- [x] output schema 점검

---

## Test Plan

원칙:
- 각 milestone 구현 직후 최소 단위 테스트를 추가한다.
- milestone이 끝날 때마다 그 milestone 범위의 테스트를 즉시 실행한다.
- 모든 milestone 구현이 끝난 뒤 마지막에 전체 테스트를 한 번 더 실행한다.

### Unit Tests
- checkpoint loader test
- checkpoint ordering test
- gradient accumulation test
- TracIn score symmetry / determinism sanity test

### Integration Tests
- small toy trajectory에서 score 계산
- saved checkpoints로 update 계산
- update 적용 후 target loss 변화 확인

### Script Smoke Tests
- train script checkpoint 저장
- TracIn score script 실행
- TracIn update script 실행

---

## Risks

### Risk 1. Checkpoint 수 부족
epoch checkpoint만으로는 TracIn 신호가 너무 거칠 수 있다.

대응:
- 최소 구현은 epoch checkpoint
- 필요하면 later milestone에서 step checkpoint 옵션 추가

### Risk 2. Per-sample gradient 부재
정교한 TracIn 구현은 per-example gradient가 필요할 수 있다.

대응:
- 초기 구현은 batch-level approximation
- 이후 per-sample gradient 확장 가능

### Risk 3. Model edit용 update 정의가 애매함
TracIn은 원래 update method가 아니다.

대응:
- benchmark baseline이라는 점을 명시
- score와 update를 분리해서 구현

---

## Immediate Next Step

다음 작업은 `M5`이다.

즉 먼저:
- GIF와 TracIn 비교 실행
- retain/self_acc 기준 비교
- output schema 정리

즉 먼저 **같은 benchmark script framework에서 GIF와 TracIn 결과를 나란히 비교 가능하게 정리**한다.

---

## Final Validation

모든 milestone 완료 후 마지막에 한 번 더 아래를 실행한다.

- [ ] `python3 -m pytest -q`
- [ ] 대표 train script smoke run
- [ ] 대표 TracIn score/update smoke run
- [ ] GIF vs TracIn benchmark smoke run
