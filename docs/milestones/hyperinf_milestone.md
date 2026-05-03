# HyperINF Milestone Plan

## Goal
`HyperINF`를 `src/gif/influence/`와 `src/gif/solvers/` 구조에 맞게 repo-native 방식으로 구현하고,  
현재 GIF benchmark 경로에서 **model edit baseline**으로 실행 가능하게 만든다.

이 milestone의 목적은 두 가지다.
- HyperINF의 핵심인 inverse approximation / Schulz-style iteration을 현재 레포에 맞게 재구현한다.
- LoRA/VLM 전용 원 구현을 그대로 가져오지 않고, 현재 레포의 full-model MNIST setting에서도 비교 가능한 baseline으로 정리한다.

---

## Open-Source Review

### Primary sources
- HyperINF paper: [arXiv:2410.05090](https://arxiv.org/abs/2410.05090)
- HyperINF repo: [Blackzxy/HyperINF](https://github.com/Blackzxy/HyperINF)
- DataInf paper: [arXiv:2310.00902](https://arxiv.org/abs/2310.00902)
- DataInf repo: [ykwon0407/DataInf](https://github.com/ykwon0407/DataInf)

### What the open-source code implies
- HyperINF paper and repo are centered on **inverse approximation via Schulz / hyperpower iteration**.
- The HyperINF repo README states that its mislabeled-data-detection code path uses **DataInf code** as a base, and its LLM/VLM path depends on **Prismatic-VLM**.
- DataInf itself is explicitly designed for **LoRA-tuned LLMs and diffusion models**.

### Consequence for this repo
- We should **not** vendor or submodule the upstream HyperINF code.
- We should implement a **repo-native HyperINF-style inverse solver** and then expose it through this repo's influence-method API.
- The goal is not exact reproduction of the LoRA-specific constant-cost claims. The goal is a fair **HyperINF-style model-edit baseline** inside the current benchmark.

---

## Scope

이번 milestone에서 다룰 범위:
- `HyperINF` inverse approximation solver 구현
- small exact problem에서 oracle 비교
- model edit용 influence/update 계산 경로 연결
- existing MNIST search/unlearning path에 연결

이번 milestone에서 제외:
- LoRA module support
- upstream HyperINF code vendoring/submodule integration
- LLM/VLM data selection pipeline
- Prismatic-VLM dependency
- 원 논문의 full experimental reproduction

---

## Design Decision

### `repo-native`의 의미
여기서 `repo-native`는 외부 HyperINF/DataInf 코드를 그대로 가져오는 것이 아니라,
**현재 이 레포의 `influence/`, `solvers/`, `scripts/search/` 구조에 맞게 핵심 알고리즘만 재구현한다**는 뜻이다.

즉:
- solver는 `src/gif/solvers/`에 둔다
- method wrapper는 `src/gif/influence/hyperinf.py`에 둔다
- 공통 autodiff는 기존 `gif.influence.common`을 재사용한다
- script는 기존 benchmark API를 그대로 사용한다

### HyperINF의 이 레포 내 역할
이 레포에서 HyperINF는 원 논문의 LoRA-tuned attribution tool 그대로가 아니라,
**inverse approximation 계열의 model-edit baseline**이다.

따라서 이 구현은 다음을 목표로 한다.

1. `solver` 단계
- Schulz-style inverse approximation 또는 그에 준하는 HyperINF-style iterative inverse update

2. `method` 단계
- `target_loss` gradient와 `total_loss` curvature operator를 받아
- model edit용 update vector를 만든다

즉 최종적으로는 `compute_update(...)` 경로가 필요하다.

### Repository placement
- `src/gif/solvers/hyperinf.py`
  - HyperINF inverse iteration core
- `src/gif/influence/hyperinf.py`
  - model/loss/index_list를 받아 update를 반환하는 wrapper

이렇게 나누면:
- solver 실험은 `solvers/`에서 독립적으로 진행 가능
- benchmark method wiring은 `influence/`에서 유지 가능

---

## Target API

## `src/gif/solvers/hyperinf.py`

예상 공개 API:

```python
def hyperinf_inverse(
    a_times,
    rhs,
    *,
    max_iter,
    tol,
    ...
):
    ...
```

또는 상태를 더 많이 보고 싶으면:

```python
def hyperinf_inverse(..., return_details=False):
    ...
```

최소 구현에서는 아래가 필요하다.

- scaling / initialization
- matrix-free operator `a_times(v)`
- iterative inverse update
- residual computation
- optional iteration trace

## `src/gif/influence/hyperinf.py`

예상 공개 API:

```python
class HyperInfluence:
    def compute_update(...)
```

또는 함수형 API:

```python
def hyperinf_update(...)
```

최소 구현에서는 아래가 필요하다.

- `compute_gradient(...)` 재사용
- restricted operator 구성
- HyperINF solver 호출
- subset projection / full update 연결

---

## Milestones

## Progress Checklist

- [x] M1. HyperINF Solver Contract
- [x] M2. Exact Toy Solver Validation
- [x] M3. Influence Wrapper and Restricted Update
- [x] M4. Script Integration
- [x] M5. Benchmark Readiness
- [x] Final end-to-end validation

진행 중인 항목은 작업 시점에 하나만 활성 상태로 본다.  
각 milestone은 **구현 직후 최소 테스트를 바로 추가하고 바로 실행**하는 방식으로 진행한다.

---

## M1. HyperINF Solver Contract

목표:
- HyperINF를 이 레포의 solver/method 구조에 맞게 배치하고 API를 고정

작업:
- [x] `src/gif/solvers/hyperinf.py` 생성
- [x] `src/gif/influence/hyperinf.py` placeholder를 실제 wrapper 형태로 교체
- [x] solver input/output contract 고정
- [x] details payload schema 고정

완료 기준:
- solver 파일과 method wrapper가 존재
- `from gif.influence import HyperInfluence`가 가능
- `from gif.solvers import hyperinf_inverse`가 가능

즉시 테스트:
- [x] import test 추가
- [x] empty-input / bad-shape error test 추가
- [x] public API smoke test 추가

---

## M2. Exact Toy Solver Validation

목표:
- 아주 작은 exact problem에서 HyperINF solver가 inverse approximation을 제대로 하는지 검증

작업:
- [x] tiny SPD matrix에 대한 matrix-free `a_times` 준비
- [x] exact inverse oracle 계산
- [x] HyperINF iteration 구현
- [x] residual trace / convergence trace 반환

완료 기준:
- toy SPD system에서 approximate solution이 oracle에 근접
- residual이 반복에 따라 줄어듦

즉시 테스트:
- [x] exact oracle comparison unit test 추가
- [x] residual monotonic-improvement sanity test 추가
- [x] scaling이 잘못됐을 때 divergence/guard 동작 test 추가
- [x] ill-conditioned toy matrix stress test 추가

---

## M3. Influence Wrapper and Restricted Update

목표:
- HyperINF solver를 현재 influence/update 경로에 연결

작업:
- [x] `target_loss` gradient 계산
- [x] restricted operator 구성
- [x] HyperINF solver 호출
- [x] subset update 또는 full update 반환

핵심 결정:
- GIF와 동일하게 `index_list` 기반 restricted update를 기본으로 본다
- 필요하면 full update path는 별도 옵션으로 둔다

완료 기준:
- `compute_update(...) -> vector` 경로가 존재
- selector와 결합 가능

즉시 테스트:
- [x] tiny neural net에서 update tensor shape test 추가
- [x] generalized path와 같은 operator/rhs를 썼을 때 finite result test 추가
- [x] target loss가 의도한 방향으로 움직이는 integration test 추가
- [x] model state mutation 여부 test 추가

---

## M4. Script Integration

목표:
- 기존 MNIST search / unlearning script에서 `hyperinf`를 method option으로 선택 가능하게 만들기

작업:
- [x] `scripts/search/_mnist_unlearning_common.py`에 method branch 추가
- [x] CLI option에 `hyperinf` 추가
- [x] HyperINF-specific solver args 추가

예상 추가 인자 예시:
- `--hyperinf-max-iter`
- `--hyperinf-tol`
- `--hyperinf-beta` 또는 scaling 계열 파라미터

완료 기준:
- CLI에서 HyperINF 선택 가능
- 기존 benchmark framework에서 update 계산 가능

즉시 테스트:
- [x] CLI parse test 또는 help smoke test 추가
- [x] small smoke run 실행
- [x] wrong-config error message test 추가

---

## M5. Benchmark Readiness

목표:
- GIF / TracIn / HyperINF를 같은 framework에서 비교 가능한 상태 만들기

작업:
- [x] 공통 metric 출력 유지
- [x] HyperINF 결과 schema를 기존 benchmark 저장 형식에 맞춤
- [x] solver trace를 optional details로 저장 가능하게 정리

완료 기준:
- 같은 모델/데이터에서 GIF, TracIn, HyperINF를 같은 script framework로 실행 가능

즉시 테스트:
- [x] ResNet18 benchmark smoke run
- [x] deep FCN benchmark smoke run
- [x] JSON output schema 점검

---

## Test Plan

### Unit tests
- `tests/solvers/test_hyperinf.py`
  - exact toy inverse comparison
  - residual reduction
  - bad scaling / guard behavior
  - ill-conditioned stress

- `tests/influence/test_hyperinf.py`
  - wrapper API
  - update shape
  - selector compatibility
  - model-state restoration

### Integration tests
- `tests/integration/test_hyperinf_script_integration.py`
  - CLI exposure
  - small script smoke run

- optional:
  - `tests/integration/test_resnet18_hyperinf.py`
  - `tests/integration/test_resnet34_hyperinf.py`

### Final validation
- full `pytest`
- `search_mnist_model.py` smoke run with `--schemes hyperinf`
- same benchmark path with `caps`, `tracin`, `hyperinf`

---

## Risks

### Risk 1. HyperINF paper assumptions do not transfer directly
원 논문은 LoRA/GFIM 맥락이 강하다.  
현재 레포에서는 inverse approximation 핵심만 가져오는 것이므로,
논문 수치와 동일한 성능/비용 특성을 기대하면 안 된다.

### Risk 2. Bad scaling can make the inverse iteration unstable
Schulz-style iteration은 scaling이 나쁘면 쉽게 폭주할 수 있다.  
따라서 toy exact test와 guard behavior test가 먼저 필요하다.

### Risk 3. HyperINF may collapse to “just another inverse solver”
이 레포의 설정에서는 LoRA-specific 장점이 사라질 수 있다.  
그래서 benchmark 목적은 “faithful reproduction”보다
“HyperINF-style inverse baseline”이라는 점을 명확히 해야 한다.

---

## Completion Definition

이 milestone은 아래가 모두 만족되면 완료로 본다.

- HyperINF solver가 toy exact problem에서 oracle과 비교 가능
- influence wrapper가 selector와 결합 가능
- search script에서 `hyperinf`가 method option으로 실행 가능
- 최소 benchmark smoke run이 통과
- 각 milestone 단계별 테스트가 모두 추가되고 통과
