# Solver Evaluation

실험 2의 목표는 `P-LiSSA`가 실제로 `LiSSA`, `CG`, `Schulz`, `Lanczos`
기반 baseline보다 더 안정적인지를 같은 restricted operator 위에서 비교하는 것이다.

비교 대상은 다음 네 가지로 고정한다.

1. `LiSSA + damping`
2. `Conjugate Gradient`
3. `Schulz iteration` (`HyperINF` 구현 재사용)
4. `Lanczos inverse approximation`

공정 비교 원칙:

- 모두 같은 restricted system을 푼다.
- 공통 연산자는 `A(x) = H_J^T H_J x` 를 사용한다.
- 공통 우변은 `rhs = H_J^T g` 를 사용한다.
- 모든 solver는 `a_times(x), rhs` 인터페이스를 우선 지원한다.
- 모든 solver는 `return_details=True`일 때 residual history와 핵심 하이퍼파라미터를 남긴다.
- script 실험에서는 동일한 selector / scaling / normalization 규칙을 쓴다.

## Current Status

- `P-LiSSA` 구현 완료
- `Schulz iteration` 구현 완료 (`HyperINF`)
- `LiSSA + damping` 미구현
- `CG` 미구현
- `Lanczos inverse approximation` 미구현

## Implementation Checklist

### Common

- [ ] restricted linear-system builder를 공용 함수로 분리
- [ ] solver 공통 세부 지표 포맷 정리
- [ ] `src/gif/solvers/__init__.py` export 정리
- [ ] `src/gif/influence/__init__.py` export 정리
- [ ] package root export 정리

### 1. LiSSA + Damping

- [ ] `lissa_inverse(a_times, rhs, damping, mu, tol, max_iter, ...)` 추가
- [ ] plain LiSSA와 damped LiSSA를 같은 구현에서 지원
- [ ] restricted wrapper `lissa_update(...)` 추가
- [ ] class API 추가

### 2. Conjugate Gradient

- [ ] `cg_inverse(a_times, rhs, damping, tol, max_iter, ...)` 추가
- [ ] SPD 가정 명시
- [ ] `(A + damping I)` 형태 지원
- [ ] restricted wrapper `cg_update(...)` 추가
- [ ] class API 추가

### 3. Schulz Iteration

- [ ] 기존 `hyperinf_inverse(...)`를 Schulz baseline으로 유지
- [ ] 세부 지표 포맷을 신규 solver와 맞춤
- [ ] 필요 시 `schulz` alias 추가 여부 결정
- [ ] script 실험 이름은 일단 `hyperinf` 유지

### 4. Lanczos

- [ ] `lanczos_inverse(a_times, rhs, rank, damping, ...)` 추가
- [ ] symmetric PSD operator 기준 설계
- [ ] top-k eigenspace inverse approximation 사용
- [ ] rank가 충분히 크면 dense exact fallback 허용
- [ ] restricted wrapper `lanczos_update(...)` 추가
- [ ] class API 추가

## Test Checklist

### Solver Unit Tests

- [ ] dense SPD toy matrix에서 oracle 해와 비교
- [ ] iteration 증가 시 residual 감소 확인
- [ ] ill-conditioned system에서도 finite output 확인
- [ ] `damping > 0`에서 안정화 동작 확인
- [ ] zero rhs 처리 확인

### Influence Wrapper Tests

- [ ] restricted operator에서 dense oracle과 비교
- [ ] subset 차원과 출력 shape 일치 확인
- [ ] finite update 확인
- [ ] 작은 step으로 target loss 방향성이 맞는지 확인

### Integration Tests

- [ ] CLI `--help`에 scheme 노출 확인
- [ ] toy smoke run에서 normalized update finite 확인
- [ ] 기존 `gif / hyperinf / datainf / tracin` 회귀 없음

## Suggested Test Files

- `tests/solvers/test_lissa_solver.py`
- `tests/solvers/test_cg_solver.py`
- `tests/solvers/test_hyperinf_solver.py`
- `tests/solvers/test_lanczos_solver.py`
- `tests/influence/test_lissa_influence.py`
- `tests/influence/test_cg_influence.py`
- `tests/influence/test_hyperinf_influence.py`
- `tests/influence/test_lanczos_influence.py`
- `tests/integration/test_solver_scheme_cli.py`

## Git Milestones

### Milestone 1

`docs: expand solver_eval plan for LiSSA, CG, Schulz, and Lanczos`

- 실험 범위 명시
- 체크리스트 작성
- 테스트 계획 작성
- 커밋 순서 고정

### Milestone 2

`refactor: extract shared restricted-system builder`

- 공용 `rhs, a_times` builder 추가
- 기존 HyperINF wrapper 마이그레이션
- 공용 builder 사용 테스트 안정화

### Milestone 3

`feat: add Lanczos inverse-approximation baseline`

- solver 구현
- influence wrapper 구현
- solver / influence unit test 추가

### Milestone 4

`feat: add damped LiSSA baseline`

- solver 구현
- wrapper 구현
- dense oracle 및 residual 테스트 추가

### Milestone 5

`feat: add conjugate-gradient baseline`

- solver 구현
- wrapper 구현
- damping 지원
- dense oracle 테스트 추가

### Milestone 6

`refactor: normalize Schulz baseline details and expose common comparison metrics`

- HyperINF details 포맷 정리
- 공통 residual metric 맞춤
- 비교 로그 일관화

### Milestone 7

`feat: wire solver baselines into search scripts`

- MNIST search script scheme 추가
- help text 업데이트
- integration test 추가

### Milestone 8

`test: add experimental smoke coverage for solver stability baselines`

- toy / smoke 설정 추가
- finite normalized update 확인
- 기존 baseline 회귀 확인

## Done Definition

- 네 baseline이 동일한 restricted operator를 푼다.
- 모든 baseline이 공통 details를 반환한다.
- unit test와 influence test가 각각 있다.
- script layer에서 동일한 방식으로 호출 가능하다.
- solver 실험 로그에 하이퍼파라미터와 residual 정보가 남는다.
