# Architecture Overview

이 레포는 `influence-function-based model editing`을 위한 코드 구조를 기준으로 정리한다.

## Source Layout

```text
src/gif/
  influence/
  selection/
  solvers/
  models/
  data/
```

## Responsibilities

- `influence/`: GIF, classical IF, second-order IF, freezing IF 같은 method 구현
- `selection/`: parameter subset 선택 로직
- `solvers/`: `p_lissa`, `ihvp` 같은 반복 해법
- `models/`: 네트워크 구조
- `data/`: dataset loader와 입력 파이프라인

## Dependency Direction

- influence methods can depend on `selection/` and `solvers/`
- solvers do not depend on high-level methods
- scripts call packaged APIs from `src/gif`
