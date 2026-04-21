# Generalized Influence

`GeneralizedInfluence`는 선택된 parameter subset `J` 위에서 restricted update를 계산한다.

핵심 단계:
- target gradient 계산
- subset projection
- restricted system 구성
- `p_lissa`로 update 근사

구현 위치:
- method: `src/gif/influence/generalized.py`
- projection: `src/gif/influence/projection.py`
- solver: `src/gif/solvers/iterative.py`
