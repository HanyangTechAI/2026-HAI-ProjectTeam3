# 2026-HAI-ProjectTeam3

GSM8K 수학 문장제에서 프롬프트 조합을 행동 공간으로 두고, 작은 정책 네트워크가 문제 특성에 따라 프롬프트를 선택하도록 학습하는 실험 코드베이스다. 정책은 강화학습 방식으로 업데이트되며, 응답 정확도와 토큰 사용량을 함께 보상에 반영한다.

## 프로젝트 개요

이 프로젝트는 다음 흐름으로 동작한다.

1. 질문 길이, 숫자 개수, 퍼센트 포함 여부 같은 간단한 feature를 추출한다.
2. 정책 네트워크가 프롬프트 action 하나를 선택한다.
3. 선택된 action으로 실제 프롬프트를 구성해 LLM에 질의한다.
4. 정답 여부와 토큰 수를 바탕으로 reward를 계산한다.
5. 그 reward로 정책을 업데이트한다.

프롬프트 action은 아래 요소의 조합으로 구성된다.

- instruction 문구
- reasoning 유도 문구
- output format 제약
- self-check 문구

현재 action space는 총 36개다.

## 디렉터리 구조

```text
.
├─ configs.py
├─ run_train.py
├─ run_eval.py
├─ run_baselines.py
├─ run_exhaustive.py
├─ run_analysis.py
├─ src/
│  ├─ analysis.py
│  ├─ baselines.py
│  ├─ data.py
│  ├─ evaluator.py
│  ├─ llm_client.py
│  ├─ policy.py
│  ├─ prompt_space.py
│  ├─ reward.py
│  ├─ trainer.py
│  └─ utils.py
└─ outputs/
```

## 요구 사항

- Python 3.10+
- PyTorch
- Hugging Face `datasets`
- OpenAI Python SDK

패키지 설치:

```bash
pip install -r requirements.txt
```

또는 conda 환경을 사용할 경우:

```bash
conda env create -f environment.yml
conda activate <env-name>
```

## OpenAI API 설정

실제 모델을 사용할 경우 `OPENAI_API_KEY`가 필요하다.

PowerShell:

```powershell
$env:OPENAI_API_KEY="your_api_key"
```

mock 모드로 빠르게 구조만 테스트하려면 [configs.py](/c:/Projects/2026-HAI-ProjectTeam3/configs.py)에서 `api_mode="mock"`으로 바꾸면 된다.

## 주요 설정

기본 실험 설정은 [configs.py](/c:/Projects/2026-HAI-ProjectTeam3/configs.py)에 있다.

- dataset: `gsm8k`, config `main`
- train samples: `30`
- test samples: `20`
- exhaustive samples: `8`
- model: `gpt-4.1-mini`
- epochs: `3`
- reward: 정답 보상 + 토큰 패널티

실험 규모를 키우거나 비용을 줄이려면 `train_samples`, `test_samples`, `epochs`, `model_name`, `api_mode`를 먼저 조정하면 된다.

## 실행 방법

### 1. 학습

```bash
python run_train.py
```

학습이 끝나면 아래 파일들이 `outputs/`에 저장된다.

- `prompt_policy.pt`
- `train_history.json`
- `train_history.csv`

### 2. 저장된 정책 평가

```bash
python run_eval.py
```

저장된 정책 파일 `outputs/prompt_policy.pt`를 불러와 테스트 셋에서 평가하고, 결과를 `outputs/rl_policy.json`에 저장한다.

### 3. exhaustive search

```bash
python run_exhaustive.py
```

테스트 샘플 일부에 대해 모든 action을 전부 평가해서 다음 정보를 저장한다.

- oracle 성능
- 전역 best action
- action별 평균 reward / accuracy

결과 파일:

- `outputs/exhaustive.json`

### 4. 분석 플롯 생성

```bash
python run_analysis.py
```

가능한 경우 아래 그림 파일들을 `outputs/`에 생성한다.

- `train_curves.png`
- `action_hist.png`
- `baseline_accuracy.png`
- `exhaustive_action_scores.png`

## 코드 구성

- [src/prompt_space.py](/c:/Projects/2026-HAI-ProjectTeam3/src/prompt_space.py): 프롬프트 action 정의 및 렌더링
- [src/policy.py](/c:/Projects/2026-HAI-ProjectTeam3/src/policy.py): 프롬프트 선택 정책 네트워크
- [src/trainer.py](/c:/Projects/2026-HAI-ProjectTeam3/src/trainer.py): REINFORCE 기반 학습 및 평가 루프
- [src/reward.py](/c:/Projects/2026-HAI-ProjectTeam3/src/reward.py): 정답 추출, 정오 판정, reward 계산
- [src/llm_client.py](/c:/Projects/2026-HAI-ProjectTeam3/src/llm_client.py): OpenAI / mock LLM 클라이언트
- [src/baselines.py](/c:/Projects/2026-HAI-ProjectTeam3/src/baselines.py): random, fixed-action, RL policy, exhaustive 유틸리티
- [src/analysis.py](/c:/Projects/2026-HAI-ProjectTeam3/src/analysis.py): 결과 시각화

## 출력물 설명

- `train_history.json`, `train_history.csv`: epoch별 학습/평가 로그
- `prompt_policy.pt`: 학습된 정책 가중치
- `rl_policy.json`: 저장된 RL 정책 평가 결과
- `exhaustive.json`: exhaustive 탐색 결과
- `*.png`: 분석용 시각화 결과

## 현재 상태 메모

- [run_train.py](/c:/Projects/2026-HAI-ProjectTeam3/run_train.py)는 현재 정상적인 학습 엔트리포인트다.
- [run_eval.py](/c:/Projects/2026-HAI-ProjectTeam3/run_eval.py)는 저장된 RL 정책 평가 스크립트로 사용된다.
- [run_baselines.py](/c:/Projects/2026-HAI-ProjectTeam3/run_baselines.py)는 현재 이름과 달리 RL 정책 평가 중심으로 구성되어 있다. `src/baselines.py`에는 random / fixed-action baseline 함수가 이미 있으므로, 필요하면 이 엔트리포인트를 확장해 leaderboard 생성 흐름으로 정리할 수 있다.

## 주의 사항

- `datasets`가 처음 실행될 때 GSM8K 다운로드가 필요하다.
- OpenAI API를 사용할 경우 호출 비용이 발생한다.
- 답안 추출은 `FINAL: <number>` 형식을 우선 사용하므로, 프롬프트 포맷 제약이 성능에 직접 영향을 준다.
- 현재 feature는 매우 단순하므로 정책 성능은 샘플 수와 프롬프트 설계에 민감하다.
