# RL 기반 LLM 추론 정책 최적화

GSM8K 수학 문제를 대상으로 LLM 추론 전략을 동적으로 선택하는 실험 프로젝트입니다.
정책은 문제 상태를 입력으로 받아 추론 길이, 모델 라우팅, 검증 사용 여부로 구성된 action을 선택합니다.
선택된 action으로 LLM을 호출한 뒤 정답 정확도와 토큰/모델 비용을 반영한 reward로 정책을 개선합니다.

## 핵심 아이디어

- State: 문제 길이, 숫자 개수, 비율/금액/다단계 힌트, 문장 embedding
- Action: `reasoning_budget` x `model_route` x `verify`
- Reward: 정답 보상 - 토큰 비용 - 모델 라우팅 비용 - 검증/포맷 패널티
- Policy: action preference model을 초기 정책으로 사용하고, rollout reward로 RL fine-tuning

현재 action space는 8개입니다.

```text
0: none  / small / no-verify
1: none  / small / verify
2: short / small / no-verify
3: short / small / verify
4: short / large / no-verify
5: short / large / verify
6: long  / small / no-verify
7: long  / large / no-verify
```

## 빠른 시연

외부 API나 데이터 다운로드 없이 실행되는 mock 데모입니다.

```powershell
python run_demo.py --mode batch
```

단일 문제 시연:

```powershell
python run_demo.py --mode single
```

직접 문제를 넣는 경우:

```powershell
python run_demo.py --mode single --question "A box has 6 rows of pencils with 4 pencils in each row. How many pencils are in the box?" --gold 24
```

결과 JSON 저장:

```powershell
python run_demo.py --mode batch --save_json outputs/demo_policy_run.json --save_html outputs/demo_report.html
```

생성된 `outputs/demo_report.html`을 브라우저에서 열면 RL 정책과 고정 action baseline 비교,
문제별 선택 action, reward, token 사용량을 한 페이지에서 확인할 수 있습니다.

## 실제 OpenAI 호출 평가

`OPENAI_API_KEY`가 필요합니다.

```powershell
$env:OPENAI_API_KEY="your_api_key"
python run_demo.py --mode batch --dataset gsm8k --api_mode openai --embedding_model sentence-transformers/all-MiniLM-L6-v2 --num_samples 10
```

## 주요 파일

- `run_demo.py`: 발표/시연용 진입점
- `run_rl_training.py`: reward rollout 기반 정책 fine-tuning
- `run_find_best_checkpoint.py`: checkpoint 성능 비교
- `run_preference_controller_holdout.py`: preference policy holdout 평가
- `src/controller/action_space.py`: 추론 action 정의
- `src/controller/runtime_controller.py`: action 실행 및 reward 계산
- `src/controller/state_encoder.py`: 문제 상태 feature/embedding 생성
- `src/preference/preference_model.py`: state-action preference scorer
- `src/rewards/heuristic_reward.py`: 정확도/비용 기반 reward
- `src/llm_client.py`: OpenAI client와 offline mock client

## 설치

```powershell
pip install -r requirements.txt
```

mock 데모만 실행할 때는 이미 설치된 PyTorch만 있으면 동작합니다. GSM8K와 real embedding을 쓰려면
`datasets`, `sentence-transformers`, `openai`가 필요합니다.

## 현재 결과 예시

`outputs/checkpoint_comparison_all_pt.json` 기준 best checkpoint:

- `outputs/action_preference_model_hard_train_0_49.pt`
- test 30 samples accuracy: `0.60`
- average reward: `0.526`

`outputs/rl_final_summary.json` 기준 RL fine-tuning 결과:

- test 30 samples accuracy: `0.60`
- average reward: `0.531`

## 개발 방향

1. mock 데모로 action 선택과 reward 구조를 안정적으로 보여준다.
2. OpenAI/GSM8K 실험으로 실제 추론 정책 성능을 측정한다.
3. reward 설계를 고도화해 비용 대비 정확도 trade-off를 더 명확히 만든다.
4. checkpoint 비교와 batch eval 결과를 발표용 표/그래프로 정리한다.
