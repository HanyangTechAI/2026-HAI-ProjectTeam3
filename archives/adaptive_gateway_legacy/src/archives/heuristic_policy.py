from src.controller.action_space import InferenceAction, InferenceActionSpace
from src.controller.state_encoder import summarize_query_state


class HeuristicInferencePolicy:
    def __init__(self, action_space: InferenceActionSpace | None = None):
        self.action_space = action_space or InferenceActionSpace()

    def choose_action(self, question: str) -> int:
        feats = summarize_query_state(question)

        normalized_length = feats["normalized_length"]
        normalized_word_count = feats["normalized_word_count"]
        normalized_digit_count = feats["normalized_digit_count"]
        has_percent = feats["has_percent"]
        has_money = feats["has_money"]
        has_ratio_words = feats["has_ratio_words"]
        has_multistep_hint = feats["has_multistep_hint"]

        # 1) 쉬운 산술/직접 계산 문제
        # 짧고 숫자 적고 관계 단순하면 small + none + no verify
        if (
            normalized_word_count < 0.18
            and normalized_digit_count <= 0.15
            and has_percent == 0.0
            and has_ratio_words == 0.0
            and has_multistep_hint == 0.0
        ):
            return self.action_space.action_to_index(
                InferenceAction(
                    reasoning_budget="none",
                    model_route="small",
                    verify=False,
                )
            )

        # 2) 비율/퍼센트/관계 문제는 더 강하게
        if has_percent == 1.0 or has_ratio_words == 1.0:
            return self.action_space.action_to_index(
                InferenceAction(
                    reasoning_budget="short",
                    model_route="large",
                    verify=True,
                )
            )

        # 3) 다단계 힌트가 강한 문제
        if has_multistep_hint == 1.0 and normalized_digit_count >= 0.10:
            return self.action_space.action_to_index(
                InferenceAction(
                    reasoning_budget="short",
                    model_route="large",
                    verify=True,
                )
            )

        # 4) 금액이 나오고 길이가 긴 문제
        if has_money == 1.0 and normalized_length >= 0.20:
            return self.action_space.action_to_index(
                InferenceAction(
                    reasoning_budget="short",
                    model_route="large",
                    verify=True,
                )
            )

        # 5) 그 외 기본값: 현재 best fixed
        return self.action_space.action_to_index(
            InferenceAction(
                reasoning_budget="short",
                model_route="large",
                verify=True,
            )
        )