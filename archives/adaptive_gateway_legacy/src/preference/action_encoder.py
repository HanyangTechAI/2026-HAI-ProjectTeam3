from src.controller.action_space import InferenceActionSpace


def encode_action_features(action_idx: int, action_space: InferenceActionSpace):
    action = action_space.get_action(action_idx)

    reasoning_onehot = [
        1.0 if action.reasoning_budget == "none" else 0.0,
        1.0 if action.reasoning_budget == "short" else 0.0,
        1.0 if action.reasoning_budget == "long" else 0.0,
    ]

    model_onehot = [
        1.0 if action.model_route == "small" else 0.0,
        1.0 if action.model_route == "large" else 0.0,
    ]

    verify_feat = [1.0 if action.verify else 0.0]

    return reasoning_onehot + model_onehot + verify_feat