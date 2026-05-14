from dataclasses import dataclass


@dataclass
class ModelRouteConfig:
    model_name: str
    estimated_cost_multiplier: float


def get_model_route_config(model_route: str) -> ModelRouteConfig:
    """
    model_route:
        - "small"
        - "large"
    """

    if model_route == "small":
        return ModelRouteConfig(
            model_name="gpt-4.1-mini",
            estimated_cost_multiplier=1.0,
        )

    if model_route == "large":
        return ModelRouteConfig(
            model_name="gpt-4.1",
            estimated_cost_multiplier=2.0,
        )

    raise ValueError(f"Unknown model_route: {model_route}")