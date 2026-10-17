from .base import ExitRules, Sizing, Strategy
from .box import BoxStrategy
from .orb import ORBStrategy
from .vcp import VCPStrategy

REGISTRY: dict[str, type[Strategy]] = {
    "vcp": VCPStrategy,
    "orb": ORBStrategy,
    "box": BoxStrategy,
}

__all__ = ["REGISTRY", "Strategy", "ExitRules", "Sizing", "VCPStrategy", "ORBStrategy", "BoxStrategy"]
