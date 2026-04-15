"""Gymnasium registration for all CausalShift environments."""

import gymnasium as gym


def register_all() -> None:
    """Register all CausalShift environments with Gymnasium.

    After calling this, environments can be created via:
        gym.make("CausalShiftXOR-v0", theta=0.2)
        gym.make("CausalShiftChain-v0")
        gym.make("CausalShiftBranch-v0", mu=0.3)
    """
    envs_to_register = [
        {
            "id": "CausalShiftXOR-v0",
            "entry_point": "causalshift.envs.xor:CausalShiftXOR",
            "kwargs": {"theta": 0.0, "horizon": 200},
        },
        {
            "id": "CausalShiftChain-v0",
            "entry_point": "causalshift.envs.chain:CausalShiftChain",
            "kwargs": {"horizon": 200},
        },
        {
            "id": "CausalShiftBranch-v0",
            "entry_point": "causalshift.envs.branch:CausalShiftBranch",
            "kwargs": {"mu": 0.0, "horizon": 200},
        },
    ]

    for env_spec in envs_to_register:
        if env_spec["id"] not in gym.envs.registry:
            gym.register(**env_spec)
