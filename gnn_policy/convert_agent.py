import sys
from pathlib import Path

import torch
from stable_baselines3 import A2C


def convert(agent_file: Path):
    a = A2C.load(agent_file)
    node_dim = int(a.observation_space["nodes"].shape[-1])
    state_dict = a.policy.state_dict()
    to_save = {}
    config = {
        "gnn_steps": a.policy_kwargs["gnn_steps"],
        "embedding_dim": a.policy_kwargs["features_extractor_kwargs"]["features_dim"],
        "separate_actor_critic": a.policy_kwargs["separate_actor_critic"],
        "action_mode": a.policy_kwargs["action_mode"],
        "node_feature_dim": node_dim,
        "num_actions": int(a.action_space.nvec[0]),
        "gnn_class": a.policy_kwargs["gnn_class"].__name__,
    }
    to_save["config"] = config
    to_save["state_dict"] = state_dict
    # with open("gnn_action_then_node_demo1_params.pth", "w") as f:
    # json.dump(to_save, f)
    stem = agent_file.stem
    torch.save(to_save, f"{stem}.pth")


agent_file = sys.argv[1]
convert(Path(agent_file))
