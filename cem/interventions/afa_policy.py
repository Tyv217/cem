import numpy as np
import torch
import logging
import gymnasium as gym

from gymnasium import spaces

class AFAEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 4}

    def __init__(self, cbm, n_concepts, n_tasks, seed, use_groups, concept_group_map = None):

        self.cbm = cbm
        self.n_concept = len(concept_group_map) if use_groups else n_concepts
        self.n_tasks = n_tasks
        self.seed = seed

        self.observation_space = spaces.Dict(
            {
                "intervened_concepts": spaces.MultiBinary(self.n_concepts, seed=self.seed),
                "flow_model_output": spaces.Box(shape = [self.n_concepts], dtype = np.float32, seed=self.seed),
                "cbm_model_output": spaces.Box(shape = [self.n_concepts], dtype = np.float32, seed=self.seed)
            }
        )

        self.action_space = spaces.Discrete(self.n_concepts)

    def _get_obs(self):
        return {
            "intervened_concepts": self.intervened_concepts,
            "flow_model_output": self._target_location
        }