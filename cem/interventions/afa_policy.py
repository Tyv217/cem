import numpy as np
import torch
import logging
import gymnasium as gym

from gymnasium import spaces

class AFAEnv(gym.Env):

    def __init__(self, cbm, n_concepts, n_tasks, seed, use_concept_groups, concept_group_map = None, emb_size = 1):

        self.cbm = cbm
        self.n_concept = len(concept_group_map) if use_groups else n_concepts
        self.n_tasks = n_tasks
        self.seed = seed
        self.use_concept_groups = use_concept_groups
        self.concept_group_map = concept_group_map
        self.emb_size = 1

        self.observation_space = spaces.Dict(
            {
                "intervened_concepts": spaces.MultiBinary(self.n_concepts, seed=self.seed),
                "ac_model_output": spaces.Box(shape = [self.n_concepts], dtype = np.float32, seed=self.seed),
                "cbm_bottleneck": spaces.Box(shape = [self.n_concepts * self.emb_size], dtype = np.float32, seed=self.seed),
                "cbm_pred_concepts": spaces.Box(shape = [self.n_concepts * self.emb_size], dtype = np.float32, seed=self.seed),
                "cbm_pred_output": spaces.Discrete(shape = [self.n_tasks], seed=self.seed),
            }
        )

        self.action_space = spaces.Discrete(self.n_concepts)

    def _get_obs(self):
        return {
            "intervened_concepts": self.intervened_concepts,
            "ac_model_output": self.ac_model_output,
            "cbm_bottleneck": self.cbm_bottleneck,
            "cbm_pred_concepts": self.cbm_pred_concepts,
            "cbm_pred_output": self.cbm_pred_output,
        }

    def _get_info(self):
        return {
            "intervened_concepts": self.intervened_concepts
        }