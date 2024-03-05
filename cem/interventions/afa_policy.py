import numpy as np
import torch
import logging
import gymnasium as gym

from gymnasium import spaces

class AFAEnv(gym.Env):

    def __init__(self, env_config):
        
        self.cbm = env_config["cbm"]
        self.ac_model = env_config["ac_model"]
        self.use_concept_groups = env_config["use_concept_groups"]
        self.concept_group_map = env_config["concept_group_map"]
        self.n_concept_groups = len(self.concept_group_map) if self.use_concept_groups else env_config["n_concepts"]
        self.n_concepts = env_config["n_concepts"]
        self.n_tasks = env_config["n_tasks"]
        self.n_tasks = self.n_tasks if self.n_tasks > 1 else 2
        self.emb_size = env_config["emb_size"]
        self.cbm_dl = env_config["cbm_dl"]
        self.torch_generator = torch.Generator()
        self.torch_generator.manual_seed(env_config["seed"])

        self.observation_space = spaces.Dict(
            {
                "intervened_concepts_map": spaces.MultiBinary(self.n_concept_groups),
                "intervened_concepts": spaces.MultiBinary(self.n_concepts),
                "ac_model_output": spaces.Box(shape = [self.n_concept_groups * self.n_tasks], dtype = np.float32),
                "cbm_bottleneck": spaces.Box(shape = [self.n_concepts * self.emb_size], dtype = np.float32),
                "cbm_pred_concepts": spaces.Box(shape = [self.n_concepts], dtype = np.float32),
                "cbm_pred_output": spaces.Discrete(self.n_tasks)
            }
        )

        self.action_space = spaces.Discrete(self.n_concepts)

    def _get_obs(self):
        return {
            "budget": self._budget,
            "intervened_concepts_map": self._intervened_concepts_map,
            "intervened_concepts": self._intervened_concepts
            "ac_model_output": self._ac_model_output,
            "cbm_bottleneck": self._cbm_bottleneck,
            "cbm_pred_concepts": self._cbm_pred_concepts,
            "cbm_pred_output": self._cbm_pred_output,
        }

    def _get_info(self):
        return {
            "intervened_concepts_map": self._intervened_concepts_map,
        }

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if options is not None:
            budget = options.get("budget", None)
            cbm_data = options.get("cbm_data", None)
            train = options.get("train", None)

        if budget is not None:
            self._budget = self.np_random.integers(low = 0, high = self.n_concepts + 1)
        else:
            self._budget = budget
        
        if options is not None:
        else:
            cbm_data = torch.utils.data.RandomSampler(cbm_data.dataset, num_samples = 1, generator = self.torch_generator)
        x, y, (c, competencies, _) = self.cbm_model._unpack_batch(cbm_data)
        outputs = self.cbm_model.forward(
            x,
            intervention_idxs=intervention_idxs,
            c=c,
            y=y,
            train=train,
            competencies=competencies,
            prev_interventions=torch.unsqueeze(torch.IntTensor(self._intervened_concept), dim = 0),
        )
        c_sem, c_logits, y_logits = outputs[0], outputs[1], outputs[2]
        latent = outputs[4]
        pos_embeddings = outputs[-2]
        neg_embeddings = outputs[-1]
        prob = prev_interventions * c_sem + (1 - prev_interventions) * prob
        embeddings = (
            torch.unsqueeze(prob, dim=-1) * pos_embeddings +
            (1 - torch.unsqueeze(prob, dim=-1)) * neg_embeddings
        )
        self._intervened_concepts_map = np.zeros(self.n_concept_groups, dtype = int)
        self._intervened_concepts = np.zeros(self.n_concepts, dtype = int)
        self._ac_model_output = torch.squeeze(
            self.ac_model(
                    b = torch.unsqueeze(torch.tensor(self._intervened_concepts_map.to(self.ac_model.device)))
                )
            
        .detach()).cpu().numpy()
        self._cbm_model_output = self.cbm_model()

    def step()