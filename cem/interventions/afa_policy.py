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
        self.cbm_forward = env_config["cbm_forward"]
        self.unpack_batch = env_config["unpack_batch"]
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
            "intervened_concepts": self._intervened_concepts,
            "ac_model_output": self._ac_model_output,
            "cbm_bottleneck": self._cbm_bottleneck,
            "cbm_pred_concepts": self._cbm_pred_concepts,
            "cbm_pred_output": self._cbm_pred_output,
        }

    def _get_info(self):
        return {
            "action_mask": 1 - self._intervened_concepts_map,
        }

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self._budget = options.get("budget", None) or \
            self.np_random.integers(low = 0, high = self.n_concepts + 1)
        self._cbm_data = options.get("cbm_data", None) or \
            torch.utils.data.RandomSampler(self.cbm_dl.dataset, num_samples = 1, generator = self.torch_generator)
        self._train = options.get("train", None) or True

        prev_interventions = torch.unsqueeze(torch.IntTensor(self._intervened_concepts), dim = 0)
        x, y, (c, competencies, _) = self.unpack_batch(self._cbm_data)
        with torch.no_grad():
            outputs = self.cbm_forward(
                x,
                intervention_idxs=np.zeros(self.n_concepts, dtype = int),
                c=c,
                y=y,
                train=self._train,
                competencies=competencies,
                prev_interventions=torch.unsqueeze(torch.IntTensor(self._intervened_concepts), dim = 0),
            )
            c_sem, c_pred, y_logits = outputs[0], outputs[1], outputs[2]
            pos_embeddings = outputs[-2]
            neg_embeddings = outputs[-1]
            prob = prev_interventions * c_sem + (1 - prev_interventions) * prob
            embeddings = (
                torch.unsqueeze(prob, dim=-1) * pos_embeddings.detach() +
                (1 - torch.unsqueeze(prob, dim=-1)) * neg_embeddings.detach()
            ).cpu().numpy()
            ac_model_output = torch.squeeze(
                self.ac_model(
                        b = torch.unsqueeze(torch.tensor(self._intervened_concepts_map.to(self.ac_model.device)))
                    )
            .detach()).cpu().numpy()
        self._intervened_concepts_map = np.zeros(self.n_concept_groups, dtype = int)
        self._intervened_concepts = np.zeros(self.n_concepts, dtype = int)
        self._ac_model_output = ac_model_output
        self._cbm_bottleneck = embeddings
        self._cbm_pred_concepts = c_pred.detach().cpu().numpy()
        self._cbm_pred_output = torch.argmax(y_logits.detach(), dim = 1).cpu().numpy()

        obs = self._get_obs()
        info = self._get_info()

        return obs, info

    def step(self, action):

        prev_interventions = torch.unsqueeze(torch.IntTensor(self._intervened_concepts), dim = 0)
        new_interventions_map = self._intervened_concepts_map
        new_interventions = prev_interventions.copy()
        new_interventions_map[0, action] = 1
        concept_group = sorted(list(self.concept_group_map.keys()))[action]
        for concept in concept_group:
            new_interventions[0, concept] = 1
        x, y, (c, competencies, _) = self.unpack_batch(self._cbm_data)
        
        with torch.no_grad():
            outputs = self.cbm_forward(
                x,
                intervention_idxs=new_interventions,
                c=c,
                y=y,
                train=self._train,
                competencies=competencies,
                prev_interventions=prev_interventions,
            )
            c_sem, c_pred, y_logits = outputs[0], outputs[1], outputs[2]
            pos_embeddings = outputs[-2]
            neg_embeddings = outputs[-1]
            prob = prev_interventions * c_sem + (1 - prev_interventions) * prob
            embeddings = (
                torch.unsqueeze(prob, dim=-1) * pos_embeddings.detach() +
                (1 - torch.unsqueeze(prob, dim=-1)) * neg_embeddings.detach()
            ).cpu().numpy()
            ac_model_output = torch.squeeze(
                self.ac_model(
                        b = torch.unsqueeze(new_interventions.to(self.ac_model.device), dim = 0)
                    )
            .detach(), dim = 0).cpu().numpy()
        self._ac_model_output = ac_model_output
        self._intervened_concepts_map = new_interventions_map
        self._intervened_concepts = new_interventions.copy().detach().squeeze().cpu().numpy()
        self._cbm_bottleneck = embeddings
        self._cbm_pred_concepts = c_pred.detach().cpu().numpy()
        self._cbm_pred_output = torch.argmax(y_logits.detach(), dim = 1).cpu().numpy()

        obs = self._get_obs()
        info = self._get_info()

        return obs, info

