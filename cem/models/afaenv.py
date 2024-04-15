import numpy as np
import torch
import logging
import gymnasium as gym
import scipy

from gymnasium import spaces

class AFAEnv(gym.Env):

    def __init__(self, cbm, ac_model, env_config):
        
        # self.cbm = env_config["cbm"]
        # self.ac_model = env_config["ac_model"]
        self.cbm = cbm
        self.ac_model = ac_model
        self.use_concept_groups = env_config["use_concept_groups"]
        self.concept_group_map = env_config["concept_map"]
        self.n_concept_groups = len(self.concept_group_map) if self.use_concept_groups else env_config["n_concepts"]
        self.n_concepts = env_config["n_concepts"]
        self.n_tasks = env_config["n_tasks"]
        self.n_tasks = self.n_tasks if self.n_tasks > 1 else 2
        self.emb_size = env_config["emb_size"]
        afa_config = env_config["afa_model_config"]
        self.step_cost = afa_config.get("step_cost", 1)
        self.cbm_dl = afa_config.get("cbm_dl", None)
        self.cbm_forward = cbm.forward
        self.unpack_batch = cbm._unpack_batch
        self.torch_generator = torch.Generator()
        self.torch_generator.manual_seed(afa_config["seed"])
        self._budget = self.n_concept_groups
        self.softmax = torch.nn.Softmax(dim = -1)
        self.entropy = scipy.stats.entropy
        self.intermediate_reward_ratio = 0.1
        self.xent_loss = torch.nn.CrossEntropyLoss()

        self.observation_space = spaces.Dict(
            {
                "intervened_concepts_map": spaces.MultiBinary(self.n_concepts),
                "intervened_concepts": spaces.MultiBinary(self.n_concepts),
                "ac_model_output": spaces.Box(shape = [self.n_concepts * self.n_tasks * 2], dtype = np.float32),
                "ac_model_output": spaces.Box(shape = [self.n_concepts * self.n_tasks * 3], dtype = np.float32),
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
            "ac_model_info": self._ac_model_info,
            "cbm_bottleneck": self._cbm_bottleneck,
            "cbm_pred_concepts": self._cbm_pred_concepts,
            "cbm_pred_output": self._cbm_pred_output,
        }

    def _get_info(self):
        return {
            "action_mask": 1 - self._intervened_concepts_map,
        }

    def reset(self, seed=None, options=None):
        import pdb
        pdb.set_trace()
        super().reset(seed=seed)
        
        self._budget = options.get("budget", None) or \
            self.np_random.integers(low = 0, high = self.n_concept_groups + 1)
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
            interventions = torch.zeros(self._intervened_concepts.shape).to(self.ac_model.device)
            xx = torch.cat((prob, prob), dim = 0)
            bb = torch.cat((interventions, interventions), dim = 0)
            ac_model_output = torch.squeeze(
                self.ac_model.compute_concept_probabilities(
                        x = xx,
                        b = bb,
                        m = bb,
                        y = None
                    )
            .detach(), dim = 0).cpu().numpy()
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

        import pdb
        pdb.set_trace()
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
            xx = torch.cat((prob, prob), dim = 0)
            bb = torch.cat((prev_interventions, torch.unsqueeze(new_interventions.to(self.ac_model.device), dim = 0)), dim = 0)
            ac_model_output = torch.squeeze(
                self.ac_model.compute_concept_probabilities(
                        x = xx,
                        b = bb,
                        m = bb,
                        y = None
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

        terminated = np.sum(self._intervened_concepts_map) == self._budget
        reward = self.calculate_reward(terminated, y_logits, y)

        return obs, reward, terminated, False, info
    
    def calculate_reward(self, terminated, predictions, y):
        if terminated:
            return -self.xent_loss(predictions, y)
        else:
            pre_prob, post_prob = np.split(self._ac_model_output, 2, axis = 0)
            return self.intermediate_reward_ratio * (self.entropy(pre_prob.T) - self.entropy(post_prob.T))

