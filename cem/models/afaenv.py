import numpy as np
import torch
import logging
import gymnasium as gym
import scipy

from gymnasium import spaces

class AFAEnv(gym.Env):

    def __init__(self, cbm, ac_model, env_config, batch_size):
        
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
        self.torch_generator = torch.Generator()
        self.torch_generator.manual_seed(afa_config["seed"])
        self._budget = self.n_concept_groups
        self.num_interventions = 0
        self.softmax = torch.nn.Softmax(dim = -1)
        self.entropy = scipy.stats.entropy
        self.intermediate_reward_ratio = 0.1
        self.xent_loss = torch.nn.CrossEntropyLoss()
        self.batch_size = batch_size

        self.observation_space_dict = {
                "budget": spaces.Box(shape = [1], dtype = np.int32, low = 1, high = self.n_concept_groups),
                "intervened_concepts_map": spaces.MultiBinary(self.n_concept_groups),
                "intervened_concepts": spaces.MultiBinary(self.n_concepts),
                "ac_model_output": spaces.Box(shape = [1], dtype = np.float32, low = -np.inf, high = np.inf),
                "ac_model_info": spaces.Box(shape = [self.n_tasks + self.n_concepts * 4], dtype = np.float32, low = -np.inf, high = np.inf),
                "cbm_bottleneck": spaces.Box(shape = [self.n_concepts * self.emb_size], dtype = np.float32, low = -np.inf, high = np.inf),
                "cbm_pred_concepts": spaces.Box(shape = [self.n_concepts], dtype = np.float32, low = -np.inf, high = np.inf),
                "cbm_pred_output": spaces.Box(shape = [self.n_tasks], dtype = np.float32, low = 0, high = 1),
            }

        total_shape = 0
        for key, value in self.observation_space_dict.items():
            shape = value.shape
            if len(shape) > 1:
                raise ValueError("Get shape unable to flatten 2d shape")
            elif len(shape) == 0:
                total_shape += 1
            else:
                total_shape += shape[0]
        
        self.observation_space = spaces.Box(shape = [self.batch_size * total_shape], dtype = np.float32, low = -np.inf, high = np.inf)
        self.single_observation_space = spaces.Box(shape = [total_shape], dtype = np.float32, low = -np.inf, high = np.inf)

        self.action_space = spaces.Box(shape = [self.batch_size], dtype = np.int32, low = 0, high = self.n_concept_groups)
        self.single_action_space_shape = self.n_concept_groups


# class AFAEnvOld(gym.Env):

#     def __init__(self, cbm, ac_model, env_config):
        
#         # self.cbm = env_config["cbm"]
#         # self.ac_model = env_config["ac_model"]
#         self.cbm = cbm
#         self.ac_model = ac_model
#         self.use_concept_groups = env_config["use_concept_groups"]
#         self.concept_group_map = env_config["concept_map"]
#         self.n_concept_groups = len(self.concept_group_map) if self.use_concept_groups else env_config["n_concepts"]
#         self.n_concepts = env_config["n_concepts"]
#         self.n_tasks = env_config["n_tasks"]
#         self.n_tasks = self.n_tasks if self.n_tasks > 1 else 2
#         self.emb_size = env_config["emb_size"]
#         afa_config = env_config["afa_model_config"]
#         self.step_cost = afa_config.get("step_cost", 1)
#         self.cbm_dl = afa_config.get("cbm_dl", None)
#         self.torch_generator = torch.Generator()
#         self.torch_generator.manual_seed(afa_config["seed"])
#         self._budget = self.n_concept_groups
#         self.num_interventions = 0
#         self.softmax = torch.nn.Softmax(dim = -1)
#         self.entropy = scipy.stats.entropy
#         self.intermediate_reward_ratio = 0.1
#         self.xent_loss = torch.nn.CrossEntropyLoss()

#         self.observation_space = spaces.Dict(
#             {
#                 "budget": spaces.Box(shape = [1], dtype = np.int32, low = 1, high = self.n_concept_groups),
#                 "intervened_concepts_map": spaces.MultiBinary(self.batch_size, self.n_concept_groups),
#                 "intervened_concepts": spaces.MultiBinary(self.n_concepts),
#                 "ac_model_output": spaces.Box(shape = [self.n_tasks], dtype = np.float32, low = -np.inf, high = np.inf),
#                 "ac_model_info": spaces.Box(shape = [self.n_tasks + self.n_concepts * 4], dtype = np.float32, low = -np.inf, high = np.inf),
#                 "cbm_bottleneck": spaces.Box(shape = [self.n_concepts * self.emb_size], dtype = np.float32, low = -np.inf, high = np.inf),
#                 "cbm_pred_concepts": spaces.Box(shape = [self.n_concepts], dtype = np.float32, low = -np.inf, high = np.inf),
#                 "cbm_pred_output": spaces.Box(shape = [2], dtype = np.float32, low = 0, high = 1),
#             }
#         )

#         self.action_space = spaces.Discrete(self.n_concept_groups)

#     def _get_obs(self):
#         return {
#             "budget": np.array([self._budget]),
#             "intervened_concepts_map": self._intervened_concepts_map,
#             "intervened_concepts": self._intervened_concepts,
#             "ac_model_output": self._ac_model_output,
#             "ac_model_info": self._ac_model_info,
#             "cbm_bottleneck": self._cbm_bottleneck,
#             "cbm_pred_concepts": self._cbm_pred_concepts,
#             "cbm_pred_output": self._cbm_pred_output,
#         }

#     def _get_info(self):
#         return {
#             "action_mask": 1 - self._intervened_concepts_map,
#         }

#     def reset(self, seed=None, options=None):
#         super().reset(seed=seed)
        
#         self.batch_size = options.get("batch_size", None) or 16
        
#         self._budget = options.get("budget", None) or \
#             self.np_random.integers(low = 0, high = self.n_concept_groups + 1)
#         self._cbm_data = options.get("cbm_data", None) or \
#             torch.utils.data.RandomSampler(self.cbm_dl.dataset, num_samples = 1, generator = self.torch_generator).to(cbm.device)

#         self._train = options.get("train", None) or True
        
#         self._intervened_concepts_map = np.zeros(self.n_concept_groups)
#         self._intervened_concepts = np.zeros(self.n_concepts)
        
#         prev_interventions = torch.tile(torch.unsqueeze(torch.IntTensor(self._intervened_concepts), dim = 0).to(self.cbm.device), (self.batch_size, 1))
#         x, y, (c, competencies, _) = self.cbm._unpack_batch(self._cbm_data)
#         if len(x.shape) == 3:
#             x = torch.unsqueeze(x, dim = 0)
#             y = torch.unsqueeze(y, dim = 0)
#             c = torch.unsqueeze(c, dim = 0)
#             if competencies is not None:
#                 competencies = torch.unsqueeze(competencies, dim = 0)
#         # x = x.to(self.cbm.device)
#         # y = y.to(self.cbm.device)
#         # c = c.to(self.cbm.device)
#         # competencies = competencies.to(self.cbm.device)
#         with torch.no_grad():
#             outputs = self.cbm._forward(
#                 x,
#                 intervention_idxs=torch.zeros((1,self.n_concepts)).to(x.device),
#                 c=c,
#                 y=y,
#                 train=self._train,
#                 competencies=competencies,
#                 prev_interventions=torch.unsqueeze(torch.IntTensor(self._intervened_concepts), dim = 0),
#             )
#             c_sem, c_pred, y_logits = outputs[0], outputs[1], outputs[2]
#             pos_embeddings = outputs[-2]
#             neg_embeddings = outputs[-1]
#             prob = prev_interventions * c + (1 - prev_interventions) * c_sem
#             embed_prob = prob.clone().repeat(1, self.emb_size)
#             embeddings = (
#                 embed_prob * pos_embeddings.detach() +
#                 (1 - embed_prob) * neg_embeddings.detach()
#             )
#             embeddings = torch.where(prev_interventions.clone().repeat(1, self.emb_size).bool()  , 0, embeddings).cpu().numpy()
#             interventions = torch.unsqueeze(torch.zeros(self._intervened_concepts.shape), dim = 0).to(self.ac_model.device)

#             xx = torch.cat((prob, prob), dim = 0)
#             bb = torch.cat((interventions, interventions), dim = 0)
#             ac_model_output, logpo, sam, pred_sam = self.ac_model.compute_concept_probabilities(
#                         x = xx,
#                         b = bb,
#                         m = bb,
#                         y = None
#                     )
#             ac_model_output = torch.squeeze(
#                 ac_model_output
#             .detach(), dim = 0).cpu().numpy()
#             # print(logpo[0,:].shape)
#             logpo = logpo[0,:].detach().cpu().numpy()
#             sam = sam[0,:,:].detach().cpu().numpy()
#             pred_sam = pred_sam[0,:,:].detach().cpu().numpy()
#             sam_mean = np.mean(sam, axis = 0)
#             sam_std = np.std(sam, axis = 0)
#             pred_sam_mean = np.mean(pred_sam, axis = 0)
#             pred_sam_std = np.std(pred_sam, axis = 0)
#             ac_model_info = np.concatenate((logpo, sam_mean, sam_std, pred_sam_mean, pred_sam_std), axis = -1)
            
#         self._ac_model_output = ac_model_output
#         self._ac_model_info = ac_model_info
#         self._cbm_bottleneck = embeddings[0,:]
#         self._cbm_pred_concepts = c_sem.detach().cpu().numpy()[0,:]
#         self._cbm_pred_output = y_logits.detach().cpu().numpy()[0,:]
#         obs = self._get_obs()
#         info = self._get_info()

#         self.num_interventions = 0
#         import pdb
#         pdb.set_trace()

#         return obs, info

#     def step(self, action):
#         prev_interventions = torch.unsqueeze(torch.IntTensor(self._intervened_concepts), dim = 0)
#         print(prev_interventions.shape)
#         new_interventions_map = torch.unsqueeze(torch.IntTensor(self._intervened_concepts_map), dim = 0)
#         print(new_interventions_map.shape)
#         new_interventions_map[0, action] = 1
#         new_interventions = prev_interventions.clone()
#         print(new_interventions.shape)
#         concept_group = sorted(list(self.concept_group_map.keys()))[action]
#         for concept in self.concept_group_map[concept_group]:
#             new_interventions[0, concept] = 1
        
#         x, y, (c, competencies, _) = self.cbm._unpack_batch(self._cbm_data)
        
#         print("Unpacking")
#         if len(x.shape) == 3:
#             x = torch.unsqueeze(x, dim = 0) 
#             y = torch.unsqueeze(y, dim = 0)
#             c = torch.unsqueeze(c, dim = 0)
#             if competencies is not None:
#                 competencies = torch.unsqueeze(competencies, dim = 0)
#         print(x.shape)
#         print(c.shape)
#         print(y.shape)
#         print(new_interventions.shape)
#         if competencies is not None:
#             print(competencies.shape)
#         print(prev_interventions.shape)
#         with torch.no_grad():
#             outputs = self.cbm._forward(
#                 x,
#                 intervention_idxs=new_interventions,
#                 c=c,
#                 y=y,
#                 train=self._train,
#                 competencies=competencies,
#                 prev_interventions=prev_interventions,
#             )
#             c_sem, c_pred, y_logits = outputs[0], outputs[1], outputs[2]
#             pos_embeddings = outputs[-2]
#             neg_embeddings = outputs[-1]
#             print("Prob calc")
#             prob = prev_interventions * c + (1 - prev_interventions) * c_sem
#             embed_prob = prob.clone().repeat(1, self.emb_size)
            
#             print("Embed calc")
#             embeddings = (
#                 embed_prob * pos_embeddings.detach() +
#                 (1 - embed_prob) * neg_embeddings.detach()
#             )
            
#             print("Embed mask calc")
#             embeddings = torch.where(prev_interventions.clone().repeat(1, self.emb_size).bool(), 0, embeddings).cpu().numpy()
#             xx = torch.cat((prob, prob), dim = 0).float().to(self.ac_model.device)
            
#             print("Cat bb")
#             bb = torch.cat((prev_interventions, new_interventions), dim = 0).float().to(self.ac_model.device)
#             print(xx)
#             print(bb)
#             ac_model_output, logpo, sam, pred_sam = self.ac_model.compute_concept_probabilities(
#                         x = xx,
#                         b = bb,
#                         m = bb,
#                         y = None
#                     )
#             print("calculating reward")
#             self.num_interventions += 1
#             terminated = self.num_interventions == self._budget
#             print("AC model output", ac_model_output)
#             reward = self.calculate_reward(terminated, y_logits, y, ac_model_output.cpu().numpy())
#             print("finished calculating reward reward")
#             ac_model_output = torch.squeeze(
#                 ac_model_output
#             .detach(), dim = 0).cpu().numpy()
#             # print(logpo[0,:].shape)
#             logpo = logpo[0,:].detach().cpu().numpy()
#             sam = sam[0,:,:].detach().cpu().numpy()
#             pred_sam = pred_sam[0,:,:].detach().cpu().numpy()
#             sam_mean = np.mean(sam, axis = 0)
#             sam_std = np.std(sam, axis = 0)
#             pred_sam_mean = np.mean(pred_sam, axis = 0)
#             pred_sam_std = np.std(pred_sam, axis = 0)
#             ac_model_info = np.concatenate((logpo, sam_mean, sam_std, pred_sam_mean, pred_sam_std), axis = -1)
            
#         self._ac_model_output = ac_model_output
#         self._ac_model_info = ac_model_info
#         self._cbm_bottleneck = embeddings[0,:]
#         self._cbm_pred_concepts = c_sem.detach().cpu().numpy()[0,:]
#         self._cbm_pred_output = y_logits.detach().cpu().numpy()[0,:]
#         obs = self._get_obs()
#         info = self._get_info()


#         return obs, reward, terminated, False, info
    
#     def calculate_reward(self, terminated, predictions, y, ac_model_output):
#         if terminated:
#             return -self.xent_loss(predictions, y)
#         else:
#             pre_prob, post_prob = np.split(ac_model_output, 2, axis = 0)
#             pre_entropy = self.entropy(pre_prob.T) if pre_prob[0] != 0 else -1e9
#             post_entropy = self.entropy(post_prob.T) if post_prob[0] != 0 else 1e9
#             return self.intermediate_reward_ratio * (pre_entropy - post_entropy)

