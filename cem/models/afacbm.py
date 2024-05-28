import os
import numpy as np
import pytorch_lightning as pl
import sklearn.metrics
import torch
import logging
import time
import gymnasium as gym
import scipy

from torchvision.models import resnet50
from torchmetrics import Accuracy
from gymnasium import spaces

from cem.models.cbm import ConceptBottleneckModel, compute_accuracy
from cem.models.cem import ConceptEmbeddingModel
from cem.models.acflow import ACFlow, ACTransformDataset
from cem.models.acenergy import ACEnergy
# from cem.models.afaenv import AFAEnv
from cem.interventions.intervention_policy import InterventionPolicy
from cem.models.rl import PPOLightningAgent
import cem.train.utils as utils

class ACConceptBottleneckModel(ConceptBottleneckModel):
    def __init__(
        self,
        n_concepts,
        n_tasks,
        concept_loss_weight=0.01,
        task_loss_weight=1,

        extra_dims=0,
        bool=False,
        sigmoidal_prob=True,
        sigmoidal_extra_capacity=True,
        bottleneck_nonlinear=None,
        output_latent=False,

        x2c_model=None,
        c_extractor_arch=utils.wrap_pretrained_model(resnet50),
        c2y_model=None,
        c2y_layers=None,
        ac_model_config = {},
        ac_model_nll_ratio = 0.5,
        ac_model_weight = 2,
        ac_model_rollouts = 1,

        optimizer="adam",
        momentum=0.9,
        learning_rate=0.01,
        weight_decay=4e-05,
        weight_loss=None,
        task_class_weights=None,

        active_intervention_values=None,
        inactive_intervention_values=None,
        intervention_policy=None,
        output_interventions=False,
        include_certainty=True,

        intervention_discount=1,
        intervention_task_discount=1.1,
        intervention_weight=5,
        horizon_rate=1.005,
        tau=1,
        average_trajectory=True,
        concept_map=None,
        use_concept_groups=False,
        max_horizon=5,
        include_task_trajectory_loss=False,
        include_only_last_trajectory_loss=False,
        horizon_binary_representation=False,
        intervention_task_loss_weight=1,
        initial_horizon=1,
        horizon_uniform_distr=True,
        beta_a=1,
        beta_b=3,
        use_horizon=True,
        rollout_init_steps=0,
        include_probs=False,
        use_full_mask_distr=False,
        int_model_layers=None,
        int_model_use_bn=False,
        initialize_discount=False,
        propagate_target_gradients=False,
        top_k_accuracy=None,
        num_rollouts=1,
        max_num_rollouts=None,
        rollout_aneal_rate=1,
        backprop_masks=True,
        legacy_mode=False,
        hard_intervention=True,
    ):
        self.hard_intervention = hard_intervention
        self.legacy_mode = legacy_mode
        self.num_rollouts = num_rollouts
        self.rollout_aneal_rate = rollout_aneal_rate
        self.backprop_masks = backprop_masks
        self.max_num_rollouts = max_num_rollouts
        self.propagate_target_gradients = propagate_target_gradients
        self.initialize_discount = initialize_discount
        self.use_horizon = use_horizon
        self.use_full_mask_distr = use_full_mask_distr
        if concept_map is None:
            concept_map = dict([
                (i, [i]) for i in range(n_concepts)
            ])
        self.concept_map = concept_map
        if len(concept_map) == n_concepts:
            use_concept_groups = False
        super(ACConceptBottleneckModel, self).__init__(
            n_concepts=n_concepts,
            n_tasks=n_tasks,
            concept_loss_weight=concept_loss_weight,
            task_loss_weight=task_loss_weight,
            extra_dims=extra_dims,
            bool=bool,
            sigmoidal_prob=sigmoidal_prob,
            sigmoidal_extra_capacity=sigmoidal_extra_capacity,
            bottleneck_nonlinear=bottleneck_nonlinear,
            output_latent=output_latent,
            x2c_model=x2c_model,
            c_extractor_arch=c_extractor_arch,
            c2y_model=c2y_model,
            c2y_layers=c2y_layers,
            optimizer=optimizer,
            momentum=momentum,
            learning_rate=learning_rate,
            weight_decay=weight_decay,
            weight_loss=weight_loss,
            task_class_weights=task_class_weights,
            active_intervention_values=active_intervention_values,
            inactive_intervention_values=inactive_intervention_values,
            intervention_policy=intervention_policy,
            output_interventions=output_interventions,
            top_k_accuracy=top_k_accuracy,
            use_concept_groups=use_concept_groups,
            include_certainty=include_certainty,
        )

        # Else we construct it here directly
        max_horizon_val = \
            len(concept_map) if self.use_concept_groups else n_concepts
        self.include_probs = include_probs
        units = [
            (n_concepts if self.use_concept_groups else len(self.concept_map)) +
            n_concepts + # Bottleneck
            n_concepts + # Prev interventions
            (n_concepts if self.include_probs else 0) + # Predicted Probs
            (
                max_horizon_val if use_horizon and horizon_binary_representation
                else int(use_horizon)
            ) # Horizon
        ] + (int_model_layers or [256, 128]) + [
            len(self.concept_map) if self.use_concept_groups else n_concepts
        ]
        layers = []
        for i in range(1, len(units)):
            if int_model_use_bn:
                layers.append(
                    torch.nn.BatchNorm1d(num_features=units[i-1]),
                )
            layers.append(torch.nn.Linear(units[i-1], units[i]))
            if i != len(units) - 1:
                layers.append(torch.nn.LeakyReLU())
        self.concept_rank_model = torch.nn.Sequential(*layers)
        if "flow" in ac_model_config['architecture']:
            self.ac_model = ACFlow(
                n_concepts = n_concepts,
                n_tasks = n_tasks,
                layer_cfg = ac_model_config['layer_cfg'],
                affine_hids = ac_model_config['affine_hids'],
                linear_rank = ac_model_config['linear_rank'],
                linear_hids = ac_model_config['linear_hids'],
                transformations = ac_model_config['transformations'],
                prior_units = ac_model_config['prior_units'],
                prior_layers = ac_model_config['prior_layers'],
                prior_hids = ac_model_config['prior_hids'],
                n_components = ac_model_config['n_components']
            ).to(self.device)
        elif "energy" in ac_model_config['architecture']:
            self.ac_model = ACEnergy(
                n_concepts = n_concepts,
                n_tasks = n_tasks,
                # embed_size = ac_model_config["embed_size"],
                # cy_perturb_prob = ac_model_config.get("cy_perturb_prob", None),
                # cy_perturb_prob = ac_model_config.get("", None)
            ).to(self.device)
        else:
            raise ValueError(f"AC{ac_model_config['architecture']} architecture not supported")
        if ac_model_config.get("save_path", None) is not None:
            chpt_exists = (
                os.path.exists(ac_model_config['save_path'])
            )
            if chpt_exists:
                self.ac_model.load_state_dict(torch.load(ac_model_config['save_path']))
                logging.debug(
                    f"AC CBM loaded AC model checkpoint from {ac_model_config['save_path']}"
                )
                self.train_ac_model = False
            # else:
            #     raise ValueError(f"AC{ac_model_config['architecture']} model checkpoint at {ac_model_config['save_path']} incorrect / not found")
            self.train_ac_model = False
        else:
            self.train_ac_model = True
            logging.debug(
                f"Training AC {ac_model_config['architecture']} model simultaneously with CEM model."
            )

        self.ac_model_nll_ratio = ac_model_nll_ratio
        self.ac_model_weight = ac_model_weight
        self.ac_model_rollouts = ac_model_rollouts
        self.ac_softmax = torch.nn.Softmax(dim = 1).to(self.device)

        self.intervention_discount = intervention_discount
        self.intervention_task_discount = intervention_task_discount
        self.horizon_rate = horizon_rate
        self.horizon_limit = torch.nn.Parameter(
            torch.FloatTensor([initial_horizon]),
            requires_grad=False,
        )
        if not legacy_mode:
            self.current_steps = torch.nn.Parameter(
                torch.IntTensor([0]),
                requires_grad=False,
            )
            if self.rollout_aneal_rate != 1:
                self.current_aneal_rate = torch.nn.Parameter(
                    torch.FloatTensor([1]),
                    requires_grad=False,
                )
        self.rollout_init_steps = rollout_init_steps
        self.tau = tau
        self.intervention_weight = intervention_weight
        self.average_trajectory = average_trajectory
        self.loss_interventions = torch.nn.CrossEntropyLoss()
        self.ac_model_xent_loss = torch.nn.CrossEntropyLoss(weight=task_class_weights) if n_tasks > 1 else torch.nn.BCEWithLogitsLoss(
            pos_weight=task_class_weights 
        )
        self.max_horizon = max_horizon
        self.emb_size = 1
        self.include_task_trajectory_loss = include_task_trajectory_loss
        self.horizon_binary_representation = horizon_binary_representation
        self.include_only_last_trajectory_loss = \
            include_only_last_trajectory_loss
        self.intervention_task_loss_weight = intervention_task_loss_weight
        
        self.horizon_uniform_distr = horizon_uniform_distr
        self.beta_a = beta_a
        self.beta_b = beta_b

    def _horizon_distr(self, init, end):
        if self.horizon_uniform_distr:
            return np.random.randint(init,end)
        else:
            return int(np.random.beta(self.beta_a, self.beta_b) * (end - init) + init)


    def get_concept_int_distribution(
        self,
        x,
        c,
        prev_interventions=None,
        competencies=None,
        horizon=1,
    ):
        if prev_interventions is None:
            prev_interventions = torch.zeros(c.shape).to(x.device)
        outputs = self._forward(
            x,
            c=c,
            y=None,
            train=False,
            competencies=competencies,
            prev_interventions=prev_interventions,
            output_embeddings=True,
            output_latent=True,
            output_interventions=True,
        )

        c_sem, c_logits, y_logits = outputs[0], outputs[1], outputs[2]
        prev_interventions = outputs[3]
        pos_embeddings = outputs[-2]
        neg_embeddings = outputs[-1]
        return self._prior_int_distribution(
            prob=c_sem,
            pos_embeddings=pos_embeddings,
            neg_embeddings=neg_embeddings,
            c=c,
            competencies=competencies,
            prev_interventions=prev_interventions,
            horizon=horizon,
            train=False,
        )


    def _prior_int_distribution(
        self,
        prob,
        pos_embeddings,
        neg_embeddings,
        c,
        competencies=None,
        prev_interventions=None,
        horizon=1,
        train=False,
    ):
        if prev_interventions is None:
            prev_interventions = torch.zeros(prob.shape).to(
                pos_embeddings.device
            )
        if competencies is None:
            competencies = torch.ones(prob.shape).to(pos_embeddings.device)
        # Shape is [B, n_concepts, emb_size]
        prob = prev_interventions * c + (1 - prev_interventions) * prob
        embeddings = (
            torch.unsqueeze(prob, dim=-1) * pos_embeddings +
            (1 - torch.unsqueeze(prob, dim=-1)) * neg_embeddings
        )
        # Zero out embeddings of previously intervened concepts
        if self.use_concept_groups:
            available_groups = torch.zeros(
                (embeddings.shape[0], len(self.concept_map))
            ).to(
                embeddings.device
            )
            for group_idx, (_, group_concepts) in enumerate(
                self.concept_map.items()
            ):
                available_groups[:, group_idx] = torch.logical_not(torch.any(
                    prev_interventions[:, group_concepts] == 1,
                    dim=-1,
                ))
            max_horizon = len(self.concept_map)
        else:
            available_groups = (1 - prev_interventions).to(embeddings.device)
            max_horizon = self.n_concepts
        used_groups = 1 - available_groups
        
        num_groups = int(torch.sum(available_groups[0]).detach())

        unintervened_groups = [torch.nonzero(available_groups[i], as_tuple = False).squeeze(dim = 1) for i in range(available_groups.shape[0])]

        try:
            unintervened_groups = torch.stack(unintervened_groups, dim = 0)
        except:
            max_length = max([t.size(0) for t in unintervened_groups])
            min_length = min([t.size(0) for t in unintervened_groups])
            padded_list = [t if t.numel() == max_length else torch.cat([t, \
                t[torch.multinomial(torch.ones_like(t, dtype = torch.float) / t.numel(), num_samples=max_length - t.numel(), replacement=True)]]) \
            for t in unintervened_groups]

            unintervened_groups = torch.stack(padded_list)
            num_groups = max_length
                    
        likel_sparse = torch.ones(used_groups.shape, dtype = torch.float32, device = used_groups.device)

        likel_sparse = likel_sparse * -1000

        mask = prev_interventions.clone().float()
        missing = prev_interventions.clone().float()
        predicted_and_intervened_concepts = prob.clone()
        concept_map_vals = list(self.concept_map.values())
        for i in range(num_groups):
            for b in range(used_groups.shape[0]):
                for concept in concept_map_vals[int(unintervened_groups[b][i])]:
                    missing[b][concept] = 1.
            if self.train_ac_model:
                    loglikel = self.ac_model.compute_concept_probabilities(x = predicted_and_intervened_concepts, b = mask, m = missing, y = None)
                    if isinstance(loglikel, tuple) or isinstance(loglikel, list):
                        loglikel = loglikel[0]
            else:
                with torch.no_grad():
                    try:
                        loglikel = self.ac_model.compute_concept_probabilities(x = predicted_and_intervened_concepts, b = mask, m = missing, y = None)
                        if isinstance(loglikel, tuple) or isinstance(loglikel, list):
                            loglikel = loglikel[0]
                    except:
                        logging.warning(
                            f"loglikel = {loglikel}"
                            f"ac_model.device:{self.ac_model.device}\n"
                            f"mask.device:{mask.device}\n"
                            f"missing.device:{missing.device}\n"
                            f"predicted_and_intervened_concepts.device:{predicted_and_intervened_concepts.device}\n"
                        )
                    # loglikel = torch.zeros_lke(loglikel)
            batches = torch.arange(used_groups.shape[0])
            indices = unintervened_groups[batches, i].cpu()
            likel_sparse[batches, indices] = loglikel[batches] 

            for b in range(used_groups.shape[0]):
                for concept in concept_map_vals[int(unintervened_groups[b][i])]:
                    missing[b][concept] = 0.
        likel_sparse = self.ac_softmax(likel_sparse)

        
        cat_inputs = [
            likel_sparse,
            torch.reshape(embeddings, [-1, self.emb_size * self.n_concepts]),
            prev_interventions,
#                 competencies,
        ]
        if self.include_probs:
            cat_inputs.append(prob)
        if self.use_horizon:
            cat_inputs.append((
                torch.ones(
                    [embeddings.shape[0], 1]
                ).to(prev_interventions) * horizon
                if (not self.horizon_binary_representation) else
                torch.concat(
                    [
                        torch.ones([embeddings.shape[0], horizon]),
                        torch.zeros(
                            [embeddings.shape[0], max_horizon - horizon]
                        ),
                    ],
                    dim=-1
                ).to(prev_interventions)
            ))
        rank_input = torch.concat(
            cat_inputs,
            dim=-1,
        )
        next_concept_group_scores = self.concept_rank_model(
            rank_input
        )
        if train:
            return next_concept_group_scores
        next_concept_group_scores = torch.where(
            used_groups == 1,
            torch.ones(used_groups.shape).to(used_groups.device) * (-1000),
            next_concept_group_scores,
        )
        return torch.nn.functional.softmax(
            next_concept_group_scores/self.tau,
            dim=-1,
        )

    def _run_step(
        self,
        batch,
        batch_idx,
        train=False,
        intervention_idxs=None,
    ):
        x, y, (c, competencies, prev_interventions) = self._unpack_batch(batch)
        outputs = self._forward(
            x,
            intervention_idxs=intervention_idxs,
            c=c,
            y=y,
            train=train,
            competencies=competencies,
            prev_interventions=prev_interventions,
            output_embeddings=True,
            output_latent=True,
            output_interventions=True,
        )
        c_sem, c_logits, y_logits = outputs[0], outputs[1], outputs[2]
        prev_interventions = outputs[3]
        latent = outputs[4]
        pos_embeddings = outputs[-2]
        neg_embeddings = outputs[-1]

        if self.task_loss_weight != 0:
            task_loss = self.loss_task(
                y_logits if y_logits.shape[-1] > 1 else y_logits.reshape(-1),
                y,
            )
            task_loss_scalar = self.task_loss_weight * task_loss.detach()
        else:
            task_loss = 0.0
            task_loss_scalar = 0.0

        ac_model_loss = 0.0
        ac_model_loss_scalar = 0.0
        # Do some rollouts for flow model
        if self.ac_model_weight != 0 and train and self.train_ac_model:
            logging.debug(
                f"Passing through ac model"
            )
            if self.rollout_aneal_rate != 1:
                ac_model_rollouts = int(round(
                    self.ac_model_rollouts * (
                        self.current_aneal_rate.detach().cpu().numpy()[0]
                    )
                ))
            else:
                ac_model_rollouts = self.ac_model_rollouts

            for _ in range(ac_model_rollouts):
                x_ac, b_ac, m_ac, y_ac = ACTransformDataset.transform_batch(c, y)
                logpu, logpo, _, _, _ = self.ac_model(x_ac, b_ac, m_ac, y_ac)
                logits = logpu + logpo
                loglikel = torch.logsumexp(logpu + logpo, dim = 1) - torch.logsumexp(logpo, dim = 1)
                nll = torch.mean(-loglikel)
                ac_model_loss += (1 - self.ac_model_nll_ratio) * self.ac_model_xent_loss(logits, y_ac) + self.ac_model_nll_ratio * nll
            

            ac_model_loss = ac_model_loss / ac_model_rollouts
            ac_model_loss_scalar = ac_model_loss.detach() * self.ac_model_weight

        else:
            self.ac_model.eval()

        intervention_task_loss = 0.0

        # Now we will do some rolls for interventions
        int_mask_accuracy = -1.0
        current_horizon = -1
        if not self.include_certainty:
            c_used = torch.where(
                torch.logical_or(c == 0, c == 1),
                c,
                c_sem.detach(),
            )
        else:
            c_used = c

        if self.cbm.include_task_trajectory_loss and (
                self.cbm.task_loss_weight == 0
            ):
            # Then we initialize the intervention trajectory task loss to
            # that of the unintervened model as this loss is not going to
            # be taken into account if we don't do this
            intervention_task_loss = first_task_discount * self.cbm.loss_task(
                (
                    y_logits if y_logits.shape[-1] > 1
                    else y_logits.reshape(-1)
                ),
                y,
            )
            if self.cbm.average_trajectory:
                intervention_task_loss = (
                    intervention_task_loss / task_trajectory_weight
                )
        
        
        if (
            train and
            (intervention_idxs is None)
        ):
            intervention_loss = 0.0
            if prev_interventions is not None:
                # This will be not None in the case of RandInt, so we can ASSUME
                # they have all been intervened the same number of times before
                intervention_idxs = prev_interventions[:]
                if self.use_concept_groups:
                    free_groups = torch.ones(
                        (prev_interventions.shape[0], len(self.concept_map))
                    ).to(
                        c_used.device
                    )
                    # BIG ASSUMPTION: THEY HAVE ALL BEEN INTERVENED ON THE SAME
                    # NUMBER OF CONCEPTS
                    cpu_ints = intervention_idxs[0, :].detach().cpu().numpy()
                    for group_idx, (_, group_concepts) in enumerate(
                        self.concept_map.items()
                    ):
                        if np.any(cpu_ints[group_concepts] == 1):
                            free_groups[:,group_idx] = 0
                    prev_num_of_interventions = int(
                        len(self.concept_map) - np.sum(
                            free_groups[0, :].detach().cpu().numpy(),
                            axis=-1,
                        ),
                    )
                else:
                    free_groups = 1 - intervention_idxs
                    prev_num_of_interventions = int(np.max(
                        np.sum(
                            intervention_idxs.detach().cpu().numpy(),
                            axis=-1,
                        ),
                        axis=-1,
                    ))
            else:
                intervention_idxs = torch.zeros(c_used.shape).to(c_used.device)
                prev_num_of_interventions = 0
                if self.use_concept_groups:
                    free_groups = torch.ones(
                        (c_used.shape[0], len(self.concept_map))
                    ).to(c_used.device)
                else:
                    free_groups = torch.ones(c_used.shape).to(c_used.device)

            # Then time to perform a forced training time intervention
            # We will set some of the concepts as definitely intervened on
            if competencies is None:
                competencies = torch.FloatTensor(
                    c_used.shape
                ).uniform_(0.5, 1).to(c_used.device)
            int_basis_lim = (
                len(self.concept_map) if self.use_concept_groups
                else self.n_concepts
            )
            horizon_lim = int(self.horizon_limit.detach().cpu().numpy()[0])
            if prev_num_of_interventions != int_basis_lim:
                bottom = min(
                    horizon_lim,
                    int_basis_lim - prev_num_of_interventions - 1,
                )  # -1 so that we at least intervene on one concept
                if bottom > 0:
                    initially_selected = np.random.randint(0, bottom)
                else:
                    initially_selected = 0
                end_horizon = min(
                    int(horizon_lim),
                    self.max_horizon,
                    int_basis_lim - prev_num_of_interventions - initially_selected,
                )
                current_horizon = self._horizon_distr(
                    init=1 if end_horizon > 1 else 0,
                    end=end_horizon,
                )

                for sample_idx in range(intervention_idxs.shape[0]):
                    probs = free_groups[sample_idx, :].detach().cpu().numpy()
                    probs = probs/np.sum(probs)
                    if self.use_concept_groups:
                        selected_groups = set(np.random.choice(
                            int_basis_lim,
                            size=initially_selected,
                            replace=False,
                            p=probs,
                        ))
                        for group_idx, (_, group_concepts) in enumerate(
                            self.concept_map.items()
                        ):
                            if group_idx in selected_groups:
                                intervention_idxs[sample_idx, group_concepts] = 1
                    else:
                        intervention_idxs[
                            sample_idx,
                            np.random.choice(
                                int_basis_lim,
                                size=initially_selected,
                                replace=False,
                                p=probs,
                            )
                        ] = 1
                if self.initialize_discount:
                    discount = (
                        self.intervention_discount ** prev_num_of_interventions
                    )
                    task_discount = self.intervention_task_discount ** (
                        prev_num_of_interventions + initially_selected
                    )
                    first_task_discount = (
                        self.intervention_task_discount **
                        prev_num_of_interventions
                    )
                else:
                    discount = 1
                    task_discount = 1
                    first_task_discount = 1

                trajectory_weight = 0
                task_trajectory_weight = 1
                if self.average_trajectory:
                    curr_discount = discount
                    curr_task_discount = task_discount
                    for i in range(current_horizon):
                        trajectory_weight += discount
                        discount *= self.intervention_discount
                        task_discount *= self.intervention_task_discount
                        if (
                            (not self.include_only_last_trajectory_loss) or
                            (i == current_horizon - 1)
                        ):
                            task_trajectory_weight += task_discount
                        
                    discount = curr_discount
                    task_discount = curr_task_discount
                else:
                    trajectory_weight = 1

                if self.n_tasks > 1:
                    one_hot_y = torch.nn.functional.one_hot(y, self.n_tasks)
            else:
                current_horizon = 0
                if self.initialize_discount:
                    task_discount = self.intervention_task_discount ** (
                        prev_num_of_interventions
                    )
                    first_task_discount = (
                        self.intervention_task_discount **
                        prev_num_of_interventions
                    )
                else:
                    task_discount = 1
                    first_task_discount = 1
                task_trajectory_weight = 1
                trajectory_weight = 1

            if self.include_task_trajectory_loss and (
                self.task_loss_weight == 0
            ):
                # Then we initialize the intervention trajectory task loss to
                # that of the unintervened model as this loss is not going to
                # be taken into account if we don't do this
                intervention_task_loss = first_task_discount * self.loss_task(
                    (
                        y_logits if y_logits.shape[-1] > 1
                        else y_logits.reshape(-1)
                    ),
                    y,
                )
                if self.average_trajectory:
                    intervention_task_loss = (
                        intervention_task_loss / task_trajectory_weight
                    )
            int_mask_accuracy = 0.0 if current_horizon else -1
            if (not self.legacy_mode) and (
                (
                    self.current_steps.detach().cpu().numpy()[0] <
                    self.rollout_init_steps
                )
            ):
                current_horizon = 0
            if self.rollout_aneal_rate != 1:
                num_rollouts = int(round(
                    self.num_rollouts * (
                        self.current_aneal_rate.detach().cpu().numpy()[0]
                    )
                ))
            else:
                num_rollouts = self.num_rollouts
            if self.max_num_rollouts is not None:
                num_rollouts = min(num_rollouts, self.max_num_rollouts)

            for i in range(num_rollouts):
                self.current_horizon = current_horizon
                for j in range(current_horizon):
                    # And generate a probability distribution over previously
                    # unseen concepts to indicate which one we should intervene
                    # on next!
                    self.horizon_index = j
                    concept_group_scores = self._prior_int_distribution(
                        prob=c_sem,
                        pos_embeddings=pos_embeddings,
                        neg_embeddings=neg_embeddings,
                        competencies=competencies,
                        prev_interventions=intervention_idxs,
                        c=c_used,
                        horizon=(current_horizon - i),
                        train=train,
                    )
                    # batch_size = c_sem.shape[0]
                    # concept_group_scores = torch.zeros((batch_size, len(self.concept_map) if self.use_concept_groups else self.n_concepts))
                    # Generate as a label the concept which increases the
                    # probability of the correct class the most when
                    # intervened on
                    target_int_logits = torch.ones(
                        concept_group_scores.shape,
                    ).to(c_used.device) * (-np.Inf)
                    for target_concept in range(target_int_logits.shape[-1]):
                        if self.use_concept_groups:
                            new_int = torch.zeros(
                                intervention_idxs.shape
                            ).to(intervention_idxs.device)
                            for group_idx, (_, group_concepts) in enumerate(
                                self.concept_map.items()
                            ):
                                if group_idx == target_concept:
                                    new_int[:, group_concepts] = 1
                                    break
                        else:
                            new_int = torch.zeros(
                                intervention_idxs.shape
                            ).to(intervention_idxs.device)
                            new_int[:, target_concept] = 1
                        if not self.propagate_target_gradients:
                            partial_ints = torch.clamp(
                                intervention_idxs.detach() + new_int,
                                0,
                                1,
                            )
                            probs = (
                                c_sem.detach() * (1 - partial_ints) +
                                c_used * partial_ints
                            )
                            c_rollout_pred = (
                                (
                                    pos_embeddings.detach() *
                                    torch.unsqueeze(probs, dim=-1)
                                ) +
                                (
                                    neg_embeddings.detach() *
                                    (1 - torch.unsqueeze(probs, dim=-1))
                                )
                            )
                        else:
                            partial_ints = torch.clamp(
                                intervention_idxs + new_int,
                                0,
                                1,
                            )
                            probs = (
                                c_sem * (1 - partial_ints) +
                                c_used * partial_ints
                            )
                            c_rollout_pred = (
                                (
                                    pos_embeddings *
                                    torch.unsqueeze(probs, dim=-1)
                                ) + (
                                    neg_embeddings *
                                    (1 - torch.unsqueeze(probs, dim=-1))
                                )
                            )
                        c_rollout_pred = c_rollout_pred.view(
                            (-1, self.emb_size * self.n_concepts)
                        )

                        rollout_y_logits = self.c2y_model(c_rollout_pred)

                        if self.n_tasks > 1:
                            target_int_logits[:, target_concept] = \
                                rollout_y_logits[
                                    one_hot_y.type(torch.BoolTensor)
                                ]
                        else:
                            pred_y_prob = torch.sigmoid(
                                torch.squeeze(rollout_y_logits, dim=-1)
                            )
                            target_int_logits[:, target_concept] = torch.where(
                                y == 1,
                                torch.log(
                                    (pred_y_prob + 1e-15) /
                                    (1 - pred_y_prob + 1e-15)
                                ),
                                torch.log(
                                    (1 - pred_y_prob + 1e-15) /
                                    (pred_y_prob+ 1e-15)
                                ),
                            )
                    if self.use_full_mask_distr:
                        target_int_labels = torch.nn.functional.softmax(
                            target_int_logits,
                            -1,
                        )
                        pred_int_labels = concept_group_scores.argmax(-1)
                        curr_acc = (
                            pred_int_labels == torch.argmax(
                                target_int_labels,
                                -1,
                            )
                        ).float().mean()
                    else:
                        target_int_labels = torch.argmax(target_int_logits, -1)
                        pred_int_labels = concept_group_scores.argmax(-1)
                        curr_acc = (
                            pred_int_labels == target_int_labels
                        ).float().mean()

                    int_mask_accuracy += curr_acc/current_horizon
                    new_loss = self.loss_interventions(
                        concept_group_scores,
                        target_int_labels,
                    )
                    if not self.backprop_masks:
                        # Then block the gradient into the masks
                        concept_group_scores = concept_group_scores.detach()

                    # Update the next-concept predictor loss
                    if self.average_trajectory:
                        intervention_loss += (
                            discount * new_loss/trajectory_weight
                        )
                    else:
                        intervention_loss += discount * new_loss

                    # Update the discount (before the task trajectory loss to
                    # start discounting from the first intervention so that the
                    # loss of the unintervened model is highest
                    discount *= self.intervention_discount
                    task_discount *= self.intervention_task_discount

                    # Sample the next concepts we will intervene on using a hard
                    # Gumbel softmax
                    if self.intervention_weight == 0:
                        selected_groups = torch.FloatTensor(
                            np.eye(concept_group_scores.shape[-1])[np.random.choice(
                                concept_group_scores.shape[-1],
                                size=concept_group_scores.shape[0]
                            )]
                        ).to(concept_group_scores.device)
                    else:
                        selected_groups = torch.nn.functional.gumbel_softmax(
                            concept_group_scores,
                            dim=-1,
                            hard=self.hard_intervention,
                            tau=self.tau,
                        )
                    if self.use_concept_groups:
                        prev_intervention_idxs = intervention_idxs.clone()
                        for sample_idx in range(intervention_idxs.shape[0]):
                            for group_idx, (_, group_concepts) in enumerate(
                                self.concept_map.items()
                            ):
                                if selected_groups[sample_idx, group_idx] == 1:
                                    intervention_idxs[
                                        sample_idx,
                                        group_concepts,
                                    ] = 1
                    else:
                        intervention_idxs += selected_groups

                    if self.include_task_trajectory_loss and (
                        (not self.include_only_last_trajectory_loss) or
                        (i == (current_horizon - 1))
                    ):
                        # Then we will also update the task loss with the loss
                        # of performing this intervention!
                        probs = (
                            c_sem * (1 - intervention_idxs) +
                            c_used * intervention_idxs
                        )
                        c_rollout_pred = (
                            pos_embeddings * torch.unsqueeze(probs, dim=-1) + (
                                neg_embeddings * (
                                    1 - torch.unsqueeze(probs, dim=-1)
                                )
                            )
                        )
                        c_rollout_pred = c_rollout_pred.view(
                            (-1, self.emb_size * self.n_concepts)
                        )
                        rollout_y_logits = self.c2y_model(c_rollout_pred)
                        rollout_y_loss = self.loss_task(
                            (
                                rollout_y_logits
                                if rollout_y_logits.shape[-1] > 1 else
                                rollout_y_logits.reshape(-1)
                            ),
                            y,
                        )
                        if self.average_trajectory:
                            intervention_task_loss += (
                                task_discount *
                                rollout_y_loss / task_trajectory_weight
                            )
                        else:
                            intervention_task_loss += (
                                task_discount * rollout_y_loss
                            )

                if (
                    self.legacy_mode or
                    (
                        self.current_steps.detach().cpu().numpy()[0] >=
                            self.rollout_init_steps
                    )
                ) and (
                    self.horizon_limit.detach().cpu().numpy()[0] <
                        int_basis_lim + 1
                ):
                    self.horizon_limit *= self.horizon_rate

            intervention_loss_scalar = \
                self.intervention_weight * intervention_loss
            intervention_loss = intervention_loss/num_rollouts
            intervention_task_loss = intervention_task_loss/num_rollouts
            int_mask_accuracy = int_mask_accuracy/num_rollouts
        else:
            intervention_loss = 0.0
            intervention_loss_scalar = 0.0

        if not self.legacy_mode:
            self.current_steps += 1
            if self.rollout_aneal_rate != 1 and (
                self.current_aneal_rate.detach().cpu().numpy()[0] < 100
            ):
                self.current_aneal_rate *= self.rollout_aneal_rate

        if self.include_task_trajectory_loss and (
            self.intervention_task_loss_weight != 0
        ):
            if isinstance(intervention_task_loss, float):
                intervention_task_loss_scalar = (
                    self.intervention_task_loss_weight * intervention_task_loss
                )
            else:
                intervention_task_loss_scalar = (
                    self.intervention_task_loss_weight *
                    intervention_task_loss.detach()
                )
        else:
            intervention_task_loss_scalar = 0.0


        if self.concept_loss_weight != 0:
            # We separate this so that we are allowed to
            # use arbitrary activations (i.e., not necessarily in [0, 1])
            # whenever no concept supervision is provided
            if self.include_certainty:
                concept_loss = self.loss_concept(c_sem, c)
                concept_loss_scalar = \
                    self.concept_loss_weight * concept_loss.detach()
            else:
                c_sem_used = torch.where(
                    torch.logical_or(c == 0, c == 1),
                    c_sem,
                    c,
                ) # This forces zero loss when c is uncertain
                concept_loss = self.loss_concept(c_sem_used, c)
                concept_loss_scalar = concept_loss.detach()
        else:
            concept_loss = 0.0
            concept_loss_scalar = 0.0

        loss = (
            self.concept_loss_weight * concept_loss +
            self.intervention_weight * intervention_loss +
            self.ac_model_weight * ac_model_loss +
            self.task_loss_weight * task_loss +
            self.intervention_task_loss_weight * intervention_task_loss
        )

        loss += self._extra_losses(
            x=x,
            y=y,
            c=c,
            c_sem=c_sem,
            c_pred=c_logits,
            y_pred=y_logits,
            competencies=competencies,
            prev_interventions=prev_interventions,
        )
        # compute accuracy
        (c_accuracy, c_auc, c_f1), (y_accuracy, y_auc, y_f1) = compute_accuracy(
            c_sem,
            y_logits,
            c,
            y,
        )
        result = {
            "c_accuracy": c_accuracy,
            "c_auc": c_auc,
            "c_f1": c_f1,
            "y_accuracy": y_accuracy,
            "y_auc": y_auc,
            "y_f1": y_f1,
            "mask_accuracy": int_mask_accuracy,
            "concept_loss": concept_loss_scalar,
            "task_loss": task_loss_scalar,
            "intervention_task_loss": intervention_task_loss_scalar,
            "intervention_loss": intervention_loss_scalar,
            "ac_model_loss": ac_model_loss_scalar,
            "loss": loss.detach() if not isinstance(loss, float) else loss,
            "avg_c_y_acc": (c_accuracy + y_accuracy) / 2,
            "horizon_limit": self.horizon_limit.detach().cpu().numpy()[0],
        }
        if not self.legacy_mode:
            result["current_steps"] = \
                self.current_steps.detach().cpu().numpy()[0]
            if self.rollout_aneal_rate != 1:
                num_rollouts = int(round(
                    self.num_rollouts * (
                        self.current_aneal_rate.detach().cpu().numpy()[0]
                    )
                ))
                if self.max_num_rollouts is not None:
                    num_rollouts = min(num_rollouts, self.max_num_rollouts)
                result["num_rollouts"] = num_rollouts

        if self.top_k_accuracy is not None:
            y_true = y.reshape(-1).cpu().detach()
            y_pred = y_logits.cpu().detach()
            labels = list(range(self.n_tasks))
            for top_k_val in self.top_k_accuracy:
                y_top_k_accuracy = sklearn.metrics.top_k_accuracy_score(
                    y_true,
                    y_pred,
                    k=top_k_val,
                    labels=labels,
                )
                result[f'y_top_{top_k_val}_accuracy'] = y_top_k_accuracy
        return loss, result

class ACConceptEmbeddingModel(
    ConceptEmbeddingModel,
    ACConceptBottleneckModel,
):
    def __init__(
        self,
        n_concepts,
        n_tasks,
        emb_size=16,
        training_intervention_prob=0.25,
        embedding_activation="leakyrelu",
        concept_loss_weight=1,
        task_loss_weight=1,

        c2y_model=None,
        c2y_layers=None,
        c_extractor_arch=utils.wrap_pretrained_model(resnet50),
        output_latent=False,

        ac_model_config = {},
        ac_model_nll_ratio = 0.5,
        ac_model_weight = 2,
        ac_model_rollouts = 1,

        optimizer="adam",
        momentum=0.9,
        learning_rate=0.01,
        weight_decay=4e-05,
        weight_loss=None,
        task_class_weights=None,

        active_intervention_values=None,
        inactive_intervention_values=None,
        intervention_policy=None,
        output_interventions=False,
        include_certainty=True,

        top_k_accuracy=None,

        intervention_discount=0.9,
        intervention_task_discount=0.9,
        intervention_weight=5,
        horizon_rate=1.005,
        tau=1,
        average_trajectory=True,
        concept_map=None,
        use_concept_groups=False,
        max_horizon=5,
        include_task_trajectory_loss=False,
        horizon_binary_representation=False,
        include_only_last_trajectory_loss=False,
        intervention_task_loss_weight=1,
        initial_horizon=1,
        horizon_uniform_distr=True,
        beta_a=1,
        beta_b=3,
        use_horizon=True,
        rollout_init_steps=0,
        include_probs=False,
        use_full_mask_distr=False,
        int_model_layers=None,
        int_model_use_bn=False,
        initialize_discount=False,
        propagate_target_gradients=False,
        num_rollouts=1,
        max_num_rollouts=None,
        rollout_aneal_rate=1,
        backprop_masks=True,
        legacy_mode=False,
        hard_intervention=True,
    ):
        self.hard_intervention = hard_intervention
        self.legacy_mode = legacy_mode
        self.num_rollouts = num_rollouts
        self.backprop_masks = backprop_masks
        self.rollout_aneal_rate = rollout_aneal_rate
        self.max_num_rollouts = max_num_rollouts
        self.propagate_target_gradients = propagate_target_gradients
        self.initialize_discount = initialize_discount
        self.use_full_mask_distr = use_full_mask_distr
        self.use_horizon = use_horizon
        if concept_map is None:
            concept_map = dict([
                (i, [i]) for i in range(n_concepts)
            ])
        self.concept_map = concept_map
        if len(concept_map) == n_concepts:
            use_concept_groups = False
        ConceptEmbeddingModel.__init__(
            self,
            n_concepts=n_concepts,
            n_tasks=n_tasks,
            emb_size=emb_size,
            training_intervention_prob=training_intervention_prob,
            embedding_activation=embedding_activation,
            shared_prob_gen=False,
            concept_loss_weight=concept_loss_weight,
            task_loss_weight=task_loss_weight,
            c2y_model=c2y_model,
            c2y_layers=c2y_layers,
            c_extractor_arch=c_extractor_arch,
            output_latent=output_latent,
            optimizer=optimizer,
            momentum=momentum,
            learning_rate=learning_rate,
            weight_decay=weight_decay,
            weight_loss=weight_loss,
            task_class_weights=task_class_weights,
            active_intervention_values=active_intervention_values,
            inactive_intervention_values=inactive_intervention_values,
            intervention_policy=intervention_policy,
            output_interventions=output_interventions,
            top_k_accuracy=top_k_accuracy,
            tau=tau,
            use_concept_groups=use_concept_groups,
            include_certainty=include_certainty,
        )
        if concept_map is None:
            concept_map = dict([
                (i, [i]) for i in range(n_concepts)
            ])
        self.concept_map = concept_map

        # Else we construct it here directly
        max_horizon_val = len(concept_map) if use_concept_groups else n_concepts
        self.include_probs = include_probs
        units = [
            (len(self.concept_map) if self.use_concept_groups else n_concepts) +
            n_concepts * emb_size + # Bottleneck
            n_concepts + # Prev interventions
            (n_concepts if include_probs else 0) + # Predicted probs
            (
                max_horizon_val if use_horizon and horizon_binary_representation
                else int(self.use_horizon)
            ) # Horizon
        ] + (int_model_layers or [256, 128]) + [
            len(self.concept_map) if self.use_concept_groups else n_concepts
        ]
        layers = []
        for i in range(1, len(units)):
            if int_model_use_bn:
                layers.append(
                    torch.nn.BatchNorm1d(num_features=units[i-1]),
                )
            layers.append(torch.nn.Linear(units[i-1], units[i]))
            if i != len(units) - 1:
                layers.append(torch.nn.LeakyReLU())
        # DEBUG
        self.units = units
        self.concept_rank_model = torch.nn.Sequential(*layers)
        if "flow" in ac_model_config['architecture']:
            self.ac_model = ACFlow(
                n_concepts = n_concepts,
                n_tasks = n_tasks,
                layer_cfg = ac_model_config['layer_cfg'],
                affine_hids = ac_model_config['affine_hids'],
                linear_rank = ac_model_config['linear_rank'],
                linear_hids = ac_model_config['linear_hids'],
                transformations = ac_model_config['transformations'],
                prior_units = ac_model_config['prior_units'],
                prior_layers = ac_model_config['prior_layers'],
                prior_hids = ac_model_config['prior_hids'],
                n_components = ac_model_config['n_components']
            ).to(self.device)
        elif "energy" in ac_model_config['architecture']:
            self.ac_model = ACEnergy(
                n_concepts = n_concepts,
                n_tasks = n_tasks,
                # embed_size = ac_model_config["embed_size"],
                # cy_perturb_prob = ac_model_config.get("cy_perturb_prob", None),
                # cy_perturb_prob = ac_model_config.get("", None)
            ).to(self.device)
        else:
            raise ValueError(f"AC{ac_model_config['architecture']} architecture not supported")
        if ac_model_config.get("save_path", None) is not None:
            chpt_exists = (
                os.path.exists(ac_model_config['save_path'])
            )
            if chpt_exists:
                self.ac_model.load_state_dict(torch.load(ac_model_config['save_path']))
                logging.debug(
                    f"AC CBM loaded AC model checkpoint from {ac_model_config['save_path']}"
                )
                self.train_ac_model = False
            # else:
            #     raise ValueError(f"AC{ac_model_config['architecture']} model checkpoint at {ac_model_config['save_path']} incorrect / not found")
            self.train_ac_model = False
        else:
            self.train_ac_model = True
            logging.debug(
                f"Training AC Flow model simultaneously with CEM model."
            )

        self.ac_model_nll_ratio = ac_model_nll_ratio
        self.ac_model_weight = ac_model_weight
        self.ac_model_rollouts = ac_model_rollouts
        self.ac_softmax = torch.nn.Softmax(dim = 1).to(self.device)

        self.intervention_discount = intervention_discount
        self.intervention_task_discount = intervention_task_discount
        self.horizon_rate = horizon_rate
        self.horizon_limit = torch.nn.Parameter(
            torch.FloatTensor([initial_horizon]),
            requires_grad=False,
        )
        if not self.legacy_mode:
            self.current_steps = torch.nn.Parameter(
                torch.IntTensor([0]),
                requires_grad=False,
            )
            if self.rollout_aneal_rate != 1:
                self.current_aneal_rate = torch.nn.Parameter(
                    torch.FloatTensor([1]),
                    requires_grad=False,
                )
        self.rollout_init_steps = rollout_init_steps
        self.tau = tau
        self.intervention_weight = intervention_weight
        self.average_trajectory = average_trajectory
        self.loss_interventions = torch.nn.CrossEntropyLoss()
        self.ac_model_xent_loss = torch.nn.CrossEntropyLoss()
        self.max_horizon = max_horizon
        self.include_task_trajectory_loss = include_task_trajectory_loss
        self.horizon_binary_representation = horizon_binary_representation
        self.include_only_last_trajectory_loss = \
            include_only_last_trajectory_loss
        self.intervention_task_loss_weight = intervention_task_loss_weight
        self.use_concept_groups = use_concept_groups

        self.horizon_uniform_distr = horizon_uniform_distr
        self.beta_a = beta_a
        self.beta_b = beta_b

    def _horizon_distr(self, init, end):
        if self.horizon_uniform_distr:
            return np.random.randint(init,end)
        else:
            return int(np.random.beta(self.beta_a, self.beta_b) * (end - init) + init)

    def _after_interventions(
        self,
        prob,
        pos_embeddings,
        neg_embeddings,
        intervention_idxs=None,
        c_true=None,
        train=False,
        competencies=None,
    ):
        if train and (self.training_intervention_prob != 0) and (
            (c_true is not None) and
            (intervention_idxs is None)
        ):
            # Then we will probabilistically intervene in some concepts
            if self.use_concept_groups:
                group_mask = np.random.binomial(
                    n=1,
                    p=self.training_intervention_prob,
                    size=len(self.concept_map),
                )
                mask = torch.zeros((c_true.shape[-1],)).to(c_true.device)
                for group_idx, (_, group_concepts) in enumerate(
                    self.concept_map.items()
                ):
                    if group_mask[group_idx] == 1:
                        mask[group_concepts] = 1
                intervention_idxs = torch.tile(
                    mask,
                    (c_true.shape[0], 1),
                )
            else:
                mask = torch.bernoulli(
                    self.ones * self.training_intervention_prob,
                )
                intervention_idxs = torch.tile(
                    mask,
                    (c_true.shape[0], 1),
                )
        if (c_true is None) or (intervention_idxs is None):
            return prob, intervention_idxs
        intervention_idxs = intervention_idxs.type(torch.FloatTensor)
        intervention_idxs = intervention_idxs.to(prob.device)
        return (
            prob * (1 - intervention_idxs) + intervention_idxs * c_true,
            intervention_idxs,
        )

    def _prior_int_distribution(
        self,
        prob,
        pos_embeddings,
        neg_embeddings,
        c,
        competencies=None,
        prev_interventions=None,
        horizon=1,
        train=False,
    ):
        return ACConceptBottleneckModel._prior_int_distribution(
            self=self,
            prob=prob,
            pos_embeddings=pos_embeddings,
            neg_embeddings=neg_embeddings,
            c=c,
            competencies=competencies,
            prev_interventions=prev_interventions,
            horizon=horizon,
            train=train,
        )

class AFAConceptBottleneckModel(ConceptBottleneckModel):
    def __init__(
        self,
        n_concepts,
        n_tasks,
        concept_loss_weight=0.01,
        task_loss_weight=1,

        extra_dims=0,
        bool=False,
        sigmoidal_prob=True,
        sigmoidal_extra_capacity=True,
        bottleneck_nonlinear=None,
        output_latent=False,

        x2c_model=None,
        c_extractor_arch=utils.wrap_pretrained_model(resnet50),
        c2y_model=None,
        c2y_layers=None,
        ac_model_config = {},
        afa_model_config = {},

        optimizer="adam",
        momentum=0.9,
        learning_rate=0.01,
        weight_decay=4e-05,
        weight_loss=None,
        task_class_weights=None,

        active_intervention_values=None,
        inactive_intervention_values=None,
        intervention_policy=None,
        output_interventions=False,
        include_certainty=True,

        intervention_discount=1,
        intervention_task_discount=1.1,
        intervention_weight=5,
        horizon_rate=1.005,
        tau=1,
        average_trajectory=True,
        concept_map=None,
        use_concept_groups=False,
        max_horizon=5,
        include_task_trajectory_loss=False,
        include_only_last_trajectory_loss=False,
        horizon_binary_representation=False,
        intervention_task_loss_weight=1,
        initial_horizon=1,
        horizon_uniform_distr=True,
        beta_a=1,
        beta_b=3,
        use_horizon=True,
        rollout_init_steps=0,
        include_probs=False,
        use_full_mask_distr=False,
        int_model_layers=None,
        int_model_use_bn=False,
        initialize_discount=False,
        propagate_target_gradients=False,
        top_k_accuracy=None,
        num_rollouts=1,
        max_num_rollouts=None,
        rollout_aneal_rate=1,
        backprop_masks=True,
        legacy_mode=False,
        hard_intervention=True,
    ):
        self.hard_intervention = hard_intervention
        self.legacy_mode = legacy_mode
        self.num_rollouts = num_rollouts
        self.rollout_aneal_rate = rollout_aneal_rate
        self.backprop_masks = backprop_masks
        self.max_num_rollouts = max_num_rollouts
        self.propagate_target_gradients = propagate_target_gradients
        self.initialize_discount = initialize_discount
        self.use_horizon = use_horizon
        self.use_full_mask_distr = use_full_mask_distr
        if concept_map is None:
            concept_map = dict([
                (i, [i]) for i in range(n_concepts)
            ])
        self.concept_map = concept_map
        if len(concept_map) == n_concepts:
            use_concept_groups = False
        super(ACConceptBottleneckModel, self).__init__(
            n_concepts=n_concepts,
            n_tasks=n_tasks,
            concept_loss_weight=concept_loss_weight,
            task_loss_weight=task_loss_weight,
            extra_dims=extra_dims,
            bool=bool,
            sigmoidal_prob=sigmoidal_prob,
            sigmoidal_extra_capacity=sigmoidal_extra_capacity,
            bottleneck_nonlinear=bottleneck_nonlinear,
            output_latent=output_latent,
            x2c_model=x2c_model,
            c_extractor_arch=c_extractor_arch,
            c2y_model=c2y_model,
            c2y_layers=c2y_layers,
            optimizer=optimizer,
            momentum=momentum,
            learning_rate=learning_rate,
            weight_decay=weight_decay,
            weight_loss=weight_loss,
            task_class_weights=task_class_weights,
            active_intervention_values=active_intervention_values,
            inactive_intervention_values=inactive_intervention_values,
            intervention_policy=intervention_policy,
            output_interventions=output_interventions,
            top_k_accuracy=top_k_accuracy,
            use_concept_groups=use_concept_groups,
            include_certainty=include_certainty,
        )

        # Else we construct it here directly
        max_horizon_val = \
            len(concept_map) if self.use_concept_groups else n_concepts
        self.include_probs = include_probs
        units = [
            (n_concepts if self.use_concept_groups else len(self.concept_map)) * 2 +
            n_concepts + # Bottleneck
            n_concepts + # Prev interventions
            (n_concepts if self.include_probs else 0) + # Predicted Probs
            (
                max_horizon_val if use_horizon and horizon_binary_representation
                else int(use_horizon)
            ) # Horizon
        ] + (int_model_layers or [256, 128]) + [
            len(self.concept_map) if self.use_concept_groups else n_concepts
        ]
        layers = []
        for i in range(1, len(units)):
            if int_model_use_bn:
                layers.append(
                    torch.nn.BatchNorm1d(num_features=units[i-1]),
                )
            layers.append(torch.nn.Linear(units[i-1], units[i]))
            if i != len(units) - 1:
                layers.append(torch.nn.LeakyReLU())
        self.concept_rank_model = torch.nn.Sequential(*layers)
        
        try:
            self.ac_model = ACFlow.load_from_checkpoint(checkpoint_path = ac_model_config['save_path'])
            logging.debug(
                f"AC CBM loaded AC model checkpoint from {ac_model_config['save_path']}"
                f"AC model trained with {self.ac_model.current_epoch} epochs"
            )
            self.train_ac_model = False
        except:
            raise ValueError(f"Pretrained ACFlow model checkpoint at {ac_model_config['save_path']} incorrect / not found")

        self.intervention_discount = intervention_discount
        self.intervention_task_discount = intervention_task_discount
        self.horizon_rate = horizon_rate
        self.horizon_limit = torch.nn.Parameter(
            torch.FloatTensor([initial_horizon]),
            requires_grad=False,
        )
        if not legacy_mode:
            self.current_steps = torch.nn.Parameter(
                torch.IntTensor([0]),
                requires_grad=False,
            )
            if self.rollout_aneal_rate != 1:
                self.current_aneal_rate = torch.nn.Parameter(
                    torch.FloatTensor([1]),
                    requires_grad=False,
                )
        self.rollout_init_steps = rollout_init_steps
        self.tau = tau
        self.intervention_weight = intervention_weight
        self.average_trajectory = average_trajectory
        self.loss_interventions = torch.nn.CrossEntropyLoss()
        self.ac_model_xent_loss = torch.nn.CrossEntropyLoss(weight=task_class_weights) if n_tasks > 1 else torch.nn.BCEWithLogitsLoss(
            pos_weight=task_class_weights 
        )
        self.max_horizon = max_horizon
        self.emb_size = 1
        self.include_task_trajectory_loss = include_task_trajectory_loss
        self.horizon_binary_representation = horizon_binary_representation
        self.include_only_last_trajectory_loss = \
            include_only_last_trajectory_loss
        self.intervention_task_loss_weight = intervention_task_loss_weight

        if horizon_uniform_distr:
            self._horizon_distr = lambda init, end: np.random.randint(
                init,
                end,
            )
        else:
            self._horizon_distr = lambda init, end: int(
                np.random.beta(beta_a, beta_b) * (end - init) + init
            )


    def get_concept_int_distribution(
        self,
        x,
        c,
        prev_interventions=None,
        competencies=None,
        horizon=1,
    ):
        if prev_interventions is None:
            prev_interventions = torch.zeros(c.shape).to(x.device)
        outputs = self._forward(
            x,
            c=c,
            y=None,
            train=False,
            competencies=competencies,
            prev_interventions=prev_interventions,
            output_embeddings=True,
            output_latent=True,
            output_interventions=True,
        )

        c_sem, c_logits, y_logits = outputs[0], outputs[1], outputs[2]
        prev_interventions = outputs[3]
        pos_embeddings = outputs[-2]
        neg_embeddings = outputs[-1]
        return self._prior_int_distribution(
            prob=c_sem,
            pos_embeddings=pos_embeddings,
            neg_embeddings=neg_embeddings,
            c=c,
            competencies=competencies,
            prev_interventions=prev_interventions,
            horizon=horizon,
            train=False,
        )


    def _prior_int_distribution(
        self,
        prob,
        pos_embeddings,
        neg_embeddings,
        c,
        competencies=None,
        prev_interventions=None,
        horizon=1,
        train=False,
    ):
        if prev_interventions is None:
            prev_interventions = torch.zeros(prob.shape).to(
                pos_embeddings.device
            )
        if competencies is None:
            competencies = torch.ones(prob.shape).to(pos_embeddings.device)
        # Shape is [B, n_concepts, emb_size]
        prob = prev_interventions * c + (1 - prev_interventions) * prob
        embeddings = (
            torch.unsqueeze(prob, dim=-1) * pos_embeddings +
            (1 - torch.unsqueeze(prob, dim=-1)) * neg_embeddings
        )
        # Zero out embeddings of previously intervened concepts
        if self.use_concept_groups:
            available_groups = torch.zeros(
                (embeddings.shape[0], len(self.concept_map))
            ).to(
                embeddings.device
            )
            for group_idx, (_, group_concepts) in enumerate(
                self.concept_map.items()
            ):
                available_groups[:, group_idx] = torch.logical_not(torch.any(
                    prev_interventions[:, group_concepts] == 1,
                    dim=-1,
                ))
            max_horizon = len(self.concept_map)
        else:
            available_groups = (1 - prev_interventions).to(embeddings.device)
            max_horizon = self.n_concepts
        used_groups = 1 - available_groups
        
        num_groups = int(torch.sum(available_groups[0]).detach())

        unintervened_groups = [torch.nonzero(available_groups[i], as_tuple = False).squeeze(dim = 1) for i in range(available_groups.shape[0])]

        try:
            unintervened_groups = torch.stack(unintervened_groups, dim = 0)
        except:
            max_length = max([t.size(0) for t in unintervened_groups])
            min_length = min([t.size(0) for t in unintervened_groups])
            padded_list = [t if t.numel() == max_length else torch.cat([t, \
                t[torch.multinomial(torch.ones_like(t, dtype = torch.float) / t.numel(), num_samples=max_length - t.numel(), replacement=True)]]) \
            for t in unintervened_groups]

            unintervened_groups = torch.stack(padded_list)
            num_groups = max_length
                    
        logpus_sparse = torch.zeros(used_groups.shape, dtype = torch.float32, device = used_groups.device)
        logpos_sparse = torch.zeros(used_groups.shape, dtype = torch.float32, device = used_groups.device)

        mask = prev_interventions.clone().float()
        missing = prev_interventions.clone().float()
        predicted_and_intervened_concepts = prob.clone()
        concept_map_vals = list(self.concept_map.values())
        for i in range(num_groups):
            for b in range(used_groups.shape[0]):
                for concept in concept_map_vals[int(unintervened_groups[b][i])]:
                    missing[b][concept] = 1.
            logpu, logpo, _, _, _ = self.ac_model(x = predicted_and_intervened_concepts, b = mask, m = missing, y = None)
            pu = torch.logsumexp(logpu, dim = -1)
            po = torch.logsumexp(logpo, dim = -1)
            batches = torch.arange(used_groups.shape[0])
            indices = unintervened_groups[batches, i].cpu()
            logpus_sparse[batches, indices] = pu[batches]            
            logpos_sparse[batches, indices] = po[batches]

            for b in range(used_groups.shape[0]):
                for concept in concept_map_vals[int(unintervened_groups[b][i])]:
                    missing[b][concept] = 0.
        cat_inputs = [
            logpus_sparse,
            logpos_sparse,
            torch.reshape(embeddings, [-1, self.emb_size * self.n_concepts]),
            prev_interventions,
#                 competencies,
        ]
        if self.include_probs:
            cat_inputs.append(prob)
        if self.use_horizon:
            cat_inputs.append((
                torch.ones(
                    [embeddings.shape[0], 1]
                ).to(prev_interventions) * horizon
                if (not self.horizon_binary_representation) else
                torch.concat(
                    [
                        torch.ones([embeddings.shape[0], horizon]),
                        torch.zeros(
                            [embeddings.shape[0], max_horizon - horizon]
                        ),
                    ],
                    dim=-1
                ).to(prev_interventions)
            ))
        rank_input = torch.concat(
            cat_inputs,
            dim=-1,
        )
        next_concept_group_scores = self.concept_rank_model(
            rank_input
        )
        if train:
            return next_concept_group_scores
        next_concept_group_scores = torch.where(
            used_groups == 1,
            torch.ones(used_groups.shape).to(used_groups.device) * (-1000),
            next_concept_group_scores,
        )
        return torch.nn.functional.softmax(
            next_concept_group_scores/self.tau,
            dim=-1,
        )

    def _run_step(
        self,
        batch,
        batch_idx,
        train=False,
        intervention_idxs=None,
    ):
        x, y, (c, competencies, prev_interventions) = self._unpack_batch(batch)
        outputs = self._forward(
            x,
            intervention_idxs=intervention_idxs,
            c=c,
            y=y,
            train=train,
            competencies=competencies,
            prev_interventions=prev_interventions,
            output_embeddings=True,
            output_latent=True,
            output_interventions=True,
        )
        c_sem, c_logits, y_logits = outputs[0], outputs[1], outputs[2]
        prev_interventions = outputs[3]
        latent = outputs[4]
        pos_embeddings = outputs[-2]
        neg_embeddings = outputs[-1]

        if self.task_loss_weight != 0:
            task_loss = self.loss_task(
                y_logits if y_logits.shape[-1] > 1 else y_logits.reshape(-1),
                y,
            )
            task_loss_scalar = self.task_loss_weight * task_loss.detach()
        else:
            task_loss = 0.0
            task_loss_scalar = 0.0

        ac_model_loss = 0.0
        ac_model_loss_scalar = 0.0
        # Do some rollouts for flow model
        if self.ac_model_weight != 0 and train and self.train_ac_model:
            if self.rollout_aneal_rate != 1:
                ac_model_rollouts = int(round(
                    self.ac_model_rollouts * (
                        self.current_aneal_rate.detach().cpu().numpy()[0]
                    )
                ))
            else:
                ac_model_rollouts = self.ac_model_rollouts

            for _ in range(ac_model_rollouts):
                x_ac, b_ac, m_ac, y_ac = ACTransformDataset.transform_batch(c, y)
                logpu, logpo, _, _, _ = self.ac_model(x_ac, b_ac, m_ac, y_ac)
                logits = logpu + logpo
                loglikel = torch.logsumexp(logpu + logpo, dim = 1) - torch.logsumexp(logpo, dim = 1)
                nll = torch.mean(-loglikel)
                ac_model_loss += (1 - self.ac_model_nll_ratio) * self.ac_model_xent_loss(logits, y_ac) + self.ac_model_nll_ratio * nll
            

            ac_model_loss = ac_model_loss / ac_model_rollouts
            ac_model_loss_scalar = ac_model_loss.detach() * self.ac_model_weight

        else:
            self.ac_model.freeze()

        intervention_task_loss = 0.0

        # Now we will do some rolls for interventions
        int_mask_accuracy = -1.0
        current_horizon = -1
        if not self.include_certainty:
            c_used = torch.where(
                torch.logical_or(c == 0, c == 1),
                c,
                c_sem.detach(),
            )
        else:
            c_used = c
        if (
            train and
            (intervention_idxs is None)
        ):
            intervention_loss = 0.0
            if prev_interventions is not None:
                # This will be not None in the case of RandInt, so we can ASSUME
                # they have all been intervened the same number of times before
                intervention_idxs = prev_interventions[:]
                if self.use_concept_groups:
                    free_groups = torch.ones(
                        (prev_interventions.shape[0], len(self.concept_map))
                    ).to(
                        c_used.device
                    )
                    # BIG ASSUMPTION: THEY HAVE ALL BEEN INTERVENED ON THE SAME
                    # NUMBER OF CONCEPTS
                    cpu_ints = intervention_idxs[0, :].detach().cpu().numpy()
                    for group_idx, (_, group_concepts) in enumerate(
                        self.concept_map.items()
                    ):
                        if np.any(cpu_ints[group_concepts] == 1):
                            free_groups[:,group_idx] = 0
                    prev_num_of_interventions = int(
                        len(self.concept_map) - np.sum(
                            free_groups[0, :].detach().cpu().numpy(),
                            axis=-1,
                        ),
                    )
                else:
                    free_groups = 1 - intervention_idxs
                    prev_num_of_interventions = int(np.max(
                        np.sum(
                            intervention_idxs.detach().cpu().numpy(),
                            axis=-1,
                        ),
                        axis=-1,
                    ))
            else:
                intervention_idxs = torch.zeros(c_used.shape).to(c_used.device)
                prev_num_of_interventions = 0
                if self.use_concept_groups:
                    free_groups = torch.ones(
                        (c_used.shape[0], len(self.concept_map))
                    ).to(c_used.device)
                else:
                    free_groups = torch.ones(c_used.shape).to(c_used.device)

            # Then time to perform a forced training time intervention
            # We will set some of the concepts as definitely intervened on
            if competencies is None:
                competencies = torch.FloatTensor(
                    c_used.shape
                ).uniform_(0.5, 1).to(c_used.device)
            int_basis_lim = (
                len(self.concept_map) if self.use_concept_groups
                else self.n_concepts
            )
            horizon_lim = int(self.horizon_limit.detach().cpu().numpy()[0])
            if prev_num_of_interventions != int_basis_lim:
                bottom = min(
                    horizon_lim,
                    int_basis_lim - prev_num_of_interventions - 1,
                )  # -1 so that we at least intervene on one concept
                if bottom > 0:
                    initially_selected = np.random.randint(0, bottom)
                else:
                    initially_selected = 0
                end_horizon = min(
                    int(horizon_lim),
                    self.max_horizon,
                    int_basis_lim - prev_num_of_interventions - initially_selected,
                )
                current_horizon = self._horizon_distr(
                    init=1 if end_horizon > 1 else 0,
                    end=end_horizon,
                )

                for sample_idx in range(intervention_idxs.shape[0]):
                    probs = free_groups[sample_idx, :].detach().cpu().numpy()
                    probs = probs/np.sum(probs)
                    if self.use_concept_groups:
                        selected_groups = set(np.random.choice(
                            int_basis_lim,
                            size=initially_selected,
                            replace=False,
                            p=probs,
                        ))
                        for group_idx, (_, group_concepts) in enumerate(
                            self.concept_map.items()
                        ):
                            if group_idx in selected_groups:
                                intervention_idxs[sample_idx, group_concepts] = 1
                    else:
                        intervention_idxs[
                            sample_idx,
                            np.random.choice(
                                int_basis_lim,
                                size=initially_selected,
                                replace=False,
                                p=probs,
                            )
                        ] = 1
                if self.initialize_discount:
                    discount = (
                        self.intervention_discount ** prev_num_of_interventions
                    )
                    task_discount = self.intervention_task_discount ** (
                        prev_num_of_interventions + initially_selected
                    )
                    first_task_discount = (
                        self.intervention_task_discount **
                        prev_num_of_interventions
                    )
                else:
                    discount = 1
                    task_discount = 1
                    first_task_discount = 1

                trajectory_weight = 0
                task_trajectory_weight = 1
                if self.average_trajectory:
                    curr_discount = discount
                    curr_task_discount = task_discount
                    for i in range(current_horizon):
                        trajectory_weight += discount
                        discount *= self.intervention_discount
                        task_discount *= self.intervention_task_discount
                        if (
                            (not self.include_only_last_trajectory_loss) or
                            (i == current_horizon - 1)
                        ):
                            task_trajectory_weight += task_discount
                        
                    discount = curr_discount
                    task_discount = curr_task_discount
                else:
                    trajectory_weight = 1

                if self.n_tasks > 1:
                    one_hot_y = torch.nn.functional.one_hot(y, self.n_tasks)
            else:
                current_horizon = 0
                if self.initialize_discount:
                    task_discount = self.intervention_task_discount ** (
                        prev_num_of_interventions
                    )
                    first_task_discount = (
                        self.intervention_task_discount **
                        prev_num_of_interventions
                    )
                else:
                    task_discount = 1
                    first_task_discount = 1
                task_trajectory_weight = 1
                trajectory_weight = 1

            if self.include_task_trajectory_loss and (
                self.task_loss_weight == 0
            ):
                # Then we initialize the intervention trajectory task loss to
                # that of the unintervened model as this loss is not going to
                # be taken into account if we don't do this
                intervention_task_loss = first_task_discount * self.loss_task(
                    (
                        y_logits if y_logits.shape[-1] > 1
                        else y_logits.reshape(-1)
                    ),
                    y,
                )
                if self.average_trajectory:
                    intervention_task_loss = (
                        intervention_task_loss / task_trajectory_weight
                    )
            int_mask_accuracy = 0.0 if current_horizon else -1
            if (not self.legacy_mode) and (
                (
                    self.current_steps.detach().cpu().numpy()[0] <
                    self.rollout_init_steps
                )
            ):
                current_horizon = 0
            if self.rollout_aneal_rate != 1:
                num_rollouts = int(round(
                    self.num_rollouts * (
                        self.current_aneal_rate.detach().cpu().numpy()[0]
                    )
                ))
            else:
                num_rollouts = self.num_rollouts
            if self.max_num_rollouts is not None:
                num_rollouts = min(num_rollouts, self.max_num_rollouts)

            for i in range(num_rollouts):
                self.current_horizon = current_horizon
                for j in range(current_horizon):
                    # And generate a probability distribution over previously
                    # unseen concepts to indicate which one we should intervene
                    # on next!
                    self.horizon_index = j
                    concept_group_scores = self._prior_int_distribution(
                        prob=c_sem,
                        pos_embeddings=pos_embeddings,
                        neg_embeddings=neg_embeddings,
                        competencies=competencies,
                        prev_interventions=intervention_idxs,
                        c=c_used,
                        horizon=(current_horizon - i),
                        train=train,
                    )
                    # Generate as a label the concept which increases the
                    # probability of the correct class the most when
                    # intervened on
                    target_int_logits = torch.ones(
                        concept_group_scores.shape,
                    ).to(c_used.device) * (-np.Inf)
                    for target_concept in range(target_int_logits.shape[-1]):
                        if self.use_concept_groups:
                            new_int = torch.zeros(
                                intervention_idxs.shape
                            ).to(intervention_idxs.device)
                            for group_idx, (_, group_concepts) in enumerate(
                                self.concept_map.items()
                            ):
                                if group_idx == target_concept:
                                    new_int[:, group_concepts] = 1
                                    break
                        else:
                            new_int = torch.zeros(
                                intervention_idxs.shape
                            ).to(intervention_idxs.device)
                            new_int[:, target_concept] = 1
                        if not self.propagate_target_gradients:
                            partial_ints = torch.clamp(
                                intervention_idxs.detach() + new_int,
                                0,
                                1,
                            )
                            probs = (
                                c_sem.detach() * (1 - partial_ints) +
                                c_used * partial_ints
                            )
                            c_rollout_pred = (
                                (
                                    pos_embeddings.detach() *
                                    torch.unsqueeze(probs, dim=-1)
                                ) +
                                (
                                    neg_embeddings.detach() *
                                    (1 - torch.unsqueeze(probs, dim=-1))
                                )
                            )
                        else:
                            partial_ints = torch.clamp(
                                intervention_idxs + new_int,
                                0,
                                1,
                            )
                            probs = (
                                c_sem * (1 - partial_ints) +
                                c_used * partial_ints
                            )
                            c_rollout_pred = (
                                (
                                    pos_embeddings *
                                    torch.unsqueeze(probs, dim=-1)
                                ) + (
                                    neg_embeddings *
                                    (1 - torch.unsqueeze(probs, dim=-1))
                                )
                            )
                        c_rollout_pred = c_rollout_pred.view(
                            (-1, self.emb_size * self.n_concepts)
                        )

                        rollout_y_logits = self.c2y_model(c_rollout_pred)

                        if self.n_tasks > 1:
                            target_int_logits[:, target_concept] = \
                                rollout_y_logits[
                                    one_hot_y.type(torch.BoolTensor)
                                ]
                        else:
                            pred_y_prob = torch.sigmoid(
                                torch.squeeze(rollout_y_logits, dim=-1)
                            )
                            target_int_logits[:, target_concept] = torch.where(
                                y == 1,
                                torch.log(
                                    (pred_y_prob + 1e-15) /
                                    (1 - pred_y_prob + 1e-15)
                                ),
                                torch.log(
                                    (1 - pred_y_prob + 1e-15) /
                                    (pred_y_prob+ 1e-15)
                                ),
                            )
                    if self.use_full_mask_distr:
                        target_int_labels = torch.nn.functional.softmax(
                            target_int_logits,
                            -1,
                        )
                        pred_int_labels = concept_group_scores.argmax(-1)
                        curr_acc = (
                            pred_int_labels == torch.argmax(
                                target_int_labels,
                                -1,
                            )
                        ).float().mean()
                    else:
                        target_int_labels = torch.argmax(target_int_logits, -1)
                        pred_int_labels = concept_group_scores.argmax(-1)
                        curr_acc = (
                            pred_int_labels == target_int_labels
                        ).float().mean()

                    int_mask_accuracy += curr_acc/current_horizon
                    new_loss = self.loss_interventions(
                        concept_group_scores,
                        target_int_labels,
                    )
                    if not self.backprop_masks:
                        # Then block the gradient into the masks
                        concept_group_scores = concept_group_scores.detach()

                    # Update the next-concept predictor loss
                    if self.average_trajectory:
                        intervention_loss += (
                            discount * new_loss/trajectory_weight
                        )
                    else:
                        intervention_loss += discount * new_loss

                    # Update the discount (before the task trajectory loss to
                    # start discounting from the first intervention so that the
                    # loss of the unintervened model is highest
                    discount *= self.intervention_discount
                    task_discount *= self.intervention_task_discount

                    # Sample the next concepts we will intervene on using a hard
                    # Gumbel softmax
                    if self.intervention_weight == 0:
                        selected_groups = torch.FloatTensor(
                            np.eye(concept_group_scores.shape[-1])[np.random.choice(
                                concept_group_scores.shape[-1],
                                size=concept_group_scores.shape[0]
                            )]
                        ).to(concept_group_scores.device)
                    else:
                        selected_groups = torch.nn.functional.gumbel_softmax(
                            concept_group_scores,
                            dim=-1,
                            hard=self.hard_intervention,
                            tau=self.tau,
                        )
                    if self.use_concept_groups:
                        prev_intervention_idxs = intervention_idxs.clone()
                        for sample_idx in range(intervention_idxs.shape[0]):
                            for group_idx, (_, group_concepts) in enumerate(
                                self.concept_map.items()
                            ):
                                if selected_groups[sample_idx, group_idx] == 1:
                                    intervention_idxs[
                                        sample_idx,
                                        group_concepts,
                                    ] = 1
                    else:
                        intervention_idxs += selected_groups

                    if self.include_task_trajectory_loss and (
                        (not self.include_only_last_trajectory_loss) or
                        (i == (current_horizon - 1))
                    ):
                        # Then we will also update the task loss with the loss
                        # of performing this intervention!
                        probs = (
                            c_sem * (1 - intervention_idxs) +
                            c_used * intervention_idxs
                        )
                        c_rollout_pred = (
                            pos_embeddings * torch.unsqueeze(probs, dim=-1) + (
                                neg_embeddings * (
                                    1 - torch.unsqueeze(probs, dim=-1)
                                )
                            )
                        )
                        c_rollout_pred = c_rollout_pred.view(
                            (-1, self.emb_size * self.n_concepts)
                        )
                        rollout_y_logits = self.c2y_model(c_rollout_pred)
                        rollout_y_loss = self.loss_task(
                            (
                                rollout_y_logits
                                if rollout_y_logits.shape[-1] > 1 else
                                rollout_y_logits.reshape(-1)
                            ),
                            y,
                        )
                        if self.average_trajectory:
                            intervention_task_loss += (
                                task_discount *
                                rollout_y_loss / task_trajectory_weight
                            )
                        else:
                            intervention_task_loss += (
                                task_discount * rollout_y_loss
                            )

                if (
                    self.legacy_mode or
                    (
                        self.current_steps.detach().cpu().numpy()[0] >=
                            self.rollout_init_steps
                    )
                ) and (
                    self.horizon_limit.detach().cpu().numpy()[0] <
                        int_basis_lim + 1
                ):
                    self.horizon_limit *= self.horizon_rate

            intervention_loss_scalar = \
                self.intervention_weight * intervention_loss
            intervention_loss = intervention_loss/num_rollouts
            intervention_task_loss = intervention_task_loss/num_rollouts
            int_mask_accuracy = int_mask_accuracy/num_rollouts
        else:
            intervention_loss = 0.0
            intervention_loss_scalar = 0.0

        if not self.legacy_mode:
            self.current_steps += 1
            if self.rollout_aneal_rate != 1 and (
                self.current_aneal_rate.detach().cpu().numpy()[0] < 100
            ):
                self.current_aneal_rate *= self.rollout_aneal_rate

        if self.include_task_trajectory_loss and (
            self.intervention_task_loss_weight != 0
        ):
            if isinstance(intervention_task_loss, float):
                intervention_task_loss_scalar = (
                    self.intervention_task_loss_weight * intervention_task_loss
                )
            else:
                intervention_task_loss_scalar = (
                    self.intervention_task_loss_weight *
                    intervention_task_loss.detach()
                )
        else:
            intervention_task_loss_scalar = 0.0


        if self.concept_loss_weight != 0:
            # We separate this so that we are allowed to
            # use arbitrary activations (i.e., not necessarily in [0, 1])
            # whenever no concept supervision is provided
            if self.include_certainty:
                concept_loss = self.loss_concept(c_sem, c)
                concept_loss_scalar = \
                    self.concept_loss_weight * concept_loss.detach()
            else:
                c_sem_used = torch.where(
                    torch.logical_or(c == 0, c == 1),
                    c_sem,
                    c,
                ) # This forces zero loss when c is uncertain
                concept_loss = self.loss_concept(c_sem_used, c)
                concept_loss_scalar = concept_loss.detach()
        else:
            concept_loss = 0.0
            concept_loss_scalar = 0.0

        loss = (
            self.concept_loss_weight * concept_loss +
            self.intervention_weight * intervention_loss +
            self.ac_model_weight * ac_model_loss +
            self.task_loss_weight * task_loss +
            self.intervention_task_loss_weight * intervention_task_loss
        )

        loss += self._extra_losses(
            x=x,
            y=y,
            c=c,
            c_sem=c_sem,
            c_pred=c_logits,
            y_pred=y_logits,
            competencies=competencies,
            prev_interventions=prev_interventions,
        )
        # compute accuracy
        (c_accuracy, c_auc, c_f1), (y_accuracy, y_auc, y_f1) = compute_accuracy(
            c_sem,
            y_logits,
            c,
            y,
        )
        result = {
            "c_accuracy": c_accuracy,
            "c_auc": c_auc,
            "c_f1": c_f1,
            "y_accuracy": y_accuracy,
            "y_auc": y_auc,
            "y_f1": y_f1,
            "mask_accuracy": int_mask_accuracy,
            "concept_loss": concept_loss_scalar,
            "task_loss": task_loss_scalar,
            "intervention_task_loss": intervention_task_loss_scalar,
            "intervention_loss": intervention_loss_scalar,
            "ac_model_loss": ac_model_loss_scalar,
            "loss": loss.detach() if not isinstance(loss, float) else loss,
            "avg_c_y_acc": (c_accuracy + y_accuracy) / 2,
            "horizon_limit": self.horizon_limit.detach().cpu().numpy()[0],
        }
        if not self.legacy_mode:
            result["current_steps"] = \
                self.current_steps.detach().cpu().numpy()[0]
            if self.rollout_aneal_rate != 1:
                num_rollouts = int(round(
                    self.num_rollouts * (
                        self.current_aneal_rate.detach().cpu().numpy()[0]
                    )
                ))
                if self.max_num_rollouts is not None:
                    num_rollouts = min(num_rollouts, self.max_num_rollouts)
                result["num_rollouts"] = num_rollouts

        if self.top_k_accuracy is not None:
            y_true = y.reshape(-1).cpu().detach()
            y_pred = y_logits.cpu().detach()
            labels = list(range(self.n_tasks))
            for top_k_val in self.top_k_accuracy:
                y_top_k_accuracy = sklearn.metrics.top_k_accuracy_score(
                    y_true,
                    y_pred,
                    k=top_k_val,
                    labels=labels,
                )
                result[f'y_top_{top_k_val}_accuracy'] = y_top_k_accuracy
        return loss, result

class AFAConceptEmbeddingModel(
    ConceptEmbeddingModel,
    AFAConceptBottleneckModel,
):
    def __init__(
        self,
        n_concepts,
        n_tasks,
        emb_size=16,
        training_intervention_prob=0.25,
        embedding_activation="leakyrelu",
        concept_loss_weight=1,
        task_loss_weight=1,

        c2y_model=None,
        c2y_layers=None,
        c_extractor_arch=utils.wrap_pretrained_model(resnet50),
        output_latent=False,

        ac_model_config = {},
        ac_model_nll_ratio = 0.5,
        ac_model_weight = 2,
        ac_model_rollouts = 1,

        optimizer="adam",
        momentum=0.9,
        learning_rate=0.01,
        weight_decay=4e-05,
        weight_loss=None,
        task_class_weights=None,

        active_intervention_values=None,
        inactive_intervention_values=None,
        intervention_policy=None,
        output_interventions=False,
        include_certainty=True,

        top_k_accuracy=None,

        intervention_discount=0.9,
        intervention_task_discount=0.9,
        intervention_weight=5,
        horizon_rate=1.005,
        tau=1,
        average_trajectory=True,
        concept_map=None,
        use_concept_groups=False,
        max_horizon=5,
        include_task_trajectory_loss=False,
        horizon_binary_representation=False,
        include_only_last_trajectory_loss=False,
        intervention_task_loss_weight=1,
        initial_horizon=1,
        horizon_uniform_distr=True,
        beta_a=1,
        beta_b=3,
        use_horizon=True,
        rollout_init_steps=0,
        include_probs=False,
        use_full_mask_distr=False,
        int_model_layers=None,
        int_model_use_bn=False,
        initialize_discount=False,
        propagate_target_gradients=False,
        num_rollouts=1,
        max_num_rollouts=None,
        rollout_aneal_rate=1,
        backprop_masks=True,
        legacy_mode=False,
        hard_intervention=True,
    ):
        self.hard_intervention = hard_intervention
        self.legacy_mode = legacy_mode
        self.num_rollouts = num_rollouts
        self.backprop_masks = backprop_masks
        self.rollout_aneal_rate = rollout_aneal_rate
        self.max_num_rollouts = max_num_rollouts
        self.propagate_target_gradients = propagate_target_gradients
        self.initialize_discount = initialize_discount
        self.use_full_mask_distr = use_full_mask_distr
        self.use_horizon = use_horizon
        if concept_map is None:
            concept_map = dict([
                (i, [i]) for i in range(n_concepts)
            ])
        self.concept_map = concept_map
        if len(concept_map) == n_concepts:
            use_concept_groups = False
        ConceptEmbeddingModel.__init__(
            self,
            n_concepts=n_concepts,
            n_tasks=n_tasks,
            emb_size=emb_size,
            training_intervention_prob=training_intervention_prob,
            embedding_activation=embedding_activation,
            shared_prob_gen=False,
            concept_loss_weight=concept_loss_weight,
            task_loss_weight=task_loss_weight,
            c2y_model=c2y_model,
            c2y_layers=c2y_layers,
            c_extractor_arch=c_extractor_arch,
            output_latent=output_latent,
            optimizer=optimizer,
            momentum=momentum,
            learning_rate=learning_rate,
            weight_decay=weight_decay,
            weight_loss=weight_loss,
            task_class_weights=task_class_weights,
            active_intervention_values=active_intervention_values,
            inactive_intervention_values=inactive_intervention_values,
            intervention_policy=intervention_policy,
            output_interventions=output_interventions,
            top_k_accuracy=top_k_accuracy,
            tau=tau,
            use_concept_groups=use_concept_groups,
            include_certainty=include_certainty,
        )
        if concept_map is None:
            concept_map = dict([
                (i, [i]) for i in range(n_concepts)
            ])
        self.concept_map = concept_map

        # Else we construct it here directly
        max_horizon_val = len(concept_map) if use_concept_groups else n_concepts
        self.include_probs = include_probs
        units = [
            (len(self.concept_map) if self.use_concept_groups else n_concepts) * 2 +
            n_concepts * emb_size + # Bottleneck
            n_concepts + # Prev interventions
            (n_concepts if include_probs else 0) + # Predicted probs
            (
                max_horizon_val if use_horizon and horizon_binary_representation
                else int(self.use_horizon)
            ) # Horizon
        ] + (int_model_layers or [256, 128]) + [
            len(self.concept_map) if self.use_concept_groups else n_concepts
        ]
        layers = []
        for i in range(1, len(units)):
            if int_model_use_bn:
                layers.append(
                    torch.nn.BatchNorm1d(num_features=units[i-1]),
                )
            layers.append(torch.nn.Linear(units[i-1], units[i]))
            if i != len(units) - 1:
                layers.append(torch.nn.LeakyReLU())
        # DEBUG
        self.units = units
        self.concept_rank_model = torch.nn.Sequential(*layers)

        if ac_model_config.get("save_path", None) is not None:
            try:
                self.ac_model = ACFlow.load_from_checkpoint(checkpoint_path = ac_model_config['save_path'])
                logging.debug(
                    f"AC CBM loaded AC model checkpoint from {ac_model_config['save_path']}"
                    f"AC model trained with {self.ac_model.current_epoch} epochs"
                )
                self.train_ac_model = False
            except:
                # raise ValueError(f"ACFlow model checkpoint at {ac_model_config['save_path']} incorrect / not found")
                self.train_ac_model = False
        else:
            self.ac_model = ACFlow(
                n_concepts = n_concepts,
                n_tasks = n_tasks,
                layer_cfg = ac_model_config['layer_cfg'],
                affine_hids = ac_model_config['affine_hids'],
                linear_rank = ac_model_config['linear_rank'],
                linear_hids = ac_model_config['linear_hids'],
                transformations = ac_model_config['transformations'],
                prior_units = ac_model_config['prior_units'],
                prior_layers = ac_model_config['prior_layers'],
                prior_hids = ac_model_config['prior_hids'],
                n_components = ac_model_config['n_components']
            )
            self.train_ac_model = True

        self.ac_model_nll_ratio = ac_model_nll_ratio
        self.ac_model_weight = ac_model_weight
        self.ac_model_rollouts = ac_model_rollouts

        self.intervention_discount = intervention_discount
        self.intervention_task_discount = intervention_task_discount
        self.horizon_rate = horizon_rate
        self.horizon_limit = torch.nn.Parameter(
            torch.FloatTensor([initial_horizon]),
            requires_grad=False,
        )
        if not self.legacy_mode:
            self.current_steps = torch.nn.Parameter(
                torch.IntTensor([0]),
                requires_grad=False,
            )
            if self.rollout_aneal_rate != 1:
                self.current_aneal_rate = torch.nn.Parameter(
                    torch.FloatTensor([1]),
                    requires_grad=False,
                )
        self.rollout_init_steps = rollout_init_steps
        self.tau = tau
        self.intervention_weight = intervention_weight
        self.average_trajectory = average_trajectory
        self.loss_interventions = torch.nn.CrossEntropyLoss()
        self.ac_model_xent_loss = torch.nn.CrossEntropyLoss()
        self.max_horizon = max_horizon
        self.include_task_trajectory_loss = include_task_trajectory_loss
        self.horizon_binary_representation = horizon_binary_representation
        self.include_only_last_trajectory_loss = \
            include_only_last_trajectory_loss
        self.intervention_task_loss_weight = intervention_task_loss_weight
        self.use_concept_groups = use_concept_groups

        if horizon_uniform_distr:
            self._horizon_distr = lambda init, end: np.random.randint(
                init,
                end,
            )
        else:
            self._horizon_distr = lambda init, end: int(
                np.random.beta(beta_a, beta_b) * (end - init) + init
            )

    def _after_interventions(
        self,
        prob,
        pos_embeddings,
        neg_embeddings,
        intervention_idxs=None,
        c_true=None,
        train=False,
        competencies=None,
    ):
        if train and (self.training_intervention_prob != 0) and (
            (c_true is not None) and
            (intervention_idxs is None)
        ):
            # Then we will probabilistically intervene in some concepts
            if self.use_concept_groups:
                group_mask = np.random.binomial(
                    n=1,
                    p=self.training_intervention_prob,
                    size=len(self.concept_map),
                )
                mask = torch.zeros((c_true.shape[-1],)).to(c_true.device)
                for group_idx, (_, group_concepts) in enumerate(
                    self.concept_map.items()
                ):
                    if group_mask[group_idx] == 1:
                        mask[group_concepts] = 1
                intervention_idxs = torch.tile(
                    mask,
                    (c_true.shape[0], 1),
                )
            else:
                mask = torch.bernoulli(
                    self.ones * self.training_intervention_prob,
                )
                intervention_idxs = torch.tile(
                    mask,
                    (c_true.shape[0], 1),
                )
        if (c_true is None) or (intervention_idxs is None):
            return prob, intervention_idxs
        intervention_idxs = intervention_idxs.type(torch.FloatTensor)
        intervention_idxs = intervention_idxs.to(prob.device)
        return (
            prob * (1 - intervention_idxs) + intervention_idxs * c_true,
            intervention_idxs,
        )

    def _prior_int_distribution(
        self,
        prob,
        pos_embeddings,
        neg_embeddings,
        c,
        competencies=None,
        prev_interventions=None,
        horizon=1,
        train=False,
    ):
        return AFAConceptBottleneckModel._prior_int_distribution(
            self=self,
            prob=prob,
            pos_embeddings=pos_embeddings,
            neg_embeddings=neg_embeddings,
            c=c,
            competencies=competencies,
            prev_interventions=prev_interventions,
            horizon=horizon,
            train=train,
        )
    
class AFAModel(pl.LightningModule):
    def __init__(self, cbm, config):
        super().__init__()
        self.cbm = cbm.to(self.device)
        self.cbm.intervention_policy = None
        self.config = config
        ac_model_config = config["ac_model_config"]
        afa_model_config = config["afa_model_config"]
        if "flow" in ac_model_config['architecture']:
            self.ac_model = ACFlow(
                n_concepts = config['n_concepts'],
                n_tasks = config['n_tasks'],
                layer_cfg = ac_model_config['layer_cfg'],
                affine_hids = ac_model_config['affine_hids'],
                linear_rank = ac_model_config['linear_rank'],
                linear_hids = ac_model_config['linear_hids'],
                transformations = ac_model_config['transformations'],
                prior_units = ac_model_config['prior_units'],
                prior_layers = ac_model_config['prior_layers'],
                prior_hids = ac_model_config['prior_hids'],
                n_components = ac_model_config['n_components']
            ).to(self.device)
        elif "energy" in ac_model_config['architecture']:
            self.ac_model = ACEnergy(
                n_concepts = config['n_concepts'],
                n_tasks = config['n_tasks'],
                # embed_size = ac_model_config["embed_size"],
                # cy_perturb_prob = ac_model_config.get("cy_perturb_prob", None),
                # cy_perturb_prob = ac_model_config.get("", None)
            ).to(self.device)
        else:
            raise ValueError(f"AC{ac_model_config['architecture']} architecture not supported")
        if ac_model_config.get("save_path", None) and os.path.exists(ac_model_config['save_path']):
            self.ac_model.load_state_dict(torch.load(ac_model_config['save_path']))
            logging.debug(
                f"AC CBM loaded AC model checkpoint from {ac_model_config['save_path']}"
            )
        # else:
        #     checkpoint_location = "" if "save_path" not in ac_model_config.keys() else f"at {ac_model_config['save_path']}"
        #     message = f"AC{ac_model_config['architecture']} model checkpoint {checkpoint_location}incorrect / not found"
        #     raise ValueError(message)
        self.ac_model.freeze()
        self.num_envs = config["afa_model_config"]["num_envs"]
        # self.env = gym.vector.make("cem/AFAEnv-v0", num_envs = self.num_envs, cbm = self.cbm, ac_model = self.ac_model, env_config = config)
        self.batch_size = config["batch_size"]
        # self.env = AFAEnv(cbm = self.cbm, ac_model = self.ac_model, env_config = config, batch_size = self.batch_size)
        
        self.train_separately = config.get("train_separately", False)
        self.train_cbm = config.get("train_cbm", True)
        self.train_rl = config.get("train_rl", True)
        self.env_budget = None
        self.use_concept_groups = self.cbm.use_concept_groups
        self.use_separate_optimizers = False
        self.cbm.cbm_aneal_rate = config.get("cbm_aneal_rate", 1.)
        self.cbm.cbm_current_aneal_rate = config.get("cbm_starting_aneal_rate", 1.)
        
        self.use_concept_groups = config["use_concept_groups"]
        self.concept_group_map = config["concept_map"]
        self.n_concept_groups = len(self.concept_group_map) if self.use_concept_groups else config["n_concepts"]
        self.n_concepts = config["n_concepts"]
        if self.concept_group_map is None:
            self.concept_group_map = {}
            for i in range(self.n_concepts):
                self.concept_group_map[i] = [i]
        self.n_tasks = config["n_tasks"]
        self.n_tasks = self.n_tasks if self.n_tasks > 1 else 2
        self.emb_size = config["emb_size"]
        self.num_rollouts = config.get("num_rollouts", 1)
        self.afa_config = config["afa_model_config"]
        self._budget = self.n_concept_groups
        self.num_interventions = 0
        self.softmax = torch.nn.Softmax(dim = -1)
        self.normalize_values = config.get("normalize_values", False)
        self.entropy = scipy.stats.entropy
        self.final_reward_ratio = config.get("final_reward_ratio", 10)
        self.intermediate_reward_ratio = config.get("intermediate_reward_ratio", 1)
        self.allow_no_action = config.get("allow_no_action", True)
        self.mask_no_action = config.get("mask_no_action", False)
        self.total_acquisition_cost = 0.01
        self.xent_loss = torch.nn.CrossEntropyLoss()
        self.model_classes = self.ac_model.model_classes

        self.use_ac_model = config.get("use_ac_model", True)
        if self.use_ac_model:
            self.observation_space_dict = {
                "remaining_budget": spaces.Box(shape = [1], dtype = np.int32, low = 0, high = self.n_concept_groups),
                "intervened_concepts_map": spaces.MultiBinary(self.n_concept_groups),
                "intervened_concepts": spaces.MultiBinary(self.n_concepts),
                "ac_model_output": spaces.Box(shape = [1], dtype = np.float32, low = -np.inf, high = np.inf),
                "ac_model_info": spaces.Box(shape = [
                    # (self.n_tasks if self.model_classes else 1) + 
                self.n_concepts * 4], dtype = np.float32, low = -np.inf, high = np.inf),
                "cbm_bottleneck": spaces.Box(shape = [self.n_concepts * self.emb_size], dtype = np.float32, low = -np.inf, high = np.inf),
                "cbm_pred_concepts": spaces.Box(shape = [self.n_concepts], dtype = np.float32, low = -np.inf, high = np.inf),
                "cbm_pred_output": spaces.Box(shape = [self.n_tasks], dtype = np.float32, low = 0, high = 1),
            }
        else:
            self.observation_space_dict = {
                "remaining_budget": spaces.Box(shape = [1], dtype = np.int32, low = 0, high = self.n_concept_groups),
                "intervened_concepts_map": spaces.MultiBinary(self.n_concept_groups),
                "intervened_concepts": spaces.MultiBinary(self.n_concepts),
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

        self.action_space = spaces.Box(shape = [self.batch_size], dtype = np.int32, low = 0, high = self.n_concept_groups + 1)
        self.single_action_space_shape = self.n_concept_groups + (1 if self.allow_no_action else 0)

        self.group_costs = np.array(([0] if self.allow_no_action else []) + [ self.total_acquisition_cost / self.n_concept_groups for _ in range(self.n_concept_groups)])

        self.agent = PPOLightningAgent(
            self.single_observation_space.shape,
            self.single_action_space_shape,
            lin_layers = afa_model_config.get("lin_layers", None),
            act_fun = afa_model_config["act_fun"],
            vf_coef = afa_model_config["vf_coef"],
            ent_coef = afa_model_config["ent_coef"],
            clip_coef = afa_model_config["clip_coef"],
            clip_vloss = afa_model_config["clip_vloss"],
            ortho_init = afa_model_config["ortho_init"],
            normalize_advantages = afa_model_config["normalize_advantages"],
        ).to(self.device)
        
        self.max_nll = 1.
        self.next_max_nll = 1.

    def get_range(self, query):
        start = 0
        end = 0
        for key, value in self.observation_space_dict.items():
            shape = value.shape
            if len(shape) > 1:
                raise ValueError("Get shape unable to flatten 2d shape")
            elif len(shape) == 0:
                if key == query:
                    return start, start + 1
                start += 1
            else:
                if key == query:
                    return start, start + shape[0]
                start += shape[0]
        
        return None, None
        

    def _get_obs(self):
        # return {
        #     "budget": np.array([self._budget]),
        #     "intervened_concepts_map": self._intervened_concepts_map,
        #     "intervened_concepts": self._intervened_concepts,
        #     "ac_model_output": self._ac_model_output,
        #     "ac_model_info": self._ac_model_info,
        #     "cbm_bottleneck": self._cbm_bottleneck,
        #     "cbm_pred_concepts": self._cbm_pred_concepts,
        #     "cbm_pred_output": self._cbm_pred_output,
        # }
        if self.use_ac_model:
            arrs = [self._remaining_budget_arr, self._intervened_concepts_map, self._intervened_concepts, self._ac_model_output, self._ac_model_info, self._cbm_bottleneck, self._cbm_pred_concepts, self._cbm_pred_output,]
        else:
            arrs = [self._remaining_budget_arr, self._intervened_concepts_map, self._intervened_concepts, self._cbm_bottleneck, self._cbm_pred_concepts, self._cbm_pred_output,]
            
        
        self._observations = np.concatenate(arrs, axis = 1)

        return self._observations

    def _get_info(self):    
        action_mask = 1 - self._intervened_concepts_map

        if self.allow_no_action:
            if self.mask_no_action:
                action_mask = np.concatenate((np.ones((self._intervened_concepts_map.shape[0], 1)), action_mask), axis = 1)
            else:
                action_mask = np.concatenate((np.zeros((self._intervened_concepts_map.shape[0], 1)), action_mask), axis = 1)

        return {
            "action_mask": action_mask,
        }


    def reset(self, budget, train, c, y, c_sem, pos_embeddings, neg_embeddings):
        
        self._budget = budget

        self._remaining_budget_arr = np.tile(np.array([self._budget], dtype = np.float32), (self.batch_size, 1))

        self._train = train
        
        self._intervened_concepts_map = np.zeros((self.batch_size, self.n_concept_groups), dtype = np.float32)
        self._intervened_concepts = np.zeros((self.batch_size, self.n_concepts), dtype = np.float32)
        
        prev_interventions = torch.tensor(self._intervened_concepts, device = self.cbm.device)
        
        # x, y, (c, competencies, _) = self.cbm._unpack_batch(self._cbm_data)
        # x = x.to(self.cbm.device)
        # y = y.to(self.cbm.device)
        # c = c.to(self.cbm.device)
        # competencies = competencies.to(self.cbm.device)
        with torch.no_grad():
            # outputs = self.cbm._forward(
            #     x,
            #     intervention_idxs=prev_interventions,
            #     c=c,
            #     y=y,
            #     train=self._train,
            #     competencies=competencies,
            #     prev_interventions=None,
            #     output_embeddings=True,
            #     output_latent=True,
            #     output_interventions=True,
            # )
            # c_sem, c_pred, y_logits = outputs[0], outputs[1], outputs[2]
            # pos_embeddings = outputs[-2]
            # neg_embeddings = outputs[-1]
            prob = c_sem # [batch_size, n_concepts]
            
            embeddings = (
                pos_embeddings.detach() * torch.unsqueeze(prob, dim=-1) + (
                    neg_embeddings.detach() * (
                        1 - torch.unsqueeze(prob, dim=-1)
                    )
                )
            ).detach().view((-1, self.cbm.emb_size * self.cbm.n_concepts))
            rollout_y_logits = self.cbm.c2y_model(embeddings)
            embeddings = embeddings.cpu().numpy()
            interventions = torch.zeros(self._intervened_concepts.shape).to(self.ac_model.device)

            # ac_model_output, logpo, _, _ = self.ac_model.compute_concept_probabilities(
            #             x = prob,
            #             b = interventions,
            #             m = interventions,
            #             y = None
            #         )

            # prob = torch.exp(ac_model_output)

            _, _, sam, pred_sam = self.ac_model.compute_concept_probabilities(
                        x = prob,
                        b = interventions,
                        m = torch.ones_like(interventions).to(interventions.device),
                        y = torch.argmax(rollout_y_logits, dim = -1)
                    )
            # print(logpo[0,:].shape)
            if self.use_ac_model:
                ac_model_output = np.zeros((self.batch_size,), dtype = np.float32)
                ac_model_output = np.expand_dims(ac_model_output, axis = 1)
                # logpo = torch.zeros((self.batch_size,self.n_tasks), dtype = torch.float32)
                sam = sam.detach().cpu().numpy()
                pred_sam = pred_sam.detach().cpu().numpy()
                sam_mean = np.mean(sam, axis = 1)
                sam_std = np.std(sam, axis = 1)
                pred_sam_mean = np.mean(pred_sam, axis = 1)
                pred_sam_std = np.std(pred_sam, axis = 1)
                ac_model_info = np.concatenate((
                    # logpo, 
                sam_mean, sam_std, pred_sam_mean, pred_sam_std), axis = -1)
            
        if self.use_ac_model:
            self._ac_model_output = ac_model_output
            self._ac_model_info = ac_model_info
        self._cbm_bottleneck = embeddings
        self._cbm_pred_concepts = c_sem.detach().cpu().numpy()
        if rollout_y_logits.shape[-1] == 1:
            rollout_y_logits = torch.sigmoid(rollout_y_logits)
            rollout_y_logits = torch.cat((1 - rollout_y_logits, rollout_y_logits), dim = -1)
        self._cbm_pred_output = rollout_y_logits.detach().cpu().numpy()

        obs = self._get_obs()
        info = self._get_info()

        self.num_interventions = 0

        return obs, info
    
    def on_train_epoch_end(self):
        self.max_nll = self.next_max_nll
        logging.debug(
            f"max_nll: {self.max_nll}"
        )
        self.next_max_nll = 1.

    def step(self, action, train, c, y, c_sem, pos_embeddings, neg_embeddings):

        indices = np.column_stack((np.arange(len(self._intervened_concepts_map)), action))
        if self.allow_no_action:
            intervene_indices_mask = indices[:, 1] != 0
            intervene_indices = indices[intervene_indices_mask]
            intervene_indices[:,1] -= 1
        else:
            intervene_indices_mask = indices[:, 1]
            intervene_indices = indices
        self._remaining_budget_arr -= 1
        already_intervened_indices = self._intervened_concepts_map[indices[:, 0], np.max(indices[:, 1] - 1, 0)] == 0
        intervene_indices_mask = np.logical_and(intervene_indices_mask, already_intervened_indices)
        
        if not train:
            assert not np.any(self._intervened_concepts_map[intervene_indices[:, 0], intervene_indices[:, 1]])

        self._intervened_concepts_map[intervene_indices[:, 0], intervene_indices[:, 1]] = 1
        
        prev_interventions = torch.tensor(self._intervened_concepts, device = self.cbm.device)
        for i in range(self._intervened_concepts.shape[0]): 
            if self.allow_no_action:
                if action[i] == 0:
                    continue
                concept_group = sorted(list(self.concept_group_map.keys()))[action[i] - 1]
            else:
                concept_group = sorted(list(self.concept_group_map.keys()))[action[i]]
            for concept in self.concept_group_map[concept_group]:
                self._intervened_concepts[i, concept] = 1

        new_interventions_map = torch.tensor(self._intervened_concepts_map, device = self.cbm.device)
        new_interventions = torch.tensor(self._intervened_concepts, device = self.cbm.device)
        
        # x, y, (c, competencies, _) = self.cbm._unpack_batch(self._cbm_data)
        with torch.no_grad():
            # outputs = self.cbm._forward(
            #     x,
            #     intervention_idxs=new_interventions,
            #     c=c,
            #     y=y,
            #     train=self._train,
            #     competencies=competencies,
            #     prev_interventions=prev_interventions,
            #     output_embeddings=True,
            #     output_latent=True,
            #     output_interventions=True,
            # )
            # c_sem, c_pred, y_logits = outputs[0], outputs[1], outputs[2]
            # pos_embeddings = outputs[-2]
            # neg_embeddings = outputs[-1]
            
            # self.pos_embeddings = pos_embeddings
            # self.neg_embeddings = neg_embeddings
            prob = new_interventions * c + (1 - new_interventions) * c_sem
            # self.new_interventions = new_interventions
            # self.prob = prob
            # self.c = c
            # self.c_sem = c_sem
            
            embeddings = (
                pos_embeddings.detach() * torch.unsqueeze(prob, dim=-1) + (
                    neg_embeddings.detach() * (
                        1 - torch.unsqueeze(prob, dim=-1)
                    )
                )
            ).detach().view((-1, self.cbm.emb_size * self.cbm.n_concepts))
            rollout_y_logits = self.cbm.c2y_model(embeddings)
            embeddings = embeddings.cpu().numpy()
            # embeddings = np.zeros((self.batch_size, self.cbm.emb_size * self.cbm.n_concepts), dtype = np.float32)
            # embeddings = torch.where(prev_interventions.clone().repeat(1, self.emb_size).bool(), 0, embeddings).cpu().numpy()
            if self.use_ac_model:
                ac_model_output, logpo, sam, pred_sam = self.ac_model.compute_concept_probabilities(
                            x = prob,
                            b = new_interventions,
                            m = torch.ones_like(new_interventions).to(new_interventions.device),
                            y = torch.argmax(rollout_y_logits, dim = -1)
                        )
                ac_model_output = ac_model_output.detach().cpu().numpy()
                self.next_max_nll = max(self.next_max_nll, np.max(ac_model_output))
                max_nll = self.max_nll if self.max_nll > 1 else 100
                # ac_model_output = np.exp(ac_model_output / max_nll)
                ac_model_output = np.exp(np.clip(ac_model_output, None, 1))
                # ac_model_output = ac_model_output / np.max(ac_model_output)d
                ac_model_output = np.where(intervene_indices_mask, ac_model_output, 0)
                ac_model_output = np.expand_dims(ac_model_output, axis = 1)
            else:
                ac_model_output = None

            self.num_interventions += 1
            terminated = self.num_interventions == self._budget
            
            reward = self.calculate_reward(action, terminated, rollout_y_logits, y, ac_model_output, c_sem, c)
            if rollout_y_logits.shape[-1] == 1:
                rollout_y_logits = torch.sigmoid(rollout_y_logits)
                rollout_y_logits = torch.cat((1 - rollout_y_logits, rollout_y_logits), dim = -1)
            else:
                rollout_y_logits = self.softmax(rollout_y_logits)
            # with open("ac_output.txt", "a") as f:
            #     f.write(str(self._remaining_budget_arr[0]) + ", " + str(self._budget) + "\n")
            #     f.write(str(ac_model_output))
            #     f.write("\n")
            if self.use_ac_model:
                if self.normalize_values:
                    # logpo = self.softmax(torch.exp(logpo))
                    sam = self.softmax(sam)
                    pred_sam = self.softmax(pred_sam)
                # logpo = logpo.detach().cpu().numpy()
                # with open("logpo.txt", "a") as f:
                #     f.write(str(self._remaining_budget_arr[0]) + ", " + str(self._budget) + "\n")
                #     f.write(str(logpo))
                #     f.write("\n")
                # sam = self.softmax(sam)
                # logging.debug(
                #     f"ac_model_ouptut.shape: {ac_model_output.shape}"
                # )
                # logging.debug(
                #     f"logpo.shape: {logpo.shape}"
                # )
                # logging.debug(
                #     f"sam.shape: {sam.shape}"
                # )
                sam = sam.detach().cpu().numpy()
                # pred_sam = self.softmax(pred_sam)
                # logging.debug(
                #     f"pred_sam.shape: {sam.shape}"
                # )
                pred_sam = pred_sam.detach().cpu().numpy()
                sam_mean = np.mean(sam, axis = 1)
                # logging.debug(
                #     f"sam_mean.shape: {sam_mean.shape}"
                # )
                sam_std = np.std(sam, axis = 1)
                pred_sam_mean = np.mean(pred_sam, axis = 1)
                pred_sam_std = np.std(pred_sam, axis = 1)
                ac_model_info = np.concatenate((
                    # logpo, 
                    sam_mean, sam_std, pred_sam_mean, pred_sam_std), axis = -1)
                # with open("ac_info.txt", "a") as f:
                #     f.write(str(self._remaining_budget_arr[0]) + ", " + str(self._budget) + "\n")
                #     f.write(str(ac_model_info))
                #     f.write("\n")
        if self.use_ac_model:    
            self._ac_model_output = ac_model_output
            self._ac_model_info = ac_model_info
        self._cbm_bottleneck = embeddings
        self._cbm_pred_concepts = c_sem.detach().cpu().numpy()
        self._cbm_pred_output = rollout_y_logits.detach().cpu().numpy()

        obs = self._get_obs()
        info = self._get_info()

        terminated = self.num_interventions == self._remaining_budget_arr
        terminated = np.squeeze(terminated, axis = -1)

        return obs, reward, terminated, False, info
    
    def calculate_reward(self, action, terminated, predictions, y, ac_model_output, c_sem, c):
        if self.allow_no_action:
            costs = self.group_costs[action]
        else:
            costs = 0

        if terminated:
            one_hot = torch.nn.functional.one_hot(y.long(), num_classes = self.n_tasks if self.n_tasks > 1 else 2)
            
            if predictions.shape[-1] == 1:
                softmax = torch.sigmoid(predictions)
                softmax = torch.cat((1 - softmax, softmax), dim = -1)
            else:
                softmax = self.softmax(predictions)
            loss = one_hot * softmax
            loss = torch.sum(loss, dim = -1).detach().cpu().numpy()
            if self.final_reward_ratio > 0:
                return self.final_reward_ratio * loss - costs
            else:
                return self.final_reward_ratio * (1 - loss) - costs
        else:
            if ac_model_output is not None:
                return self.intermediate_reward_ratio * ac_model_output - costs
            else:
                return np.zeros(y.shape)

    def is_model_frozen(self, model):
        # Iterate through all parameters in the model
        for param in model.parameters():
            # If any parameter requires gradient, model is not completely frozen
            if param.requires_grad:
                return False
        # If all parameters are frozen, return True
        return True

    def set_budget(self, budget):
        self.env_budget = budget

    def get_ith(self, data, i):
        if isinstance(data, list) or isinstance(data, tuple):
            return [item[i] for item in data]
        else:
            return data[i]

    def forward(
        self,
        x,
        c=None,
        y=None,
        latent=None,
        intervention_idxs=None,
        competencies=None,
        prev_interventions=None,
    ):
        return self.cbm.forward(
            x,
            c=c,
            y=y,
            competencies=competencies,
            prev_interventions=prev_interventions,
            intervention_idxs=intervention_idxs,
            latent=latent,
        )

    def _run_interventions(self, c_sem, c_used, c, y, y_logits, competencies, pos_embeddings, neg_embeddings, budget, train):
        if budget == 0:
            return 0., 0., 0., 0., 0., 0., 0.
            
        batch_size = c_sem.shape[0]

        try:
            train_rl = self.train_rl
        except:
            train_rl = False

        if batch_size != self.batch_size:
            self.batch_size = batch_size
            # self.env = AFAEnv(cbm = self.cbm, ac_model = self.ac_model, env_config = self.config, batch_size = self.batch_size)
        
        intervention_loss = 0.0
        pg_losses = 0.0
        ent_losses = 0.0
        v_losses = 0.0
        intervention_task_loss = 0.0
        prev_num_of_interventions = 0
        # Then time to perform a forced training time intervention
        # We will set some of the concepts as definitely intervened on
        if competencies is None:
            competencies = torch.FloatTensor(
                c_used.shape
            ).uniform_(0.5, 1).to(c_used.device)
        # int_basis_lim = (
        #     len(self.cbm.concept_map) if self.cbm.use_concept_groups
        #     else self.cbm.n_concepts
        # )
        # horizon_lim = int(self.cbm.horizon_limit.detach().cpu().numpy()[0])
        # if prev_num_of_interventions != int_basis_lim:
        #     bottom = min(
        #         horizon_lim,
        #         int_basis_lim - prev_num_of_interventions - 1,
        #     )  # -1 so that we at least intervene on one concept
        #     if bottom > 0:
        #         initially_selected = np.random.randint(0, bottom)
        #     else:
        #         initially_selected = 0
        #     end_horizon = min(
        #         int(horizon_lim),
        #         self.cbm.max_horizon,
        #         int_basis_lim - prev_num_of_interventions - initially_selected,
        #     )
        #     current_horizon = self.cbm._horizon_distr(
        #         init=1 if end_horizon > 1 else 0,
        #         end=end_horizon,
        #     )

        #     for sample_idx in range(intervention_idxs.shape[0]):
        #         probs = free_groups[sample_idx, :].detach().cpu().numpy()
        #         probs = probs/np.sum(probs)
        #         if self.cbm.use_concept_groups:
        #             selected_groups = set(np.random.choice(
        #                 int_basis_lim,
        #                 size=initially_selected,
        #                 replace=False,
        #                 p=probs,
        #             ))
        #             for group_idx, (_, group_concepts) in enumerate(
        #                 self.concept_map.items()
        #             ):
        #                 if group_idx in selected_groups:
        #                     intervention_idxs[sample_idx, group_concepts] = 1
        #         else:
        #             intervention_idxs[
        #                 sample_idx,
        #                 np.random.choice(
        #                     int_basis_lim,
        #                     size=initially_selected,
        #                     replace=False,
        #                     p=probs,
        #                 )
        #             ] = 1
        if self.cbm.initialize_discount:
            # discount = (
            #     self.intervention_discount ** prev_num_of_interventions
            # )
            task_discount = self.cbm.intervention_task_discount ** (
                prev_num_of_interventions
            )
            first_task_discount = (
                self.intervention_task_discount **
                prev_num_of_interventions
            )
        else:
            # discount = 1
            task_discount = 1
            first_task_discount = 1

        # trajectory_weight = 0
        # else:
        #     trajectory_weight = 1

        if self.cbm.n_tasks > 1:
            one_hot_y = torch.nn.functional.one_hot(y, self.cbm.n_tasks)
        # else:
        #     current_horizon = 0
        #     if self.initialize_discount:
        #         task_discount = self.intervention_task_discount ** (
        #             prev_num_of_interventions
        #         )
        #         first_task_discount = (
        #             self.intervention_task_discount **
        #             prev_num_of_interventions
        #         )
        #     else:
        #         task_discount = 1
        #         first_task_discount = 1
        #     task_trajectory_weight = 1
        #     trajectory_weight = 1

        int_mask_accuracy = 0.0
        accuracy = 0.0
        auc = 0.0
        # if (not self.legacy_mode) and (
        #     (
        #         self.current_steps.detach().cpu().numpy()[0] <
        #         self.rollout_init_steps
        #     )
        # ):
        #     current_horizon = 0
        if self.cbm.rollout_aneal_rate != 1:
            num_rollouts = int(round(
                self.num_rollouts * (
                    self.cbm.current_aneal_rate.detach().cpu().numpy()[0]
                )
            ))
        else:
            num_rollouts = self.cbm.num_rollouts
        if self.cbm.max_num_rollouts is not None:
            num_rollouts = min(num_rollouts, self.cbm.max_num_rollouts)
        batch_size = c_sem.shape[0]

        if budget is not None:
            num_rollouts = 1
        
        mean_rewards = 0

        for _ in range(num_rollouts):
            # for i in range(batch_size):
            task_trajectory_weight = 1
            # if self.cbm.average_trajectory:
                # curr_discount = discount
            curr_task_discount = task_discount

            if budget is None:
                curr_budget = np.random.randint(1, len(self.cbm.concept_map))
            else:
                curr_budget = budget
            if self.cbm.average_trajectory:
                curr_task_discount = task_discount
                for i in range(curr_budget):
                    # trajectory_weight += discount
                    # discount *= self.cbm.intervention_discount

                    task_discount *= self.cbm.intervention_task_discount
                    if (
                        (not self.cbm.include_only_last_trajectory_loss) or
                        (i == curr_budget - 1)
                    ):
                        task_trajectory_weight += task_discount
                task_discount = curr_task_discount
            else:
                task_trajectory_weight = 1
                
            if self.cbm.include_task_trajectory_loss and (
                self.cbm.task_loss_weight == 0
            ):
                # Then we initialize the intervention trajectory task loss to
                # that of the unintervened model as this loss is not going to
                # be taken into account if we don't do this
                curr_task_loss = first_task_discount * self.cbm.loss_task(
                    (
                        y_logits if y_logits.shape[-1] > 1
                        else y_logits.reshape(-1)
                    ),
                    y,
                )
                if self.cbm.average_trajectory:
                    curr_task_loss = curr_task_loss / task_trajectory_weight
                intervention_task_loss += (
                    curr_task_loss
                )
            # discount = curr_discount
            task_discount = curr_task_discount
            obs = torch.zeros((batch_size * curr_budget, ) + self.single_observation_space.shape).to(self.device)
            actions = torch.zeros((batch_size * curr_budget, )).to(self.device)
            rewards = torch.zeros((batch_size * curr_budget, )).to(self.device)
            dones = torch.zeros((batch_size * curr_budget, )).to(self.device)
            logprobs = torch.zeros((batch_size * curr_budget, )).to(self.device)
            values = torch.zeros((batch_size * curr_budget, )).to(self.device)

            next_obs, next_info = self.reset(curr_budget, train, c, y, c_sem, pos_embeddings, neg_embeddings)
            next_obs = torch.tensor(next_obs, device = self.agent.device)            
            next_done = torch.zeros(1)

            for step in range(curr_budget):
                
                key_start = batch_size * step
                key_end = batch_size * (step + 1)

                obs[key_start:key_end] = next_obs
                dones[key_start:key_end] = next_done

                with torch.no_grad():
                    action, logprob, _, value = self.agent.get_action_and_value(next_obs, info = next_info, train = train)
                    values[key_start:key_end] = value.flatten()

                actions[key_start:key_end] = action
                logprobs[key_start:key_end] = logprob

                next_obs, reward, done, truncated, next_info = self.step(action.cpu().numpy(), train, c, y, c_sem, pos_embeddings, neg_embeddings)
                done = torch.logical_or(torch.tensor(done), torch.tensor(truncated))
                
                rewards[key_start:key_end] = torch.tensor(reward).view(-1).to(self.device)
                next_obs, next_done = torch.tensor(next_obs).to(self.device), done.to(self.device)
                task_discount *= self.cbm.intervention_task_discount
                
                if self.cbm.include_task_trajectory_loss and (
                    (not self.cbm.include_only_last_trajectory_loss) or
                    (step == (curr_budget - 1))
                ):
                    # Then we will also update the task loss with the loss
                    # of performing this intervention!
                    rollout_y_logits = self.cbm.c2y_model(torch.tensor(self._cbm_bottleneck).to(self.cbm.device))
                    rollout_y_loss = self.cbm.loss_task(
                        (
                            rollout_y_logits
                            if rollout_y_logits.shape[-1] > 1 else
                            rollout_y_logits.reshape(-1)
                        ),
                        y,
                    )
                    if self.cbm.average_trajectory:
                        intervention_task_loss += (
                            task_discount *
                            rollout_y_loss / task_trajectory_weight
                        )
                    else:
                        intervention_task_loss += (
                            task_discount * rollout_y_loss
                        )

                    # outputs = self.cbm._forward(
                    #     x,
                    #     intervention_idxs=prev_interventions,
                    #     c=c,
                    #     y=y,
                    #     train=self._train,
                    #     competencies=competencies,
                    #     prev_interventions=None,
                    #     output_embeddings=True,
                    #     output_latent=True,
                    #     output_interventions=True,
                    # )
                    # c_sem, c_pred, y_logits = outputs[0], outputs[1], outputs[2]
                if step == (curr_budget - 1):
                    (c_accuracy, c_auc, c_f1), (y_accuracy, y_auc, y_f1) = compute_accuracy(
                        c_sem,
                        rollout_y_logits,
                        c,
                        y,
                    )
                    accuracy += y_accuracy
                    auc += y_auc
            
            returns, advantages = self.agent.estimate_returns_and_advantages(
                rewards, values, dones, next_obs, next_done, curr_budget, batch_size
            )
            # if torch.any(torch.abs(advantages) > 100):
            #     import pdb
            #     pdb.set_trace()
            local_data = {
                "obs": obs.reshape((-1,) + self.single_observation_space.shape),
                "logprobs": logprobs.reshape(-1),
                "actions": actions.reshape((-1,)),
                "advantages": advantages.reshape(-1),
                "returns": returns.reshape(-1),
                "values": values.reshape(-1),
            }

            pg_loss, ent_loss, v_loss = self.agent.training_step(local_data)

            if self.normalize_values:
                v_loss = torch.clamp(v_loss, -curr_budget, curr_budget)
                pg_loss = torch.clamp(pg_loss, -curr_budget, curr_budget)

            pg_losses += pg_loss / curr_budget
            ent_losses += ent_loss / curr_budget
            v_losses += v_loss / curr_budget
        
            self.log("mean_rewards", np.mean(reward), on_step = True, on_epoch = True, prog_bar=False, sync_dist = True)
        self.log("pg_loss", pg_losses.detach() if not isinstance(pg_losses, float) else pg_losses, prog_bar=False, sync_dist = True)
        self.log("ent_loss", ent_losses.detach() if not isinstance(ent_losses, float) else ent_losses, prog_bar=False, sync_dist = True)
        self.log("v_loss", v_losses.detach() if not isinstance(v_losses, float) else v_losses, prog_bar=False, sync_dist = True)

        # with open("v_loss.txt", "a") as f:
        #     msg = f"Budget: {budget}\n"
        #     f.write(msg)
        #     f.write(str(v_losses.detach() / budget if not isinstance(v_losses, float) else v_losses  / budget))
        #     f.write("\n")

        intervention_loss += pg_losses
        intervention_loss += ent_losses
        intervention_loss += v_losses

        intervention_loss_scalar = \
            self.cbm.intervention_weight * intervention_loss/num_rollouts
        intervention_loss = intervention_loss/num_rollouts
        intervention_task_loss_scalar = \
            self.cbm.intervention_task_loss_weight * intervention_task_loss/num_rollouts
        intervention_task_loss = intervention_task_loss/num_rollouts
        int_mask_accuracy = int_mask_accuracy/num_rollouts
        accuracy = accuracy / num_rollouts
        auc = auc / num_rollouts
        return intervention_loss_scalar, intervention_loss, intervention_task_loss, intervention_task_loss_scalar, int_mask_accuracy, accuracy, auc

    def on_train_batch_end(self, *args, **kwargs):
        lr = self.trainer.optimizers[0].param_groups[0]['lr']
        # Log learning rate
        self.log('learning_rate', lr, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.log('cbm_current_aneal_rate', self.cbm.cbm_current_aneal_rate, on_step=False, on_epoch=True, prog_bar=True, logger=True)

    def _run_step(self,
        batch,
        batch_idx,
        budget=None,
        train=False,
        intervention_idxs=None
    ):
        x, y, (c, competencies, prev_interventions) = self.cbm._unpack_batch(batch)
        outputs = self.cbm._forward(
            x,
            intervention_idxs=intervention_idxs,
            c=c,
            y=y,
            train=train,
            competencies=competencies,
            prev_interventions=prev_interventions,
            output_embeddings=True,
            output_latent=True,
            output_interventions=True,
        )
        c_sem, c_logits, y_logits = outputs[0], outputs[1], outputs[2]
        latent = outputs[4]
        pos_embeddings = outputs[-2]
        neg_embeddings = outputs[-1]

        if self.cbm.task_loss_weight != 0:
            task_loss = self.cbm.loss_task(
                y_logits if y_logits.shape[-1] > 1 else y_logits.reshape(-1),
                y,
            )
            task_loss_scalar = self.cbm.task_loss_weight * task_loss.detach()
        else:
            task_loss = 0.0
            task_loss_scalar = 0.0

        intervention_task_loss = 0.0

        # Now we will do some rolls for interventions
        int_mask_accuracy = -1.0
        if not self.cbm.include_certainty:
            c_used = torch.where(
                torch.logical_or(c == 0, c == 1),
                c,
                c_sem.detach(),
            )
        else:
            c_used = c
        if not self.cbm.legacy_mode:
            self.cbm.current_steps += 1
            if self.cbm.rollout_aneal_rate != 1 and (
                self.cbm.current_aneal_rate.detach().cpu().numpy()[0] < 100
            ):
                self.cbm.current_aneal_rate *= self.cbm.rollout_aneal_rate

        if self.cbm.concept_loss_weight != 0:
            # We separate this so that we are allowed to
            # use arbitrary activations (i.e., not necessarily in [0, 1])
            # whenever no concept supervision is provided
            if self.cbm.include_certainty:
                concept_loss = self.cbm.loss_concept(c_sem, c)
                concept_loss_scalar = \
                    self.cbm.concept_loss_weight * concept_loss.detach()
            else:
                c_sem_used = torch.where(
                    torch.logical_or(c == 0, c == 1),
                    c_sem,
                    c,
                ) # This forces zero loss when c is uncertain
                concept_loss = self.cbm.loss_concept(c_sem_used, c)
                concept_loss_scalar = concept_loss.detach()
        else:
            concept_loss = 0.0
            concept_loss_scalar = 0.0

        (c_accuracy, c_auc, c_f1), (y_accuracy, y_auc, y_f1) = compute_accuracy(
            c_sem,
            y_logits,
            c,
            y,
        )
        
        if self.cbm.intervention_weight != 0 or self.cbm.intervention_task_loss_weight != 0:
        
            intervention_loss_scalar, intervention_loss, intervention_task_loss, intervention_task_loss_scalar, int_mask_accuracy, intervention_accuracy, intervention_auc = \
                self._run_interventions(c_sem, c_used, c, y, y_logits, competencies, pos_embeddings, neg_embeddings, budget, train)

        else:
            intervention_loss_scalar, intervention_loss, intervention_task_loss, intervention_task_loss_scalar, int_mask_accuracy, intervention_accuracy, intervention_auc = \
                0., 0., 0., 0., 0., 0., 0.

        if self.cbm.cbm_aneal_rate != 1:
            self.cbm.cbm_current_aneal_rate *= self.cbm.cbm_aneal_rate

        loss = (
            self.cbm.cbm_current_aneal_rate * self.cbm.concept_loss_weight * concept_loss +
            self.cbm.intervention_weight * intervention_loss +
            self.cbm.cbm_current_aneal_rate * self.cbm.task_loss_weight * task_loss +
            self.cbm.intervention_task_loss_weight * intervention_task_loss
        )

        loss += self.cbm._extra_losses(
            x=x,
            y=y,
            c=c,
            c_sem=c_sem,
            c_pred=c_logits,
            y_pred=y_logits,
            competencies=competencies,
        )
        # compute accuracy
        result = {
            "c_accuracy": c_accuracy,
            "c_auc": c_auc,
            "c_f1": c_f1,
            "y_accuracy": y_accuracy,
            "y_auc": y_auc,
            "y_f1": y_f1,
            "mask_accuracy": int_mask_accuracy,
            "concept_loss": concept_loss_scalar,
            "task_loss": task_loss_scalar,
            "intervention_task_loss": intervention_task_loss_scalar,
            "intervention_loss": intervention_loss_scalar,
            "cbm_loss": task_loss_scalar + concept_loss_scalar + intervention_task_loss_scalar,
            "agent_loss": intervention_loss_scalar,
            "loss": loss.detach() if not isinstance(loss, float) else loss,
            "avg_c_y_acc": (c_accuracy + y_accuracy) / 2,
            "horizon_limit": self.cbm.horizon_limit.detach().cpu().numpy()[0],
            "intervention_accuracy": intervention_accuracy,
            "intervention_auc": intervention_auc,
        }
        if not self.cbm.legacy_mode:
            result["current_steps"] = \
                self.cbm.current_steps.detach().cpu().numpy()[0]
            if self.cbm.rollout_aneal_rate != 1:
                num_rollouts = int(round(
                    self.cbm.num_rollouts * (
                        self.cbm.current_aneal_rate.detach().cpu().numpy()[0]
                    )
                ))
                if self.cbm.max_num_rollouts is not None:
                    num_rollouts = min(num_rollouts, self.cbm.max_num_rollouts)
                result["num_rollouts"] = num_rollouts

        if self.cbm.top_k_accuracy is not None:
            y_true = y.reshape(-1).cpu().detach()
            y_pred = y_logits.cpu().detach()
            labels = list(range(self.cbm.n_tasks))
            for top_k_val in self.cbm.top_k_accuracy:
                y_top_k_accuracy = sklearn.metrics.top_k_accuracy_score(
                    y_true,
                    y_pred,
                    k=top_k_val,
                    labels=labels,
                )
                result[f'y_top_{top_k_val}_accuracy'] = y_top_k_accuracy
        return loss, result
    
    def training_step(self, batch, batch_idx):
        loss, result = self._run_step(batch, batch_idx, budget = None, train=True)
        for name, val in result.items():
            if self.cbm.n_tasks <= 2:
                prog_bar = (
                    ("auc" in name) or
                    ("mask_accuracy" in name) or
                    ("current_steps" in name) or
                    ("num_rollouts" in name)
                )
            else:
                prog_bar = (
                    ("c_auc" in name) or
                    ("y_accuracy" in name) or
                    ("mask_accuracy" in name) or
                    ("current_steps" in name) or
                    ("num_rollouts" in name)

                )
            self.log(name, val, prog_bar=prog_bar, on_epoch = True, sync_dist = True)
        return {
            "loss": loss,
            "log": {
                "c_accuracy": result['c_accuracy'],
                "c_auc": result['c_auc'],
                "c_f1": result['c_f1'],
                "y_accuracy": result['y_accuracy'],
                "y_auc": result['y_auc'],
                "y_f1": result['y_f1'],
                "concept_loss": result['concept_loss'],
                "task_loss": result['task_loss'],
                "loss": result['loss'],
                "avg_c_y_acc": result['avg_c_y_acc'],
                "intervention_task_loss": result["intervention_task_loss"],
                "intervention_loss": result["intervention_loss"],
                "cbm_loss": result["cbm_loss"],
                "agent_loss": result["agent_loss"],
            },
        }

    def validation_step(self, batch, batch_idx):
        _, result = self._run_step(batch, batch_idx, train=False)
        for name, val in result.items():
            if self.cbm.n_tasks <= 2:
                prog_bar = (("auc" in name))
            else:
                prog_bar = (("c_auc" in name) or ("y_accuracy" in name))
            self.log("val_" + name, val, prog_bar=prog_bar, sync_dist = True)
        result = {
            "val_" + key: val
            for key, val in result.items()
        }
        return result

    def on_test_epoch_start(self):
        logging.debug(
            f"Testing with budget {self.env_budget}"
        )

    def test_step(self, batch, batch_idx):
        loss, result = self._run_step(batch, batch_idx, budget = self.env_budget, train=False)
        for name, val in result.items():
            self.log("test_" + name, val, prog_bar=True, sync_dist = True)
        return result
    
    def configure_optimizers(self):
        if self.use_separate_optimizers:
            if self.cbm.optimizer_name.lower() == "adam":
                optimizer1 = torch.optim.Adam(
                    list(self.cbm.parameters()),
                    lr=self.cbm.learning_rate,
                    weight_decay=self.cbm.weight_decay,
                )
                optimizer2 = torch.optim.Adam(
                    list(self.agent.parameters()),
                    lr=self.cbm.learning_rate,
                    weight_decay=self.cbm.weight_decay,
                )
            else:
                optimizer1 = torch.optim.SGD(
                    filter(lambda p: p.requires_grad, 
                    list(self.cbm.parameters())),
                    lr=self.cbm.learning_rate,
                    momentum=self.cbm.momentum,
                    weight_decay=self.cbm.weight_decay,
                )
                optimizer2 = torch.optim.SGD(
                    filter(lambda p: p.requires_grad, 
                    list(self.agent.parameters())),
                    lr=self.cbm.learning_rate,
                    momentum=self.cbm.momentum,
                    weight_decay=self.cbm.weight_decay,
                )
            lr_scheduler1 = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer1,
                verbose=True,
            )

            scheduler1 = {
                'scheduler': lr_scheduler1,
                'monitor': 'cbm_loss',
                'interval': 'epoch',
                'frequency': 1,
                'reduce_on_plateau': True
            }

            lr_scheduler2 = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer2,
                verbose=True,
            )

            scheduler2 = {
                'scheduler': lr_scheduler2,
                'monitor': 'agent_loss',
                'interval': 'epoch',
                'frequency': 1,
                'reduce_on_plateau': True
            }
            
            return [optimizer1, optimizer2], [scheduler1, scheduler2]
        else:
            if self.cbm.optimizer_name.lower() == "adam":
                optimizer1 = torch.optim.AdamW(
                    list(self.cbm.parameters())
                    + list(self.agent.parameters()),
                    lr=self.cbm.learning_rate,
                    weight_decay=self.cbm.weight_decay,
                )
            else:
                optimizer1 = torch.optim.SGD(
                    filter(lambda p: p.requires_grad, 
                    list(self.cbm.parameters())
                    + list(self.agent.parameters())),
                    lr=self.cbm.learning_rate,
                    momentum=self.cbm.momentum,
                    weight_decay=self.cbm.weight_decay,
                )
            lr_scheduler1 = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer1,
                verbose=True,
            )

            scheduler1 = {
                'scheduler': lr_scheduler1,
                'monitor': 'loss',
                'interval': 'epoch',
                'frequency': 1,
                'reduce_on_plateau': True
            }
            
            return [optimizer1], [scheduler1]

    def predict_step(
        self,
        batch,
        batch_idx,
        intervention_idxs=None,
        dataloader_idx=0,
    ):
        x, y, (c, competencies, prev_interventions) = self.cbm._unpack_batch(batch)
        return self.cbm._forward(
            x,
            intervention_idxs=intervention_idxs,
            c=c,
            y=y,
            train=False,
            competencies=competencies,
            prev_interventions=prev_interventions,
        )



    def intervene(
        self,
        x,
        pred_c,
        c,
        y=None,
        competencies=None,
        prev_interventions=None,
        prior_distribution=None,
    ):
        pass
        