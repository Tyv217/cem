import torch
import logging

import pytorch_lightning as pl
import torch.nn.functional as F
import numpy as np
import random

def generate_random_numbers(n, n_concepts):
    numbers = set()
    while len(numbers) < float(n):
        numbers.add(random.randint(0, n_concepts-1))
    return list(numbers)

class ACEnergy(pl.LightningModule):
    
    # def __init__(self,args,num_classes, input_size=1000, hid_size=1000, cpt_size=None):
    def __init__(self, n_concepts, n_tasks, embed_size = 1000, cy_perturb_prob = 0.2, cy_permute_prob = 0.2, energy_model_architecture = "linear", class_weights = None, optimizer = None, learning_rate = None, weight_decay = None, momentum = None):
        super().__init__()

        self.n_concepts = n_concepts
        self.n_tasks = n_tasks if n_tasks > 1 else 2
        self.embed_size = embed_size
        self.cy_perturb_prob = cy_perturb_prob
        self.cy_permute_prob = cy_permute_prob

        # self.y_prob = torch.nn.Parameter(torch.randn((self.n_tasks)))
        self.y_embedding = torch.nn.Parameter(torch.randn((self.n_tasks, embed_size), requires_grad = True))
        
        # self.c_prob = torch.nn.Parameter(torch.randn((self.n_concepts)))
        self.c_embedding = torch.nn.Parameter(torch.randn((self.n_concepts*2, embed_size), requires_grad = True))
        self.classifier_cy = torch.nn.Linear(embed_size, 1)
        self.concept_proj = torch.nn.Linear(self.n_concepts * embed_size, embed_size)
        self.smx_y = torch.nn.Softmax(dim=-2)
        self.smx_c = torch.nn.Softmax(dim=-1)
        self.dropout = torch.nn.Dropout(p=0.2)

        class_weights = torch.tensor(np.array(class_weights or [1. for _ in range(self.n_tasks)]).astype("float32")).to(self.device)
        class_weights /= torch.sum(class_weights)
        self.class_weights = torch.log(class_weights)

        self.class_list = [i for i in range(self.n_tasks)]

        self.optimizer_name = optimizer
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.momentum = momentum

    def unfreeze(self):
        self.y_embedding.requires_grad = True
        self.c_embedding.requires_grad = True
            
    def cy_augment(self,c_gt,permute_ratio,permute_prob=0.2):
        """
        Applies augmentation to the given ground truth tensor.

        Args:
            c_gt (torch.Tensor): The ground truth tensor.
            permute_ratio (float): The ratio of concepts to permute.
            permute_prob (float, optional): The ratio of samples in a batch to permute. Defaults to 0.2.
        
        Returns:
            torch.Tensor: The augmented ground truth tensor.
        """
        
        c_gt=c_gt.squeeze(-1)
        bs,all_length=c_gt.shape[0],c_gt.shape[-1]
        permute_concept_number=all_length*permute_ratio
        permute_sample_number=bs*permute_prob
        permute_concept_idx=torch.tensor(generate_random_numbers(permute_concept_number,all_length)).to(c_gt.device)
        permute_samps=torch.tensor(generate_random_numbers(permute_sample_number,bs)).to(c_gt.device)
        c_gt=c_gt.long()
        to_be_interf=c_gt[permute_samps]
        to_be_interf[:,permute_concept_idx]=to_be_interf[:,permute_concept_idx]^1
        for idx,permidx in enumerate(permute_samps):
            c_gt[permidx]=to_be_interf[idx]
        c_gt=c_gt.unsqueeze(-1)

        return c_gt

    def forward(self, c_gt, mask, train):
        # input x is encoded image.
        bs = c_gt.shape[0]

        # #### X->Y energy ###
        # # project y into the embedding space to calculate class-wise energy
        y_embed=self.y_embedding.unsqueeze(0) # [1,label_size, hidden_size]
        y_embed=y_embed.repeat(bs,1,1) # [bs,label_size,hidden_size]

        # if not is_training:
        #     y_prob=self.y_prob
        #     y_prob=self.smx(y_prob)
        #     y_embed=torch.sum(y_prob*y_embed,dim=1) # [bs,hidden_size]

        y_embed= F.normalize(y_embed, p=2, dim=-1)        
        # # project x into the same space to calculate energy
        # x_embed = self.xy_fc1(x) # [bs, hidden_size]
        # x_embed = self.dropout(x_embed)
        # if is_training:
        #     x_embed = x_embed[:,None,:].expand_as(y_embed) # [bs,label_size, hidden_size]
        # # z: x->y energy
        # z_xy = x_embed * y_embed
        # z_xy = x_embed + z_xy
        # z_xy = F.relu(z_xy)
        # # reduce "energy embedding to 1 dim"
        # xy_energy = self.classifier_xy(z_xy) # [bs, label_size, 1]
        # # do a class-wise energy transpose.
        # xy_energy = xy_energy.view(bs, -1)


        
        # #### x->c energy ####
        # x_embed = self.xc_fc1(x) # [bs, hidden_size]
        # x_embed = self.dropout(x_embed)
        # c_embed=self.c_embedding.unsqueeze(0) # [1,concept_size, hidden_size]
        # c_embed=c_embed.repeat(bs,1,1) # [bs,concept_size,hidden_size]
        # c_embed_cy=c_embed
        # if not is_training:
        #     c_prob=self.c_prob
        #     c_prob=self.smx_c(c_prob)
        #     c_embed=c_embed[:,:self.n_concepts]*c_prob[:,:,0:1]+c_embed[:,self.n_concepts:]*c_prob[:,:,1:2]
        #     # print(c_embed.shape)
        # c_embed= F.normalize(c_embed, p=2, dim=-1)
        # x_embed = x_embed[:,None,:].expand_as(c_embed) # [bs,concept_size*2, hidden_size]

        # xc_energy=[]
        # for i in range(self.n_concepts):
        #     if not is_training:
        #         # print(c_embed[:,i].shape)
        #         xc_embed = x_embed[:,i] * c_embed[:,i]
        #         xc_embed = x_embed[:,i] + xc_embed
        #         xc_embed = F.relu(xc_embed)
        #         xc_energy_single =self.classifier_xc[i](xc_embed) # [bs,hidden_size] -> [bs,1]
        #         xc_energy_single = xc_energy_single.view(bs, -1)
        #         xc_energy_single = xc_energy_single.unsqueeze(1) # [bs,1,1]
        #     else:
        #         # z: x->y energy
        #         # print(x_c_pos.shape,c_embed[:,i,:self.hid_size].shape)
        #         xc_pos_embed = x_embed[:,i] * c_embed[:,i]
        #         xc_pos_embed = x_embed[:,i] + xc_pos_embed
        #         xc_pos_embed = F.relu(xc_pos_embed)

        #         xc_neg_embed =  x_embed[:,i+self.n_concepts] * c_embed[:,i+self.n_concepts]
        #         xc_neg_embed = x_embed[:,i+self.n_concepts] + xc_neg_embed
        #         xc_neg_embed = F.relu(xc_neg_embed)
        #         xc_embed = torch.stack([xc_pos_embed,xc_neg_embed],dim=1) # [bs, 2, hidden_size]

        #         xc_energy_single = self.classifier_xc[i](xc_embed) # [bs, 2, 1]
        #         xc_energy_single = xc_energy_single.view(bs, -1)
        #         xc_energy_single = xc_energy_single.unsqueeze(1)
        #     xc_energy.append(xc_energy_single)
        # xc_energy=torch.cat(xc_energy,dim=1)

        #### c->y energy.####
        c_embed_cy=self.c_embedding.unsqueeze(0) # [1,concept_size, hidden_size]
        c_embed_cy=c_embed_cy.repeat(bs,1,1) # [bs,concept_size,hidden_size]
        
        # if not is_training:
        #     concepts = (c_prob > 0.5).long()
        #     if(len(concepts.shape) != len(c_embed_cy.shape)):
        #         concepts = torch.unsqueeze(concepts, dim = -1)
        #         concepts = concepts.expand_as(c_embed_cy[:, :self.n_concepts, :])
        #     c_embed_positive = concepts * c_embed_cy[:, :self.n_concepts, :]
        #     c_embed_negative = (1 - concepts) * c_embed_cy[:, self.n_concepts:, :]

        #     c_embed = c_embed_positive + c_embed_negative
        #     # for k in range(c_prob.shape[1]):
        #     #     single_c_embed=torch.where(c_prob[:,k,1:2]>0.5,c_embed_cy[:,k,:],c_embed_cy[:,k+self.n_concepts,:])
        #     #     single_c_embed=single_c_embed.unsqueeze(1)
        #     #     c_embed.append(single_c_embed)
        #     # c_embed=torch.cat(c_embed,dim=1).view(bs,-1)
        #     c_embed = c_embed.view(bs, -1)
        #     c_embed=self.concept_proj(c_embed)
        #     # c_single_embed = c_single_embed[:,None,:].expand_as(y_embed) # [bs,label_size, hidden_size]
        #     cy_embed = c_embed * y_embed
        #     cy_embed = c_embed + cy_embed
        #     cy_embed = F.relu(cy_embed) # [bs,label_size, hidden_size]
        #     # c_y_energy_embererund=z_cy.view(bs*self.num_classes,-1)
        #     cy_energy = self.p(cy_embed) # [bs, 1, 1]
        #     # do a class-wise energy transpose.
        #     cy_energy = cy_energy.view(bs, -1)

        # else:
        c_pos=c_gt
        c_pos=c_pos.unsqueeze(-1)
        if train:
            c_pos=self.cy_augment(c_gt=c_pos,permute_ratio=self.cy_perturb_prob,permute_prob=self.cy_permute_prob)
        # for k in range(c_pos.shape[1]):
        #     single_c_embed=torch.where(c_pos[:,k]==1,c_embed_cy[:,k,:],c_embed_cy[:,k+self.n_concepts,:])
        #     # print(single_c_embed.shape)
        #     single_c_embed=single_c_embed.unsqueeze(1)
        #     c_embed.append(single_c_embed)
        # c_embed=torch.cat(c_embed,dim=1)
        # c_embed=c_embed.view(bs,-1)
        # # print(c_embed.shape)
        # c_embed=self.concept_proj(c_embed)
        concepts = (c_pos > 0.5).long()
        if(len(concepts.shape) != len(c_embed_cy.shape)):
            concepts = torch.unsqueeze(concepts, dim = -1)
            concepts = concepts.expand_as(c_embed_cy[:, :self.n_concepts, :])
        c_embed_positive = concepts * c_embed_cy[:, :self.n_concepts, :]
        c_embed_negative = (1 - concepts) * c_embed_cy[:, self.n_concepts:, :]

        c_embed = c_embed_positive + c_embed_negative

        while len(mask.shape) < len(c_embed.shape):
            mask = torch.unsqueeze(mask, dim = -1)
        mask = mask.expand_as(c_embed).bool()

        c_embed = torch.where(mask, c_embed, 0)
        # for k in range(c_prob.shape[1]):
        #     single_c_embed=torch.where(c_prob[:,k,1:2]>0.5,c_embed_cy[:,k,:],c_embed_cy[:,k+self.n_concepts,:])
        #     single_c_embed=single_c_embed.unsqueeze(1)
        #     c_embed.append(single_c_embed)
        # c_embed=torch.cat(c_embed,dim=1).view(bs,-1)
        c_embed = c_embed.view(bs, -1)
        c_embed=self.concept_proj(c_embed)
        c_embed = c_embed[:,None,:].expand_as(y_embed) # [bs,label_size, hidden_size]
        cy_embed = c_embed * y_embed
        cy_embed = c_embed + cy_embed
        cy_embed = F.relu(cy_embed) # [bs,label_size, hidden_size]
        # c_y_energy_embed=cy_embed.view(bs*self.num_classes,-1)
        cy_energy = self.classifier_cy(cy_embed) # [bs, 1, 1]
        # do a class-wise energy transpose.
        cy_energy = cy_energy.view(bs, -1)

        # if not is_training:
        #     # print('x-y',x,'x-c',x_c,'c-y',c_proj)
        #     return cy_energy,(y_prob)
        # else:
        return cy_energy
        
    def _run_step(self, energy, y = None, train = False):
        # x -> concepts, m -> 0 = missing data, y = label

        # (batch_size, n_concepts)



        # what do we want to model here
        # p(x_u | x_o, y) = p(x_u, x_o | y) / p(x_o | y)

        # p(x_u, x_o | y) = e^(E(x_u + x_o, y)) / sum_y(E(x_u + x_o, y))

        # e^(E(x_u + x_o, y))
        
        if y is None:
            # p(c | y)
            batch_size=energy.size(0)

            class_weights = self.class_weights.to(energy.device)
            
            energy_sum = torch.sum(energy, dim=1, keepdim=True)

            energy = torch.exp(-energy)

            energy = energy * class_weights

            return energy / energy_sum

            # p(c | y) * p(y)


        else:
            batch_size=energy.size(0)
            y_tem = torch.tensor([self.class_list.index(tem) for tem in y]).long().to(self.device)
            y_tem = y_tem.view(batch_size, 1)
            energy_pos = energy.gather(dim=1, index=y_tem)
            
            energy_sum = torch.sum(energy, dim=1, keepdim=True)

            energy_pos = torch.exp(-energy_pos)

            return energy_pos / energy_sum
        

    def training_step(self, batch, batch_idx):
        x, b, m, y = batch['x'], batch['b'], batch['m'], batch['y']

        concepts = x

        energy = self.forward(concepts, m, train = True)
        predL = self._run_step(energy, y, train = True)

        loss = predL.mean()

        result = {
            "loss": loss.detach()
        }

        for name, val in result.items():
            self.log("train_" + name, val, prog_bar=("accuracy" in name), sync_dist = True)

        return {"loss": loss}
    
    def validation_step(self, batch, batch_idx):
        x, b, m, y = batch['x'], batch['b'], batch['m'], batch['y']

        concepts = x * m

        energy = self.forward(concepts, m, train = False)
        predL = self._run_step(energy, y, train = False)

        loss = predL.mean()

        result = {
            "loss": loss.detach()
        }

        test_result = self._test(x, b, m, y)

        for key, val in test_result.items():
            result[key] = val

        for name, val in result.items():
            self.log("val_" + name, val, prog_bar=("accuracy" in name), sync_dist = True)

        return {"loss": loss, **test_result}
    
    def _test(self, x, b, m, y):

        all_concepts = x

        all_concepts_energy = self(all_concepts, m, train = False)

        all_concepts_probabilities = self._run_step(all_concepts_energy, y, train = False)

        # logging.debug(
        #     f"all_concepts.shape: {all_concepts.shape}\n"
        #     f"all_concepts_energy.shape: {all_concepts_energy.shape}\n"
        #     f"all_concepts_probabilities.shape: {all_concepts_probabilities.shape}\n"
        # )

        # p(x_o | y)
        observed_concepts = x
        observed_concepts_energy = self(observed_concepts, m * b, train = False)
        observed_concepts_probabilities = self._run_step(observed_concepts_energy, y, train = False)

        # logging.debug(
        #     f"observed_concepts.shape: {all_concepts.shape}\n"
        #     f"observed_concepts_energy.shape: {all_concepts_energy.shape}\n"
        #     f"observed_concepts_probabilities.shape: {all_concepts_probabilities.shape}\n"
        # )

        # p(x_u | x_o, y)
        concept_probabilities = all_concepts_probabilities / observed_concepts_probabilities

        # p(x_o | y)
        reversed_concepts = 1 - x
        reversed_new_concept = reversed_concepts * torch.logical_xor(m, b)
        incorrect_all_concepts = reversed_new_concept + x * b
        incorrect_all_concepts_energy = self(incorrect_all_concepts, m, train = False)
        incorrect_all_concepts_probabilities = self._run_step(incorrect_all_concepts_energy, y, train = False)

        # p(x_u | x_o, y)
        incorrect_concept_probabilities = incorrect_all_concepts_probabilities / observed_concepts_probabilities

        # p(x_u | x_o)
        class_weights = torch.tile(torch.unsqueeze(self.class_weights, dim = 0), [x.shape[0], 1])

        concept_probabilities = concept_probabilities / (concept_probabilities + incorrect_concept_probabilities)

        acc = (concept_probabilities > 0.5).float().mean()

        result = {
            "accuracy": acc.detach()
        }

        return result
    
    def test_step(self, batch, batch_idx):
        x, b, m, y = batch['x'], batch['b'], batch['m'], batch['y']

        result = self._test(x, b, m, y)

        for name, val in result.items():
            self.log("test_" + name, val, prog_bar=("accuracy" in name), sync_dist = True)

        return result
    
    def compute_concept_probabilities(self, x, b, m, y):
        # p(x_o, x_u | y)
        all_concepts = x

        all_concepts_energy = self(all_concepts, m, train = False)

        all_concepts_probabilities = self._run_step(all_concepts_energy, y, train = False)

        # p(x_o | y)
        observed_concepts = x
        observed_concepts_energy = self(observed_concepts, m * b, train = False)

        observed_concepts_probabilities = self._run_step(observed_concepts_energy, y, train = False)

        # p(x_u | x_o, y)
        concept_probabilities = all_concepts_probabilities / observed_concepts_probabilities

        # p(x_u | x_o)
        if y is None:
            class_weights = torch.tile(torch.unsqueeze(self.class_weights, dim = 0), [x.shape[0], 1]).to(concept_probabilities.device)
            concept_probabilities = concept_probabilities * class_weights
    
        concept_probabilities = torch.sum(concept_probabilities, dim = 1)

        return concept_probabilities
    
    
    def configure_optimizers(self):
        if self.optimizer_name.lower() == "adam":
            optimizer = torch.optim.Adam(
                [self.y_embedding,
                self.c_embedding]
                + list(self.concept_proj.parameters())
                + list(self.classifier_cy.parameters()),
                lr=self.learning_rate,
                weight_decay=self.weight_decay,
            )
        else:
            optimizer = torch.optim.SGD(
                filter(lambda p: p.requires_grad, 
                [self.y_embedding,
                self.c_embedding]
                + list(self.concept_proj.parameters())
                + list(self.classifier_cy.parameters())),
                lr=self.learning_rate,
                momentum=self.momentum,
                weight_decay=self.weight_decay,
            )
        lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            verbose=True,
        )
        
        return {
            "optimizer": optimizer,
            "lr_scheduler": lr_scheduler,
            "monitor": "val_loss",
        }
        

        
# [0,1]^k concepts -> [0,1]^n tasks
# E(c,y) = concept embeddings + class embeddings -> energy 
# [batch_size, n_concepts, c_hidden_size] -> [batch_size, n_tasks, y_hidden_size] -> [batch_size, energy]
# n_concepts, c_hidden_size -> hidden_size
# n_tasks, y_hidden_size -> hidden_size 
# hidden_size -> 1