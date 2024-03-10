import torch
import logging

import pytorch_lightning as pl
import torch.nn.functional as F
import numpy as np

def generate_random_numbers(n,n_concept):
    numbers = set()
    while len(numbers) < float(n):
        numbers.add(random.randint(0, n_concept-1))
    return list(numbers)


# y: [batch_size, n_tasks, hidden_size]
# c: [batch_size, n_concepts, hidden_size]

# c -> y energy




class EBM(pl.LightningModule):
    
    # def __init__(self,args,num_classes, input_size=1000, hid_size=1000, cpt_size=None):
    def __init__(self,n_concepts, n_tasks, c_embed_size, y_embed_size, energy_model_architecture = "linear", optimizer = None, learning_rate = None, weight_decay = None, momentum = None):
        super().__init__()

        self.n_concepts = n_concepts
        self.n_tasks = n_tasks
        self.c_embed_size = c_embed_size
        self.y_embed_size = y_embed_size

        self.cy_perturb_prob = cy_perturb_prob
        self.cy_permute_prob = cy_permute_prob

        self.y_prob = torch.nn.Parameter(torch.randn((self.n_tasks)))
        self.y_embedding = torch.nn.Parameter(torch.randn((self.n_tasks, y_embed_size)))
        
        self.c_prob = torch.nn.Parameter(torch.randn((self.n_concepts)))
        self.c_embedding = torch.nn.Parameter(torch.randn((self.n_concepts*2, y_embed_size)))

        self.classifier_cy = torch.nn.Linear(embed_size, 1)
        self.smx_y = torch.nn.Softmax(dim=-2)
        self.smx_c = torch.nn.Softmax(dim=-1)
        self.dropout = torch.nn.Dropout(p=0.2)

            
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
        permute_concept_idx=torch.tensor(generate_random_numbers(permute_concept_number,all_length)).cuda()
        permute_samps=torch.tensor(generate_random_numbers(permute_sample_number,bs)).cuda()
        c_gt=c_gt.long()
        to_be_interf=c_gt[permute_samps]
        to_be_interf[:,permute_concept_idx]=to_be_interf[:,permute_concept_idx]^1
        for idx,permidx in enumerate(permute_samps):
            c_gt[permidx]=to_be_interf[idx]
        c_gt=c_gt.unsqueeze(-1)

        return c_gt


        

    def forward(self, x, c_gt, is_training,use_cy=True):
        # input x is encoded image.
        bs = x.shape[0]

        #### X->Y energy ###
        # project y into the embedding space to calculate class-wise energy
        y_embed=self.y_embedding.unsqueeze(0) # [1,label_size, hidden_size]
        y_embed=y_embed.repeat(bs,1,1) # [bs,label_size,hidden_size]

        if not is_training:
            y_prob=self.y_prob
            y_prob=self.smx(y_prob)
            y_embed=torch.sum(y_prob*y_embed,dim=1) # [bs,hidden_size]

        y_embed= F.normalize(y_embed, p=2, dim=-1)        
        # project x into the same space to calculate energy
        x_embed = self.xy_fc1(x) # [bs, hidden_size]
        x_embed = self.dropout(x_embed)
        if is_training:
            x_embed = x_embed[:,None,:].expand_as(y_embed) # [bs,label_size, hidden_size]
        # z: x->y energy
        z_xy = x_embed * y_embed
        z_xy = x_embed + z_xy
        z_xy = F.relu(z_xy)
        # reduce "energy embedding to 1 dim"
        xy_energy = self.classifier_xy(z_xy) # [bs, label_size, 1]
        # do a class-wise energy transpose.
        xy_energy = xy_energy.view(bs, -1)


        
        #### x->c energy ####
        x_embed = self.xc_fc1(x) # [bs, hidden_size]
        x_embed = self.dropout(x_embed)
        c_embed=self.c_embedding.unsqueeze(0) # [1,concept_size, hidden_size]
        c_embed=c_embed.repeat(bs,1,1) # [bs,concept_size,hidden_size]
        c_embed_cy=c_embed
        if not is_training:
            c_prob=self.c_prob
            c_prob=self.smx_c(c_prob)
            c_embed=c_embed[:,:self.n_concepts]*c_prob[:,:,0:1]+c_embed[:,self.n_concepts:]*c_prob[:,:,1:2]
            # print(c_embed.shape)
        c_embed= F.normalize(c_embed, p=2, dim=-1)
        x_embed = x_embed[:,None,:].expand_as(c_embed) # [bs,concept_size*2, hidden_size]

        xc_energy=[]
        for i in range(self.n_concepts):
            if not is_training:
                # print(c_embed[:,i].shape)
                xc_embed = x_embed[:,i] * c_embed[:,i]
                xc_embed = x_embed[:,i] + xc_embed
                xc_embed = F.relu(xc_embed)
                xc_energy_single =self.classifier_xc[i](xc_embed) # [bs,hidden_size] -> [bs,1]
                xc_energy_single = xc_energy_single.view(bs, -1)
                xc_energy_single = xc_energy_single.unsqueeze(1) # [bs,1,1]
            else:
                # z: x->y energy
                # print(x_c_pos.shape,c_embed[:,i,:self.hid_size].shape)
                xc_pos_embed = x_embed[:,i] * c_embed[:,i]
                xc_pos_embed = x_embed[:,i] + xc_pos_embed
                xc_pos_embed = F.relu(xc_pos_embed)

                xc_neg_embed =  x_embed[:,i+self.n_concepts] * c_embed[:,i+self.n_concepts]
                xc_neg_embed = x_embed[:,i+self.n_concepts] + xc_neg_embed
                xc_neg_embed = F.relu(xc_neg_embed)
                xc_embed = torch.stack([xc_pos_embed,xc_neg_embed],dim=1) # [bs, 2, hidden_size]

                xc_energy_single = self.classifier_xc[i](xc_embed) # [bs, 2, 1]
                xc_energy_single = xc_energy_single.view(bs, -1)
                xc_energy_single = xc_energy_single.unsqueeze(1)
            xc_energy.append(xc_energy_single)
        xc_energy=torch.cat(xc_energy,dim=1)

            

        #### c->y energy.####
        if use_cy:
            if not is_training:
                c_embed=[]
                for k in range(c_prob.shape[1]):
                    single_c_embed=torch.where(c_prob[:,k,1:2]>0.5,c_embed_cy[:,k,:],c_embed_cy[:,k+self.n_concepts,:])
                    single_c_embed=single_c_embed.unsqueeze(1)
                    c_embed.append(single_c_embed)
                c_embed=torch.cat(c_embed,dim=1).view(bs,-1)
                c_embed=self.concept_proj(c_embed)
                # c_single_embed = c_single_embed[:,None,:].expand_as(y_embed) # [bs,label_size, hidden_size]
                cy_embed = c_embed * y_embed
                cy_embed = c_embed + cy_embed
                cy_embed = F.relu(cy_embed) # [bs,label_size, hidden_size]
                # c_y_energy_embed=z_cy.view(bs*self.num_classes,-1)
                cy_energy = self.classifier_cy(cy_embed) # [bs, 1, 1]
                # do a class-wise energy transpose.
                cy_energy = cy_energy.view(bs, -1)

            else:
                c_pos=c_gt
                c_pos=c_pos.unsqueeze(-1)
                c_embed=[]
                c_pos=self.cy_augment(c_gt=c_pos,permute_ratio=self.cy_concept_perturb_prob,permute_prob=self.cy_sample_perturb_prob)
                for k in range(c_pos.shape[1]):
                    single_c_embed=torch.where(c_pos[:,k]==1,c_embed_cy[:,k,:],c_embed_cy[:,k+self.n_concepts,:])
                    # print(single_c_embed.shape)
                    single_c_embed=single_c_embed.unsqueeze(1)
                    c_embed.append(single_c_embed)
                c_embed=torch.cat(c_embed,dim=1)
                c_embed=c_embed.view(bs,-1)
                # print(c_embed.shape)
                c_embed=self.concept_proj(c_embed)
                c_embed = c_embed[:,None,:].expand_as(y_embed) # [bs,label_size, hidden_size]
                cy_embed = c_embed * y_embed
                cy_embed = c_embed + cy_embed
                cy_embed = F.relu(cy_embed) # [bs,label_size, hidden_size]
                # c_y_energy_embed=cy_embed.view(bs*self.num_classes,-1)
                cy_energy = self.classifier_cy(cy_embed) # [bs, 1, 1]
                # do a class-wise energy transpose.
                cy_energy = cy_energy.view(bs, -1)

        else:
            cy_energy=None

        if not is_training:
            # print('x-y',x,'x-c',x_c,'c-y',c_proj)
            return xy_energy,cy_energy,xc_energy,(y_prob,c_prob)
        else:
            return xy_energy,cy_energy,xc_energy