# Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# This work is licensed under a Creative Commons
# Attribution-NonCommercial-ShareAlike 4.0 International License.
# You should have received a copy of the license along with this
# work. If not, see http://creativecommons.org/licenses/by-nc-sa/4.0/

"""Loss functions used in the paper
"Elucidating the Design Space of Diffusion-Based Generative Models"."""

import torch
from torch_utils import persistence
import matplotlib.pyplot as plt
import torch.distributed as dist
#----------------------------------------------------------------------------
# Loss function corresponding to the variance preserving (VP) formulation
# from the paper "Score-Based Generative Modeling through Stochastic
# Differential Equations".

@persistence.persistent_class
class VPLoss:
    def __init__(self, beta_d=19.9, beta_min=0.1, epsilon_t=1e-5):
        self.beta_d = beta_d
        self.beta_min = beta_min
        self.epsilon_t = epsilon_t

    def __call__(self, net, images, labels, augment_pipe=None):
        rnd_uniform = torch.rand([images.shape[0], 1, 1, 1], device=images.device)
        sigma = self.sigma(1 + rnd_uniform * (self.epsilon_t - 1))
        weight = 1 / sigma ** 2
        y, augment_labels = augment_pipe(images) if augment_pipe is not None else (images, None)
        n = torch.randn_like(y) * sigma
        D_yn = net(y + n, sigma, labels, augment_labels=augment_labels)
        loss = weight * ((D_yn - y) ** 2)
        return loss

    def sigma(self, t):
        t = torch.as_tensor(t)
        return ((0.5 * self.beta_d * (t ** 2) + self.beta_min * t).exp() - 1).sqrt()

#----------------------------------------------------------------------------
# Loss function corresponding to the variance exploding (VE) formulation
# from the paper "Score-Based Generative Modeling through Stochastic
# Differential Equations".

@persistence.persistent_class
class VELoss:
    def __init__(self, sigma_min=0.02, sigma_max=100):
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max

    def __call__(self, net, images, labels, augment_pipe=None):
        rnd_uniform = torch.rand([images.shape[0], 1, 1, 1], device=images.device)
        sigma = self.sigma_min * ((self.sigma_max / self.sigma_min) ** rnd_uniform)
        weight = 1 / sigma ** 2
        y, augment_labels = augment_pipe(images) if augment_pipe is not None else (images, None)
        n = torch.randn_like(y) * sigma
        D_yn = net(y + n, sigma, labels, augment_labels=augment_labels)
        loss = weight * ((D_yn - y) ** 2)
        return loss

#----------------------------------------------------------------------------
# Improved loss function proposed in the paper "Elucidating the Design Space
# of Diffusion-Based Generative Models" (EDM).

@persistence.persistent_class
class EDMLoss:
    def __init__(self, P_mean=-1.2, P_std=1.2, sigma_data=.5): #cambiato a -1.4 per Bumberg refining
        self.P_mean = P_mean
        self.P_std = P_std
        self.sigma_data = sigma_data
        
    def __call__(self, net, images, labels=None, augment_pipe=None):
        rnd_normal = torch.randn([images.shape[0], 1, 1, 1], device=images.device)
        sigma = (rnd_normal * self.P_std + self.P_mean).exp()
        weight = (sigma ** 2 + self.sigma_data ** 2) / (sigma * self.sigma_data) ** 2
        y, augment_labels = augment_pipe(images) if augment_pipe is not None else (images, None)

        n = torch.randn_like(y) * sigma
        D_yn = net(y + n, sigma, labels, augment_labels=augment_labels)
        loss = weight * ((D_yn - y) ** 2)
        
        return loss


#----------------------------------------------------------------------------
# Improved loss function proposed in the paper "Elucidating the Design Space
# of Diffusion-Based Generative Models" (EDM) + loss for geostatistical parameters.

@persistence.persistent_class
class EDMLoss_geost:
    def __init__(self, P_mean=-1.2, P_std=1.2, sigma_data=.5): 
        self.P_mean = P_mean
        self.P_std = P_std
        self.sigma_data = sigma_data
        self.d_lbl = 2

    def __call__(self, net, images, labels=None, augment_pipe=None):
        rnd_normal = torch.randn([images.shape[0], 1, 1, 1], device=images.device)
        sigma = (rnd_normal * self.P_std + self.P_mean).exp()
        weight = (sigma ** 2 + self.sigma_data ** 2) / (sigma * self.sigma_data) ** 2
        
        #augment should be none for now 
        y, augment_labels = augment_pipe(images) if augment_pipe is not None else (images, None)
        
        #we want noise only on the dimension(s) corresponding to the output
        n = torch.randn_like(y) * sigma
        
        D_yn, geost_label = net(y + n, sigma, labels, augment_labels=augment_labels)
        loss1 = weight * ((D_yn - y) ** 2)
        
        #the weighting follows the same pattern of weight, or c_skip in network preconditioning (see karras or code) [Roberto Miele 2026]
        lbl_weight = 1+((self.sigma_data/self.d_lbl) ** 2 / (sigma.squeeze() ** 2 + (self.sigma_data/self.d_lbl) ** 2))
        
        loss2 = lbl_weight[:,None] * ((geost_label - labels) ** 2)
        
        #loss2 weights as much as an additional channel of image size
        return loss1, loss2

#------------------------Other loss functions-----------------------------------
  
#class EDMLoss_w:
#    """This loss is problematic: weights are changing the frequency of facies in the generated images"""
#    def __init__(self, P_mean=-1.6, P_std=1.3, sigma_data=1):
#        self.P_mean = P_mean
#        self.P_std = P_std
#        self.sigma_data = sigma_data
#        
#    def __call__(self, net, images, labels=None, augment_pipe=None):
#        rnd_normal = torch.randn([images.shape[0], 1, 1, 1], device=images.device)
#        sigma = (rnd_normal * self.P_std + self.P_mean).exp()
#        weight = (sigma ** 2 + self.sigma_data ** 2) / (sigma * self.sigma_data) ** 2
#
#        f_w = images[:,-1,None,:]
#        images = images[:,:-1]
#
#        y, augment_labels = augment_pipe(images) if augment_pipe is not None else (images, None)
#
#        n = torch.randn_like(y) * sigma
#        D_yn = net(y + n, sigma, labels, augment_labels=augment_labels)
#    
#        loss =  f_w * weight * ((D_yn - y) ** 2)
#
#        return loss
#

##----------------------------------------------------------------------------
##EDM loss with Ekblom distance rather than L2Norm
#
#@persistence.persistent_class
#class EDMLossEkblom:
#    def __init__(self, P_mean=-1.4, P_std=1.3, sigma_data=1, epsilon=.1):
#        """exponent assumed to be 1"""
#        self.P_mean = P_mean
#        self.P_std = P_std
#        self.sigma_data = sigma_data
#        self.epsilon = epsilon
#        self.numel = None
#
#    def __call__(self, net, images, labels=None, augment_pipe=None):
#        rnd_normal = torch.randn([images.shape[0], 1, 1, 1], device=images.device)
#        sigma = (rnd_normal * self.P_std + self.P_mean).exp()
#        weight = (sigma * sigma + self.sigma_data * self.sigma_data) / (sigma * self.sigma_data) ** 2
#        y, augment_labels = augment_pipe(images) if augment_pipe is not None else (images, None)
#
#        n = torch.randn_like(y) * sigma
#        D_yn = net(y + n, sigma, labels, augment_labels=augment_labels)
#        diff = (D_yn - y)
#        diff = diff * diff
#
#        if self.numel is None:
#            self.numel = diff.numel()
#            self.epsilon = self.epsilon/self.numel
#        
#        Ekblom = torch.sqrt(diff + self.epsilon*self.epsilon)
#        loss =  weight * Ekblom
#        
#        self.epsilon = Ekblom.min().item() #update the minimum of the loss function for the next iter
#        return loss
#
#
#class EDMLossHuber:
#    def __init__(self, P_mean=-1.2, P_std=1.2, sigma_data=1, delta=.5):
#        self.P_mean = P_mean
#        self.P_std = P_std
#        self.sigma_data = sigma_data
#        self.delta = delta
#        """
#        Pytorch implements it differently (it does not behave like L2)
#        z = (input - target).abs()
#        loss = torch.where(z < delta, 0.5 * z * z, delta * (z - 0.5 * delta))
#
#        Here is implemented so that if |x|<=delta then loss = x^2 else loss = 2*delta*|x|-delta^2 (basically the gradients are twice as much as normal)
#        """
#
#    def __call__(self, net, images, labels=None, augment_pipe=None):
#        rnd_normal = torch.randn([images.shape[0], 1, 1, 1], device=images.device)
#        sigma = (rnd_normal * self.P_std + self.P_mean).exp()
#        weight = (sigma ** 2 + self.sigma_data ** 2) / (sigma * self.sigma_data) ** 2
#        y, augment_labels = augment_pipe(images) if augment_pipe is not None else (images, None)
#
#        n = torch.randn_like(y) * sigma
#        D_yn = net(y + n, sigma, labels, augment_labels=augment_labels)
#        diff = (D_yn - y)
#
#        loss = torch.where(diff < self.delta, diff * diff, 2* self.delta * diff - self.delta*self.delta)
#
#        return loss