'''
Created on 24.02.2023

@author: vital
'''

from captum.attr import LayerIntegratedGradients

class AttributionsCalculator(object):
  
    def __init__(self, model,embedding, normalize=False):
        self.lig = LayerIntegratedGradients(model,
            embedding)
        self.normalize = normalize
        self.last_convergence_delta = None

    def attribute(self, x, ref, n_steps, observed_class, internal_batch_size):
        attributions_ig, delta = self.lig.attribute(
            x,
            ref,
            n_steps=n_steps,
            return_convergence_delta=True,
            target=observed_class,
            internal_batch_size=internal_batch_size
        )
        attributions_ig = attributions_ig[:, :, :].sum(dim=-1).cpu()
        self.last_convergence_delta = delta.detach().cpu()
        if self.normalize:
            denominator = attributions_ig.abs().max(dim=1, keepdim=True)[0].clamp(min=1e-12)
            attributions_ig = attributions_ig / denominator
        return attributions_ig
