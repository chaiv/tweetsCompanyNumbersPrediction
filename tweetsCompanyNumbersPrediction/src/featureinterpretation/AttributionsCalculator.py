'''
Created on 24.02.2023

@author: vital
'''

from captum.attr import LayerIntegratedGradients

class AttributionsCalculator(object):
  
    def __init__(self, model,embedding):
        self.lig = LayerIntegratedGradients(model,
            embedding)

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
        attributions_ig = attributions_ig / attributions_ig.abs().max(dim=1, keepdim=True)[0]
        return attributions_ig