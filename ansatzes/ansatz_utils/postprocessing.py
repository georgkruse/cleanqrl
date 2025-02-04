import torch
from gymnasium.spaces import Box, Discrete


def postprocessing(prediction, config, action_space, weights, type):

    if type == 'actor' and config['use_output_scaling_actor']:
        scaling = weights[f'output_scaling_{type}']
    elif type in ['es', 'ga']:
        if config['use_output_scaling_actor']:
            if isinstance(config['init_output_scaling_actor'], list):
                if isinstance(action_space, Box):
                    scaling = weights[-action_space.shape[0]*2:]
                elif isinstance(action_space, Discrete):
                    scaling = weights[-action_space.n:]
            else:
                scaling = weights[-1]

    if type == 'critic':
        if config['postprocessing_critic'] == 'default':
            value = prediction*weights[f'output_scaling_{type}'][0] 
        elif config['postprocessing_critic'] == 'weighted_sum':
            if len(prediction.shape) > 1:
                if prediction.shape[1] > 1:
                    value = prediction*weights[f'output_scaling_{type}'][:prediction.shape[1]] 
                    value = torch.sum(value, dim=1)
                else:
                    value = prediction*weights[f'output_scaling_{type}'][0] 
            elif len(prediction.shape) == 0:
                value = prediction*weights[f'output_scaling_{type}'][0]
            else:
                if prediction.shape[0] <= weights[f'output_scaling_{type}'].shape[0]:
                    value = prediction*weights[f'output_scaling_{type}'][:prediction.shape[0]] 
                    value = torch.sum(value)
                else:
                    value = prediction*weights[f'output_scaling_{type}'][0] 
            # value = torch.sum(value) + weights[f'output_scaling_{type}_bias']
        elif config['postprocessing_critic'] == 'weighted_sum_triple':
            value = prediction*weights[f'output_scaling_{type}'][:3] 
            value = torch.sum(value)
            # value = torch.sum(value) + weights[f'output_scaling_{type}_bias']
        elif config['postprocessing_critic'] == 'relu':
            value = prediction*weights[f'output_scaling_{type}'][0]
            # value = torch.sum(value) + weights[f'output_scaling_{type}_bias']
        # elif config['postprocessing_critic'] == 'single_relu':
        #     value = prediction*weights[f'output_scaling_{type}'][0:4]
            # value = torch.sum(value) + weights[f'output_scaling_{type}_bias']
        else:
            value = prediction[0]*weights[f'output_scaling_{type}'] 
        return value
    
    elif type == 'actor':

        if config['postprocessing_actor'] == 'standard':
            if len(prediction.shape) > 1:
                pred = prediction[:,:action_space]*scaling[:action_space]
            else:
                pred = prediction[:action_space]*scaling[:action_space]

        elif config['postprocessing_actor'] == 'plain_probs':
            pred = prediction
        
        elif config['postprocessing_actor'] == 'constant':
            scaling.requires_grad = False
            if len(prediction.shape) > 1:
                pred = prediction[:,:action_space]*scaling[:action_space]
            else:
                pred = prediction[:action_space]*scaling[:action_space]

        elif config['postprocessing_actor'] == 'sqrt+constant':
            scaling.requires_grad = False
            if len(prediction.shape) > 1:
                pred = prediction[:,:action_space]*scaling[:action_space]
            else:
                pred = prediction[:action_space]*scaling[:action_space]

        elif config['postprocessing_actor'] == 'sqrt+standard':
            if len(prediction.shape) > 1:
                pred = prediction[:,:action_space]*scaling[:action_space]
            else:
                pred = prediction[:action_space]*scaling[:action_space]

    return pred 

