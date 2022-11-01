import abc
import itertools
from typing import Any
from torch import nn
from torch.nn import functional as F
from torch import optim

import numpy as np
import torch
from torch import distributions

from cs285.infrastructure import pytorch_util as ptu
from cs285.policies.base_policy import BasePolicy


class MLPPolicy(BasePolicy, nn.Module, metaclass=abc.ABCMeta):

    def __init__(self,
                 ac_dim,
                 ob_dim,
                 n_layers,
                 size,
                 discrete=False,
                 learning_rate=1e-4,
                 training=True,
                 nn_baseline=False,
                 **kwargs
                 ):
        super().__init__(**kwargs)

        # init vars
        self.ac_dim = ac_dim
        self.ob_dim = ob_dim
        self.n_layers = n_layers
        self.discrete = discrete
        self.size = size
        self.learning_rate = learning_rate
        self.training = training
        self.nn_baseline = nn_baseline

        if self.discrete:
            self.logits_na = ptu.build_mlp(
                input_size=self.ob_dim,
                output_size=self.ac_dim,
                n_layers=self.n_layers,
                size=self.size,
            )
            self.logits_na.to(ptu.device)
            self.mean_net = None
            self.logstd = None
            self.optimizer = optim.Adam(self.logits_na.parameters(),
                                        self.learning_rate)
        else:
            self.logits_na = None
            self.mean_net = ptu.build_mlp(
                input_size=self.ob_dim,
                output_size=self.ac_dim,
                n_layers=self.n_layers,
                size=self.size,
            )
            self.mean_net.to(ptu.device)
            self.logstd = nn.Parameter(
                torch.zeros(self.ac_dim, dtype=torch.float32, device=ptu.device)
            )
            self.logstd.to(ptu.device)
            self.optimizer = optim.Adam(
                itertools.chain([self.logstd], self.mean_net.parameters()),
                self.learning_rate
            )
            print(f"created mlp: {self.mean_net}")

    ##################################

    def save(self, filepath):
        torch.save(self.state_dict(), filepath)

    ##################################

    def get_action(self, obs: np.ndarray) -> np.ndarray:
        if len(obs.shape) > 1:
            observation = obs
        else:
            observation = obs[None]

        # TODO return the action that the policy prescribes
        return ptu.to_numpy(self.forward(ptu.from_numpy(observation)))
        raise NotImplementedError

    # update/train this policy
    def update(self, observations, actions, **kwargs):
        raise NotImplementedError

    # This function defines the forward pass of the network.
    # You can return anything you want, but you should be able to differentiate
    # through it. For example, you can return a torch.FloatTensor. You can also
    # return more flexible objects, such as a
    # `torch.distributions.Distribution` object. It's up to you!
    def forward(self, observation: torch.FloatTensor) -> Any:
        if not isinstance(observation, torch.FloatTensor):
            raise ValueError('observation should be of instance torch.FloatTensor not : {}'.format(type(observation)))
        if self.discrete:
            #print("discrete model")
            actions = self.logits_na.forward(observation)
        else:
            #print("continuous model")
            actions = self.mean_net.forward(observation)

        #print(f"actions: {actions}")
        #rint(f"actions size: {actions.size()}")

        return actions
        raise NotImplementedError


#####################################################
#####################################################

class MLPPolicySL(MLPPolicy):
    def __init__(self, ac_dim, ob_dim, n_layers, size, **kwargs):
        super().__init__(ac_dim, ob_dim, n_layers, size, **kwargs)
        self.loss = nn.MSELoss()

    def update(
            self, observations, actions,
            adv_n=None, acs_labels_na=None, qvals=None
    ):
        # TODO: update the policy and return the loss
        if not isinstance(observations, torch.FloatTensor):
            observations = torch.from_numpy(observations)
        if not isinstance(actions, torch.FloatTensor):
            actions = torch.from_numpy(actions)

        #print(f'logits_na: "{self.logits_na}"')
        #print(f'mean_net: "{self.mean_net}"')
        #print(f'mean_net layers: "{self.mean_net.layers}"')
        #print(f'mean_net out layer: "{self.mean_net.layers[-1]}"')
        #print(f'mean_net out layer weights before step: "{self.mean_net.layer_1.weight[0:5,0:5]}"')

        #print(f'mean_net parameters: {self.mean_net.named_parameters()}')
        #for name, p in self.mean_net.named_parameters():
        #    print(f'name: {name}, shape: {p.shape}, gradient: {p.grad}')

        #print(f"observations: {observations}")
        #print(f"observations size: {observations.size()}")

        policy_actions = self.forward(observations)

        #print(f"policy_actions size: {policy_actions.size()}")
        #print(f"policy_actions[0:5,0:5]: {policy_actions[0:5,0:5]}")
        #print(f"policy_actions size: {policy_actions.size()}")

        #print(f"target actions size: {actions.size()}")
        #print(f"target actions[0:5,0:5]: {actions[0:5,0:5]}")

        loss = self.loss(policy_actions, actions)

        #print(f'type(loss): {type(loss)}')
        #print(f'loss size: {loss.size()}')
        #print(f'loss: {loss}')

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        #print(f'mean_net out layer weight after step: "{self.mean_net.layer_1.weight[0:5,0:5]}"')

        return {
            # You can add extra logging information here, but keep this line
            'Training Loss': ptu.to_numpy(loss),
        }
