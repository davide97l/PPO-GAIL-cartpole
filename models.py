import torch
from torch import nn
from torch.distributions import Categorical
from torch.nn import functional as F


# Concatenates the state and one-hot version of an action
def _join_state_action(state, action, action_size):
    return torch.cat([state, F.one_hot(action, action_size).to(dtype=torch.float32)], dim=1)


class Actor(nn.Module):
    def __init__(self, state_size, action_size, hidden_size, dropout=0):
        super().__init__()
        if dropout > 0:
            self.actor = nn.Sequential(nn.Linear(state_size, hidden_size), nn.Dropout(p=dropout), nn.Tanh(), nn.Linear(hidden_size, hidden_size), nn.Dropout(p=dropout), nn.Tanh(), nn.Linear(hidden_size, action_size))
        else:
            self.actor = nn.Sequential(nn.Linear(state_size, hidden_size), nn.Tanh(), nn.Linear(hidden_size, hidden_size), nn.Tanh(), nn.Linear(hidden_size, action_size))

    def forward(self, state):
        policy = Categorical(logits=self.actor(state))
        return policy

    # Calculates the log probability of an action a with the policy π(·|s) given state s
    def log_prob(self, state, action):
        return self.forward(state).log_prob(action)


class Critic(nn.Module):
    def __init__(self, state_size, hidden_size):
        super().__init__()
        self.critic = nn.Sequential(nn.Linear(state_size, hidden_size), nn.Tanh(), nn.Linear(hidden_size, hidden_size), nn.Tanh(), nn.Linear(hidden_size, 1))

    def forward(self, state):
        value = self.critic(state).squeeze(dim=1)
        return value


class ActorCritic(nn.Module):
    def __init__(self, state_size, action_size, hidden_size):
        super().__init__()
        self.actor = Actor(state_size, action_size, hidden_size)
        self.critic = Critic(state_size, hidden_size)

    def forward(self, state):
        policy, value = self.actor(state), self.critic(state)
        return policy, value

    # Calculates the log probability of an action a with the policy π(·|s) given state s
    def log_prob(self, state, action):
        return self.actor.log_prob(state, action)


class GAILDiscriminator(nn.Module):
    def __init__(self, state_size, action_size, hidden_size, state_only=False):
        super().__init__()
        self.action_size, self.state_only = action_size, state_only
        input_layer = nn.Linear(state_size if state_only else state_size + action_size, hidden_size)
        self.discriminator = nn.Sequential(input_layer, nn.Tanh(), nn.Linear(hidden_size, hidden_size), nn.Tanh(), nn.Linear(hidden_size, 1), nn.Sigmoid())

    def forward(self, state, action):
        return self.discriminator(state if self.state_only else
                                  _join_state_action(state, action, self.action_size)).squeeze(dim=1)

    def predict_reward(self, state, action):
        D = self.forward(state, action)
        h = torch.log(D) - torch.log1p(-D)
        return h
