from torch import nn

from .base_model import BaseModel
from fqf_iqn_qrdqn.network import DQNBase, NoisyLinear


class DQN(BaseModel):

    def __init__(self, num_channels, num_actions, embedding_dim=7*7*64,
                 dueling_net=False, noisy_net=False):
        super(DQN, self).__init__()
        linear = NoisyLinear if noisy_net else nn.Linear

        # Feature extractor of DQN.
        self.dqn_net = DQNBase(num_channels=num_channels)
        # Quantile network.
        if not dueling_net:
            self.q_net = nn.Sequential(
                linear(embedding_dim, 512),
                nn.ReLU(),
                linear(512, num_actions),
            )
        else:
            self.advantage_net = nn.Sequential(
                linear(embedding_dim, 512),
                nn.ReLU(),
                linear(512, num_actions),
            )
            self.baseline_net = nn.Sequential(
                linear(embedding_dim, 512),
                nn.ReLU(),
                linear(512, 1),
            )

        self.num_channels = num_channels
        self.num_actions = num_actions
        self.embedding_dim = embedding_dim
        self.dueling_net = dueling_net
        self.noisy_net = noisy_net

    def forward(self, states=None, state_embeddings=None):
        assert states is not None or state_embeddings is not None
        batch_size = states.shape[0] if states is not None\
            else state_embeddings.shape[0]

        if state_embeddings is None:
            state_embeddings = self.dqn_net(states)

        if not self.dueling_net:
            x = self.q_net(state_embeddings)
        else:
            advantages = self.advantage_net(state_embeddings)
            baselines = self.baseline_net(state_embeddings)
            x = baselines + advantages - advantages.mean(dim=1, keepdim=True)

        return x

    def calculate_q(self, states=None, state_embeddings=None):
        assert states is not None or state_embeddings is not None
        batch_size = states.shape[0] if states is not None\
            else state_embeddings.shape[0]

        # forwarding
        q = self(states=states, state_embeddings=state_embeddings)

        return q
