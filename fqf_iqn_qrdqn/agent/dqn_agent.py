import torch
from torch.optim import Adam

from fqf_iqn_qrdqn.model import DQN
from fqf_iqn_qrdqn.utils import update_params, disable_gradients

from .base_agent import BaseAgent


class DQNAgent(BaseAgent):

    def __init__(self, env, test_env, log_dir, num_steps=5*(10**7),
                 batch_size=32, lr=5e-5, memory_size=10**6,
                 gamma=0.99, multi_step=1, update_interval=4,
                 target_update_interval=10000, start_steps=50000,
                 epsilon_train=0.01, epsilon_eval=0.001,
                 epsilon_decay_steps=250000, double_q_learning=False,
                 dueling_net=False, noisy_net=False, use_per=False,
                 log_interval=100, eval_interval=250000, num_eval_steps=125000,
                 max_episode_steps=27000, grad_cliping=None, cuda=True,
                 seed=0):
        super(DQNAgent, self).__init__(
            env, test_env, log_dir, num_steps, batch_size, memory_size,
            gamma, multi_step, update_interval, target_update_interval,
            start_steps, epsilon_train, epsilon_eval, epsilon_decay_steps,
            double_q_learning, dueling_net, noisy_net, use_per, log_interval,
            eval_interval, num_eval_steps, max_episode_steps, grad_cliping,
            cuda, seed)

        # Online network.
        self.online_net = DQN(
            num_channels=env.observation_space.shape[0],
            num_actions=self.num_actions, dueling_net=dueling_net,
            noisy_net=noisy_net).to(self.device)
        # Target network.
        self.target_net = DQN(
            num_channels=env.observation_space.shape[0],
            num_actions=self.num_actions, dueling_net=dueling_net,
            noisy_net=noisy_net).to(self.device).to(self.device)

        # Copy parameters of the learning network to the target network.
        self.update_target()
        # Disable calculations of gradients of the target network.
        disable_gradients(self.target_net)

        self.optim = Adam(
            self.online_net.parameters(),
            lr=lr, eps=1e-2/batch_size)

    def learn(self):
        self.learning_steps += 1
        self.online_net.sample_noise()
        self.target_net.sample_noise()

        if self.use_per:
            (states, actions, rewards, next_states, dones), weights =\
                self.memory.sample(self.batch_size)
        else:
            states, actions, rewards, next_states, dones =\
                self.memory.sample(self.batch_size)
            weights = None

        loss, mean_q, errors = self.calculate_loss(
            states, actions, rewards, next_states, dones, weights)
        assert errors.shape == (self.batch_size, 1)

        update_params(
            self.optim, loss,
            networks=[self.online_net],
            retain_graph=False, grad_cliping=self.grad_cliping)

        if self.use_per:
            self.memory.update_priority(errors)

        if 4*self.steps % self.log_interval == 0:
            self.writer.add_scalar(
                'loss/q_loss', loss.detach().item(),
                4*self.steps)
            self.writer.add_scalar('stats/mean_Q', mean_q, 4*self.steps)

    def calculate_loss(self, states, actions, rewards, next_states, dones,
                       weights):

        q = self.online_net.calculate_q(states=states).gather(1, actions)

        with torch.no_grad():
            # Calculate Q values of next states.
            if self.double_q_learning:
                self.online_net.sample_noise()
                next_q = self.online_net.calculate_q(states=next_states)
                action_index = next_q.max(dim=1)[1].view(-1, 1)
                # choose related Q from target net
                next_q = self.target_net.calculate_q(states=next_states).gather(dim=1, index=action_index.long())
            else:
                next_q = self.target_net.calculate_q(states=next_states).detach().max(1)[0].unsqueeze(1)
            
            q_target = rewards + self.gamma_n * next_q * (1 - dones)

        if weights is not None:
            loss = (((q - q_target) ** 2) * weights).mean()
        else:
            criterion = torch.nn.MSELoss()
            loss = criterion(q, q_target)

        td_errors = q_target - q

        return loss, next_q.detach().mean().item(), td_errors.detach().abs()
