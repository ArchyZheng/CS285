from typing import Optional, Sequence
import numpy as np
import torch

from cs285.networks.policies import MLPPolicyPG
from cs285.networks.critics import ValueCritic
from cs285.infrastructure import pytorch_util as ptu
from torch import nn


class PGAgent(nn.Module):
    def __init__(
            self,
            ob_dim: int,
            ac_dim: int,
            discrete: bool,
            n_layers: int,
            layer_size: int,
            gamma: float,
            learning_rate: float,
            use_baseline: bool,
            use_reward_to_go: bool,
            baseline_learning_rate: Optional[float],
            baseline_gradient_steps: Optional[int],
            gae_lambda: Optional[float],
            normalize_advantages: bool,
    ):
        super().__init__()

        # create the actor (policy) network
        self.actor = MLPPolicyPG(
            ac_dim, ob_dim, discrete, n_layers, layer_size, learning_rate
        )

        # create the critic (baseline) network, if needed
        if use_baseline:
            self.critic = ValueCritic(
                ob_dim, n_layers, layer_size, baseline_learning_rate
            )
            self.baseline_gradient_steps = baseline_gradient_steps
            self.count = 0
        else:
            self.critic = None

        # other agent parameters
        self.gamma = gamma
        self.use_reward_to_go = use_reward_to_go
        self.gae_lambda = gae_lambda
        self.normalize_advantages = normalize_advantages

    def update(
            self,
            obs: Sequence[np.ndarray],
            actions: Sequence[np.ndarray],
            rewards: Sequence[np.ndarray],
            terminals: Sequence[np.ndarray],
    ) -> dict:
        """The train step for PG involves updating its actor using the given observations/actions and the calculated
        qvals/advantages that come from the seen rewards.

        Each input is a list of NumPy arrays, where each array corresponds to a single trajectory. The batch size is the
        total number of samples across all trajectories (i.e. the sum of the lengths of all the arrays).
        """

        # step 1: calculate Q values of each (s_t, a_t) point, using rewards (r_0, ..., r_t, ..., r_T)
        q_values: Sequence[np.ndarray] = self._calculate_q_vals(rewards)

        obs = np.concatenate(obs)
        rewards = np.concatenate(rewards)
        actions = np.concatenate(actions)
        q_values = np.concatenate(q_values)
        terminals = np.concatenate(terminals)

        # step 2: calculate advantages from Q values
        advantages: np.ndarray = self._estimate_advantage(
            obs, rewards, q_values, terminals
        )

        # step 3: use all datapoints (s_t, a_t, adv_t) to update the PG actor/policy
        # update the PG actor/policy network once using the advantages
        self.actor.optimizer.zero_grad()
        obs = torch.from_numpy(obs).float().to(ptu.device)
        actions = torch.from_numpy(actions).float().to(ptu.device)
        advantages = torch.from_numpy(advantages).float().to(ptu.device)
        loss = -1 * self.actor.forward(obs).log_prob(actions) * advantages
        loss = loss.mean()
        loss.backward()
        self.actor.optimizer.step()
        info: dict = {"loss:": loss}

        # step 4: if needed, use all datapoints (s_t, a_t, q_t) to update the PG critic/baseline
        if self.critic is not None:
            # TODO: perform `self.baseline_gradient_steps` updates to the critic/baseline network
            if self.count % self.baseline_gradient_steps == 0:
                critic_info: dict = self.critic.update(obs, q_values)
                info.update(critic_info)
            self.count += 1

        return info

    def _calculate_q_vals(self, rewards: Sequence[np.ndarray]) -> Sequence[np.ndarray]:
        """Monte Carlo estimation of the Q function."""

        if not self.use_reward_to_go:
            # Case 1: in trajectory-based PG, we ignore the timestep and instead use the discounted return for the entire
            # trajectory at each point.
            # In other words: Q(s_t, a_t) = sum_{t'=0}^T gamma^t' r_{t'}
            q_values_output = []
            for reward in rewards:
                temp = 0.0
                for i in range(1, len(reward) + 1):
                    temp += reward[-1 * i]
                    if i < len(reward):
                        temp *= self.gamma
                q_values = np.ones_like(reward) * temp
                q_values_output.append(q_values)

        else:
            # Case 2: in reward-to-go PG, we only use the rewards after timestep t to estimate the Q-value for (s_t, a_t).
            # In other words: Q(s_t, a_t) = sum_{t'=t}^T gamma^(t'-t) * r_{t'}
            q_values_output = []
            for reward in rewards:
                q_values = np.zeros_like(reward)
                temp = 0.0
                for i in range(1, len(reward) + 1):
                    temp += reward[-1 * i]
                    q_values[-1 * i] = temp
                    temp *= self.gamma
                q_values_output.append(q_values)

        return q_values_output

    def _estimate_advantage(
            self,
            obs: np.ndarray,
            rewards: np.ndarray,
            q_values: np.ndarray,
            terminals: np.ndarray,
    ) -> np.ndarray:
        """Computes advantages by (possibly) subtracting a value baseline from the estimated Q-values.

        Operates on flat 1D NumPy arrays.
        """
        if self.critic is None:
            advantages = q_values
        else:
            # TODO: run the critic and use it as a baseline
            obs = torch.from_numpy(obs).float().to(device=ptu.device)
            values = self.critic.forward(obs).squeeze(1)
            values = values.detach().cpu().numpy()
            assert values.shape == q_values.shape

            if self.gae_lambda is None:
                # TODO: if using a baseline, but not GAE, what are the advantages?
                advantages = q_values - values
            else:
                # TODO: implement GAE
                batch_size = obs.shape[0]

                # HINT: append a dummy T+1 value for simpler recursive calculation
                values = np.append(values, [0])
                advantages = np.zeros(batch_size + 1)

                for i in reversed(range(batch_size)):
                    # TODO: recursively compute advantage estimates starting from timestep T.
                    # HINT: use terminals to handle edge cases. terminals[i] is 1 if the state is the last in its
                    # trajectory, and 0 otherwise.
                    # This is very hard part. I can not see how to deal with it.
                    if terminals[i] == 1:
                        delta_t = rewards[i] - values[i]
                        advantages[i] = delta_t
                    else:
                        delta_t = rewards[i] + self.gamma * values[i] - values[i + 1]
                        advantages[i] = delta_t + self.gamma * self.gae_lambda * advantages[i + 1]

                # remove dummy advantage
                advantages = advantages[:-1]

        # normalize the advantages to have a mean of zero and a standard deviation of one within the batch
        if self.normalize_advantages:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-10)

        return advantages

    def _discounted_return(self, rewards: Sequence[float]) -> Sequence[float]:
        """
        Helper function which takes a list of rewards {r_0, r_1, ..., r_t', ... r_T} and returns
        a list where each index t contains sum_{t'=0}^T gamma^t' r_{t'}

        Note that all entries of the output list should be the exact same because each sum is from 0 to T (and doesn't
        involve t)!
        """
        return None

    def _discounted_reward_to_go(self, rewards: Sequence[float]) -> Sequence[float]:
        """
        Helper function which takes a list of rewards {r_0, r_1, ..., r_t', ... r_T} and returns a list where the entry
        in each index t' is sum_{t'=t}^T gamma^(t'-t) * r_{t'}.
        """
        return None
