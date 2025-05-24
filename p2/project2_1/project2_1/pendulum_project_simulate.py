from __future__ import annotations
import numpy as np
import torch
import torch.nn as nn
from torch.distributions.normal import Normal

import gymnasium as gym 
from env import CustomWrapper


class Agent:
    """Agent that learns to solve the Inverted Pendulum task using a policy gradient algorithm.
    The agent utilizes a policy network to sample actions and update its policy based on
    collected rewards.
    """

    def __init__(self, obs_space_dims: int, action_space_dims: int):
        """Initializes the agent with a neural network policy.
        
        Args:
            obs_space_dims (int): Dimension of the observation space.
            action_space_dims (int): Dimension of the action space.
        """

        # Hyperparameters
        self.learning_rate = 2e-4  # Learning rate for policy optimization
        self.gamma = 0.99  # Discount factor
        self.eps = 1e-6  # small number for mathematical stability

        self.probs = []  # Stores probability values of the sampled action
        self.rewards = []  # Stores the corresponding rewards

        self.policy_network = Policy_Network(obs_space_dims, action_space_dims)
        self.optimizer = torch.optim.AdamW(self.policy_network.parameters(), lr=self.learning_rate)

    def sample_action(self, state: np.ndarray) -> float:
        """Returns an action, conditioned on the policy and observation.

        Args:
            state: Observation from the environment

        Returns:
            action: Action to be performed
        """
        state = torch.tensor(np.array([state]))
        action_means, action_stddevs = self.policy_network(state)

        distrib = Normal(action_means[0] + self.eps, action_stddevs[0] + self.eps)
        action = distrib.sample()
        prob = distrib.log_prob(action)

        action = action.numpy()
        self.probs.append(prob)

        return action

    def update(self):
        """Updates the policy network using the REINFORCE algorithm based on collected rewards and log probabilities.
        
        Args:
            rewards (list): Collected rewards from the environment.
            log_probs (list): Log probabilities of the actions taken.
        """
        # The actual implementation of the REINFORCE update will be done here.
        running_g = 0     # Initialize the current cumulative return
        gs = []        # Discounted return

        # Discounted return (backwards)
        for R in self.rewards[::-1]:
            running_g = R + self.gamma * running_g
            gs.insert(0, running_g)

        deltas = torch.tensor(gs)

        # Calculate loss
        loss = 0
        for log_prob, delta in zip(self.probs, deltas):
            loss += log_prob.mean() * delta * (-1)

        # Update the policy network
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # Empty out all episode related variables
        self.probs = []
        self.rewards = [] 
        


class Policy_Network(nn.Module):
    """Parametrized Policy Network."""

    def __init__(self, obs_space_dims: int, action_space_dims: int):
        """Initializes layers of the neural network.
        
        Args:
            obs_space_dims (int): Dimension of the observation space.
            action_space_dims (int): Dimension of the action space.
        """
        super().__init__()

        hidden_space1 = 16  # Nothing special with 16, feel free to change
        hidden_space2 = 32  # Nothing special with 32, feel free to change

        # Define the neural network layers here
        hidden_space1 = 16
        hidden_space2 = 32

        self.fc1 = nn.Linear(obs_space_dims, hidden_space1)
        self.tanh1 = nn.Tanh()
        self.fc2 = nn.Linear(hidden_space1, hidden_space2)
        self.tanh2 = nn.Tanh()

        self.policy_mean_fc = nn.Linear(hidden_space2, action_space_dims)

        self.policy_stddev_fc = nn.Linear(hidden_space2, action_space_dims)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Predicts parameters of the action distribution given the state.
        
        Args:
            x (torch.Tensor): The state observation.
        
        Returns:
            tuple[torch.Tensor, torch.Tensor]: Predicted mean and standard deviation of the action distribution.
        """
        x = x.float()
        x = self.tanh1(self.fc1(x))
        x = self.tanh2(self.fc2(x))

        action_means = self.policy_mean_fc(x)
        action_stddevs = torch.log(1 + torch.exp(self.policy_stddev_fc(x)))
        return action_means, action_stddevs  
    


if __name__ == "__main__":
    env = gym.make("InvertedPendulum-v4", render_mode="human", width=320, height=240)
    env = CustomWrapper(env)
    wrapped_env = gym.wrappers.RecordEpisodeStatistics(env, 50)  # Wrap the environment to record statistics
    
    obs_space_dims = env.observation_space.shape[0]  # Dimension of the observation space
    action_space_dims = env.action_space.shape[0]  # Dimension of the action space
    agent = Agent(obs_space_dims, action_space_dims)  # Instantiate the agent
    policy_network = Policy_Network(obs_space_dims, action_space_dims)
    policy_network.load_state_dict(torch.load('inverted_pendulum_RLmodel_2337.pth'))
    agent.policy_network = policy_network
    
    # Simulation main loop
    num_simulations = 1
    reward_over_episodes = []
    for episode in range(num_simulations):
        obs = wrapped_env.reset()[0]  # Reset the environment at the start of each episode
        done = False
        episode_rewards = []
        while not done:
            action = agent.sample_action(obs)  # Sample an action based on the current observation
            obs, reward, terminated, truncated, info = wrapped_env.step(action)
            agent.rewards.append(reward)
            done = terminated or truncated
            episode_rewards.append(reward)
        # The collection of rewards and log probabilities should happen within the loop.
        total_reward = sum(episode_rewards)
        reward_over_episodes.append(total_reward)
        print("Episode:", episode, "Average Reward:",total_reward)
env.close()
