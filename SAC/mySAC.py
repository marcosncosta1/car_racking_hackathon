import torch
from torch import nn
import numpy as np
from tqdm import tqdm
from copy import deepcopy
import time


def make_CNN(hidden_dim=32):
    return nn.Sequential(
            nn.Conv2d(6,hidden_dim,4,2),
            nn.ReLU(),
            nn.Conv2d(hidden_dim,hidden_dim,4,2),
            nn.ReLU(),
            nn.Conv2d(hidden_dim,hidden_dim,4,2),
            nn.ReLU(),
            nn.Conv2d(hidden_dim,32,4,2), # (1,6,96,96) -> (1,...)
            nn.ReLU(),
        )

def make_MLP(in_dim, out_dim, hidden_dim=32):
    return nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, out_dim),
        )

class Actor(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.CNN = make_CNN()
        self.MLP = make_MLP(512, 6)
    
    def forward(self, observation):
        features = self.CNN(observation).flatten(1)
        out = self.MLP(features)
        mean, log_std = out.chunk(2, dim=-1)
        log_std = log_std.clamp(-20, 2)
        return mean, log_std

    def sample_action(self, mean, log_std):
        """Training: sample + squash, return action and log_prob"""
        std = log_std.exp()
        dist = torch.distributions.Normal(mean, std)
        x = dist.rsample()                          # reparameterized sample
        action = torch.tanh(x)                       # squash to [-1, 1]
        # log_prob with tanh correction
        log_prob = dist.log_prob(x) - torch.log(1 - action.pow(2) + 1e-6)
        log_prob = log_prob.sum(dim=-1)              # (B,)
        action = torch.cat([action[:, :1], (action[:, 1:] + 1) / 2], dim=1)
        return action, log_prob

    def eval_action(self, mean, log_std):
        """Eval: just use the mean, squashed"""
        action = torch.tanh(mean)
        action = torch.cat([action[:, :1], (action[:, 1:] + 1) / 2], dim=1)
        return action


class QNetwork(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.CNN = make_CNN()
        self.MLP = make_MLP(512+3, 1)
    
    def forward(self, observation, action):
        features = self.CNN(observation).flatten(1)
        combined = torch.cat([features, action], dim=-1)
        value = self.MLP(combined)
        return value


class CarV3ReplayBuffer:
    def __init__(self, device, size=100_000):
        self.size = size
        self.device = device
        self.frames = np.zeros((size + 2, 96, 96, 3), dtype=np.uint8)
        self.actions = np.zeros((size, 3), dtype=np.float32)
        self.rewards = np.zeros(size, dtype=np.float32)
        self.position = 0

    def add_episode(self, frames_np, actions_np, rewards_np):
        N = len(frames_np)
        if N <= 3: return
        n = N - 2
        p = self.position
        self.frames[p:p+n+2] = frames_np  # store raw uint8
        self.actions[p:p+n] = actions_np[1:N-1]
        self.rewards[p:p+n] = rewards_np[1:N-1]
        self.position += n

    def sample_batches(self, batch_size, n_batches, device):
        """Sample n_batches at once, return list of tuples"""
        total = batch_size * n_batches
        idx = np.random.randint(0, self.position, size=total)

        prev = torch.from_numpy(self.frames[idx]).float() / 255.0
        curr = torch.from_numpy(self.frames[idx + 1]).float() / 255.0
        nxt = torch.from_numpy(self.frames[idx + 2]).float() / 255.0

        obs = torch.cat([prev, curr], dim=-1).permute(0, 3, 1, 2).to(device)
        next_obs = torch.cat([curr, nxt], dim=-1).permute(0, 3, 1, 2).to(device)
        actions = torch.from_numpy(self.actions[idx]).to(device)
        rewards = torch.from_numpy(self.rewards[idx]).to(device)

        return (
            obs.split(batch_size),
            next_obs.split(batch_size),
            actions.split(batch_size),
            rewards.split(batch_size),
        )
            


def stack_obs(last_frame_np, current_frame_np, dtype, device):
    prev = torch.from_numpy(last_frame_np).float() / 255.0
    curr = torch.from_numpy(current_frame_np).float() / 255.0
    return torch.cat([prev, curr], dim=-1).permute(2, 0, 1).unsqueeze(0).to(dtype=dtype, device=device)


class mySAC(nn.Module):
    def __init__(self, env, device, dtype, gamma=0.99, lr=3e-4, tau=0.005, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.env = env
        self.device = device
        self.dtype = dtype
        self.gamma = gamma
        self.tau = tau

        self.Actor = Actor()
        self.Q1 = QNetwork()
        self.Q2 = QNetwork()
        self.Actor.to(device)
        self.Q1.to(device)
        self.Q2.to(device)
        self.Q1_target = deepcopy(self.Q1)
        self.Q2_target = deepcopy(self.Q2)

        self.ReplayBuffer = CarV3ReplayBuffer(self.device)

        for param in self.Q1_target.parameters(): param.requires_grad = False
        for param in self.Q2_target.parameters(): param.requires_grad = False

        self.log_alpha = nn.Parameter(torch.zeros(1, device=device))
        self.target_entropy = -3.0  # -action_dim

        self.actor_optim = torch.optim.Adam(self.Actor.parameters(), lr=lr)
        self.q_optim = torch.optim.Adam(list(self.Q1.parameters()) + list(self.Q2.parameters()), lr=lr)
        self.alpha_optim = torch.optim.Adam([self.log_alpha], lr=lr)
    
    @torch.no_grad()
    def update_targets(self):
        for p, pt in zip(self.Q1.parameters(), self.Q1_target.parameters()):
            pt.data.lerp_(p.data, self.tau)
        for p, pt in zip(self.Q2.parameters(), self.Q2_target.parameters()):
            pt.data.lerp_(p.data, self.tau)
    
    @torch.no_grad()
    def collect_episode(self):

        observation, info = self.env.reset()
        
        frames = [observation, observation]  # duplicate first frame as "previous"
        actions = [np.zeros(3)]             # dummy action for frame 0
        rewards = [0.0]                     # dummy reward for frame 0
        
        done = False
        last_frame_np = observation
        current_frame_np = observation

        while not done:
            obs = stack_obs(last_frame_np, current_frame_np, self.dtype, self.device)
            mean, log_std = self.Actor(obs)
            action, _ = self.Actor.sample_action(mean, log_std)
            action_np = action.squeeze(0).cpu().numpy()
            
            observation, reward, terminated, truncated, info = self.env.step(action_np)
            done = terminated or truncated
            
            frames.append(observation)
            actions.append(action_np)
            rewards.append(reward)
            
            last_frame_np = current_frame_np
            current_frame_np = observation

        return np.stack(frames), np.stack(actions), np.array(rewards)
    
    def learn(self, total_timesteps=10_000):
        pbar = tqdm(total=total_timesteps)
        steps_taken = 0
        while steps_taken < total_timesteps:
            frames, actions, rewards = self.collect_episode()
            ep_len = len(rewards) - 1
            ep_reward = sum(rewards)
            self.ReplayBuffer.add_episode(frames, actions, rewards)

            n_updates = max(1, ep_len//5) #  // 10
            n_max = 50
            for _ in range(n_updates // n_max):
                all_obs, all_next, all_act, all_rew = self.ReplayBuffer.sample_batches(128, n_max, self.device)
                for i in range(n_max):  # n_max, not n_updates
                    self.update(all_obs[i], all_next[i], all_act[i], all_rew[i])

            steps_taken += ep_len
            pbar.update(ep_len)
            pbar.set_description(f"ep_reward={ep_reward:.1f} alpha={self.log_alpha.exp().item():.3f}")
        pbar.close()

    def update(self, states, next_states, actions, rewards):
        self.actor_optim.zero_grad()
        self.q_optim.zero_grad()

        alpha = self.log_alpha.exp()

        next_mean, next_log_std = self.Actor(next_states)
        next_action, next_log_prob = self.Actor.sample_action(next_mean, next_log_std)
        Q12min = torch.min(self.Q1_target(next_states, next_action), self.Q2_target(next_states, next_action))
        Q_target = (rewards.unsqueeze(-1) + self.gamma * (Q12min - alpha * next_log_prob.unsqueeze(-1))).detach()

        Loss_Q1 = nn.functional.mse_loss(self.Q1(states, actions), Q_target)
        Loss_Q2 = nn.functional.mse_loss(self.Q2(states, actions), Q_target)

        # Step 1: update Q
        self.q_optim.zero_grad()
        (Loss_Q1 + Loss_Q2).backward()
        self.q_optim.step()

        new_mean, new_log_std = self.Actor(states)
        new_action, new_log_prob = self.Actor.sample_action(new_mean, new_log_std)
        Q12min_new = torch.min(self.Q1(states, new_action), self.Q2(states, new_action))
        Loss_Actor = torch.mean(alpha * new_log_prob - Q12min_new)

        Loss_alpha = -(self.log_alpha * (new_log_prob.detach() + self.target_entropy)).mean()
        
        # Step 2: update actor
        self.actor_optim.zero_grad()
        Loss_Actor.backward()
        self.actor_optim.step()

        # Step 3: update alpha
        self.alpha_optim.zero_grad()
        Loss_alpha.backward()
        self.alpha_optim.step()

        # Step 4: polyak
        self.update_targets()



if __name__ == "__main__":
    import gymnasium as gym
    env = gym.make("CarRacing-v3", render_mode="rgb_array", lap_complete_percent=0.95, domain_randomize=False, continuous=True)
    eval_env = gym.make("CarRacing-v3", render_mode="human", lap_complete_percent=0.95, domain_randomize=False, continuous=True)

    model = mySAC(env, "cuda", torch.float32)
    for _ in range(20):
        model.learn(total_timesteps=10_000)
        model.env = eval_env
        model.learn(total_timesteps=500)
        model.env = env