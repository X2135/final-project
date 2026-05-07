"""
DRL-based DBSCAN Parameter Selection
=====================================
EN（PPO）EN DBSCAN EN。

EN：
1. PPO (Proximal Policy Optimization) EN
2. Actor-Critic EN
3. EN（EN DTW EN D EN）
4. EN（EN）
5. EN

EN：EN Adaptive-DBSCAN，EN。
"""

import os
import csv
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Normal
from typing import Tuple, List, Dict, Optional
from sklearn.cluster import DBSCAN
from sklearn.metrics import silhouette_score
import warnings
warnings.filterwarnings('ignore')


def constrain_distance_matrix(D: np.ndarray) -> np.ndarray:
    """
    DRL EN：EN，EN eps EN。

    - q_low=1%, q_high=95%
    - D'=(D-q_low)/(q_high-q_low)，clip EN [0,1]
    - EN 0
    - EN：EN/NaN/Inf
    """
    if not isinstance(D, np.ndarray):
        D = np.asarray(D)
    if D.ndim != 2 or D.shape[0] != D.shape[1]:
        return np.nan_to_num(D, nan=0.0, posinf=1.0, neginf=0.0).astype(np.float32)

    D_in = D.astype(np.float32, copy=True)

    valid = D_in[(D_in > 0) & np.isfinite(D_in) & (D_in < 1e6)]
    if valid.size == 0:
        out = np.zeros_like(D_in, dtype=np.float32)
        np.fill_diagonal(out, 0.0)
        return out

    vmin = float(np.min(valid))
    vmax = float(np.max(valid))
    if 0.0 <= vmin and vmax <= 1.0:
        out = np.nan_to_num(D_in, nan=0.0, posinf=1.0, neginf=0.0)
        out = np.clip(out, 0.0, 1.0).astype(np.float32)
        np.fill_diagonal(out, 0.0)
        return out

    q_low = float(np.quantile(valid, 0.01))
    q_high = float(np.quantile(valid, 0.95))
    denom = float(q_high - q_low)
    if (not np.isfinite(denom)) or denom <= 1e-12:
        out = np.zeros_like(D_in, dtype=np.float32)
        np.fill_diagonal(out, 0.0)
        return out

    out = (D_in - q_low) / denom
    out = np.nan_to_num(out, nan=0.0, posinf=1.0, neginf=0.0)
    out = np.clip(out, 0.0, 1.0).astype(np.float32)
    np.fill_diagonal(out, 0.0)
    return out



def extract_state_features(D: np.ndarray) -> np.ndarray:
    """
    EN DTW EN D EN **EN、EN、EN** EN DBSCAN EN。

    EN：
    - EN DBSCAN EN“EN eps EN minPts”EN/EN，EN silhouette EN。
    - EN，EN/EN（dummy_silhouette / skew-hopkins / kurtosis-gap）。

    EN（EN 8–10 EN；EN 9 EN）：
    1) kdist_q50        : k-distance EN（EN，EN eps EN）
    2) kdist_q90        : k-distance EN（EN）
    3) kdist_std        : k-distance EN（EN/EN）
    4) neighbor_mean    : eps_ref EN（minPts EN）
    5) neighbor_std     : EN（EN）
    6) neighbor_q10     : EN 10% EN（EN）
    7) neighbor_q90     : EN 90% EN（EN）
    8) density_cv       : std/mean（EN，EN Hopkins/Gap）
    9) ratio_isolated   : EN（EN）
    """
    D = constrain_distance_matrix(D)
    valid = D[(D > 0) & np.isfinite(D) & (D < 1e6)]
    N = int(D.shape[0]) if isinstance(D, np.ndarray) and D.ndim == 2 else 0
    if valid.size == 0 or N <= 1:
        return np.zeros(9, dtype=np.float32)

    k_ref = 5
    k_distances = np.full((N,), np.nan, dtype=np.float64)
    for i in range(N):
        row = D[i, :]
        row_valid = row[(row > 0) & np.isfinite(row) & (row < 1e6)]
        if row_valid.size == 0:
            continue
        row_valid.sort()
        idx = min(k_ref - 1, row_valid.size - 1)
        k_distances[i] = float(row_valid[idx])

    k_valid = k_distances[np.isfinite(k_distances) & (k_distances > 0)]
    if k_valid.size == 0:
        kdist_q50 = float(np.quantile(valid, 0.50))
        kdist_q90 = float(np.quantile(valid, 0.90))
        kdist_std = float(np.std(valid))
    else:
        kdist_q50 = float(np.quantile(k_valid, 0.50))
        kdist_q90 = float(np.quantile(k_valid, 0.90))
        kdist_std = float(np.std(k_valid))

    eps_ref = kdist_q50
    neighbor_counts = np.zeros((N,), dtype=np.float64)
    for i in range(N):
        row = D[i, :]
        mask = (row <= eps_ref) & (row > 0) & np.isfinite(row) & (row < 1e6)
        neighbor_counts[i] = float(np.sum(mask))

    neighbor_mean = float(np.mean(neighbor_counts))
    neighbor_std = float(np.std(neighbor_counts))
    neighbor_q10 = float(np.quantile(neighbor_counts, 0.10))
    neighbor_q90 = float(np.quantile(neighbor_counts, 0.90))

    denom = neighbor_mean if neighbor_mean > 1e-6 else 1e-6
    density_cv = float(neighbor_std / denom)

    ratio_isolated = float(np.mean(neighbor_counts <= 1.0))

    features = np.array(
        [
            kdist_q50,
            kdist_q90,
            kdist_std,
            neighbor_mean,
            neighbor_std,
            neighbor_q10,
            neighbor_q90,
            density_cv,
            ratio_isolated,
        ],
        dtype=np.float32,
    )
    features = np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)

    try:
        key = (N, float(np.quantile(valid, 0.50)), float(np.mean(valid)))
    except Exception:
        key = (N, float(N), float(valid.size))
    printed = getattr(extract_state_features, "_printed_keys", set())
    if key not in printed:
        printed.add(key)
        setattr(extract_state_features, "_printed_keys", printed)

        if k_valid.size > 0:
            kd_min = float(np.min(k_valid))
            kd_q10 = float(np.quantile(k_valid, 0.10))
            kd_q50 = float(np.quantile(k_valid, 0.50))
            kd_q90 = float(np.quantile(k_valid, 0.90))
            kd_max = float(np.max(k_valid))
        else:
            kd_min = kd_q10 = kd_q50 = kd_q90 = kd_max = 0.0

        print("\n" + "=" * 60)
        print("🔎 [State Debug] DBSCAN EN（EN）")
        print("=" * 60)
        print(f"- N={N}, k_ref={k_ref}, eps_ref(kdist_q50)={eps_ref:.6f}")
        print(f"- k-distance: min={kd_min:.6f}, q10={kd_q10:.6f}, q50={kd_q50:.6f}, q90={kd_q90:.6f}, max={kd_max:.6f}, std={kdist_std:.6f}")
        print(f"- neighbor_count: mean={neighbor_mean:.3f}, std={neighbor_std:.3f}, q10={neighbor_q10:.3f}, q90={neighbor_q90:.3f}")
        print(f"- density_cv(std/mean)={density_cv:.6f}")
        print(f"- ratio_isolated(n_i<=1)={ratio_isolated:.6f}")

    return features



class ActorNetwork(nn.Module):
    """
    Actor EN：EN，EN eps EN（latent EN）。

    EN：
    - EN DBSCAN EN（K=0 / EN）EN minPts EN、EN；
    - EN DRL EN“EN eps”，minPts EN D EN（EN PPOAgent._minpts_from_D）。
    """
    def __init__(self, state_dim: int = 9, hidden_dim: int = 64):
        super(ActorNetwork, self).__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, hidden_dim)
        
        self.eps_mean = nn.Linear(hidden_dim, 1)
        self.eps_std = nn.Linear(hidden_dim, 1)
        
        self._initialize_weights()
    
    def _initialize_weights(self):
        """EN"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
                nn.init.constant_(m.bias, 0.0)
    
    def forward(self, state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        EN
        
        EN
        ----
        state : torch.Tensor, shape=(batch_size, state_dim)
            EN
        
        EN
        ----
        eps_mean, eps_std : torch.Tensor
            latent EN eps EN
            - EN R EN（EN sigmoid）
            - EN（softplus EN）
        """
        x = torch.relu(self.fc1(state))
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))

        eps_mean = self.eps_mean(x)
        eps_std = F.softplus(self.eps_std(x)) + 1e-3
        return eps_mean, eps_std


class CriticNetwork(nn.Module):
    """
    Critic EN：EN，EN V(s)。
    """
    def __init__(self, state_dim: int = 9, hidden_dim: int = 64):
        super(CriticNetwork, self).__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, hidden_dim)
        self.value = nn.Linear(hidden_dim, 1)
        
        self._initialize_weights()
    
    def _initialize_weights(self):
        """EN"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
                nn.init.constant_(m.bias, 0.0)
    
    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """
        EN
        
        EN
        ----
        state : torch.Tensor, shape=(batch_size, state_dim)
            EN
        
        EN
        ----
        value : torch.Tensor, shape=(batch_size, 1)
            EN
        """
        x = torch.relu(self.fc1(state))
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))
        value = self.value(x)
        return value



class PPOAgent:
    """
    PPO (Proximal Policy Optimization) EN
    
    EN D EN (eps, minPts) EN。
    """
    def __init__(
        self,
        state_dim: int = 9,
        hidden_dim: int = 64,
        lr_actor: float = 3e-4,
        lr_critic: float = 3e-4,
        gamma: float = 0.99,
        eps_clip: float = 0.2,
        k_epochs: int = 10,
        device: Optional[torch.device] = None,
    ):

        self.device = device if device is not None else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.gamma = gamma
        self.eps_clip = eps_clip
        self.k_epochs = k_epochs
        
        self.actor = ActorNetwork(state_dim, hidden_dim).to(self.device)
        self.critic = CriticNetwork(state_dim, hidden_dim).to(self.device)
        
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=lr_actor)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=lr_critic)
        
        self.reset_buffer()
    
    def reset_buffer(self):
        """EN"""
        self.states: List[np.ndarray] = []
        self.actions: List[Tuple[float, int]] = []
        self.latent_actions: List[float] = []
        self.rewards: List[float] = []
        self.log_probs: List[torch.Tensor] = []
        self.is_terminals: List[bool] = []

    @staticmethod
    def _minpts_from_D(
        D_constrained: np.ndarray,
        eps: float,
        *,
        stat: str = "mean",
        min_minpts: int = 1,
        max_minpts: int | None = 10,
    ) -> int:
        """
        EN eps EN，EN minPts（EN）。

        - D_constrained EN constrain_distance_matrix EN
        - EN：count(row < eps, exclude diagonal)
        - minPts = round(mean/median(neighbor_counts))
        - EN（EN [1,10]），EN minPts EN K=0
        """
        D = np.asarray(D_constrained, dtype=float)
        if D.ndim != 2 or D.shape[0] != D.shape[1]:
            return int(max(1, min_minpts))
        n = int(D.shape[0])
        if n <= 1:
            return int(max(1, min_minpts))
        offdiag = ~np.eye(n, dtype=bool)
        eps_f = float(eps)
        if not np.isfinite(eps_f):
            eps_f = 0.0
        Num = np.count_nonzero((D > 0) & np.isfinite(D) & (D < eps_f) & offdiag, axis=1)
        if Num.size == 0:
            raw = 1.0
        else:
            s = (stat or "mean").strip().lower()
            if s in ("median", "med", "p50"):
                raw = float(np.median(Num))
            else:
                raw = float(np.mean(Num))
        mp = int(round(raw))
        mp = max(int(min_minpts), mp)
        if max_minpts is not None:
            mp = min(int(max_minpts), mp)
        mp = max(1, mp)
        return int(mp)
    
    def select_action(
        self,
        state: np.ndarray,
        D_for_minpts: np.ndarray,
        eps_min: float,
        eps_max: float,
        minpts_stat: str = "mean",
        minpts_min: int = 1,
        minpts_max: int = 10,
        deterministic: bool = False,
    ) -> Tuple[float, int]:

        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        
        with torch.set_grad_enabled(not deterministic):
            eps_mean, eps_std = self.actor(state_tensor)
        
        eps_mean = eps_mean.squeeze(-1)
        eps_std = eps_std.squeeze(-1)

        eps_mean = torch.nan_to_num(eps_mean, nan=0.0, posinf=0.0, neginf=0.0)
        eps_std = torch.nan_to_num(eps_std, nan=0.5, posinf=1.0, neginf=1.0)
        eps_std = eps_std.clamp(min=1e-3, max=10.0)

        eps_dist = Normal(eps_mean, eps_std)

        if deterministic:
            z_eps = eps_mean
        else:
            z_eps = eps_dist.rsample()

        eps_norm = torch.sigmoid(z_eps)

        eps = eps_min + eps_norm.item() * (eps_max - eps_min)
        minpts = self._minpts_from_D(
            np.asarray(D_for_minpts, dtype=float),
            float(eps),
            stat=str(minpts_stat),
            min_minpts=int(minpts_min),
            max_minpts=int(minpts_max),
        )
        
        if not deterministic:
            self.states.append(state)
            self.actions.append((float(eps), int(minpts)))
            self.latent_actions.append(float(z_eps.item()))
            log_prob = eps_dist.log_prob(z_eps)
            self.log_probs.append(log_prob.detach())
        
        return eps, minpts
    
    def compute_reward(
        self,
        D: np.ndarray,
        eps: float,
        minPts: int,
        w1: float = 1.0,
        w2: float = 1.5,
        w3: float = 3.0,
        w4: float = 0.30,
        w5: float = 0.35,
        w6: float = 0.15,
        eps_hi: float = 0.95,
        eps_lo: float = 0.05,
        w8: float = 0.20,
        k_center: float = 4.0,
        k_half_width: float = 2.0,
    ) -> float:
        """
        EN
        
        R = w1 * silhouette
            - w2 * noise_ratio
            - w3 * I(num_clusters < 1)
            + w4 * core_ratio
            - w5 * I(num_clusters == 1)
            - w6 * edge_penalty(eps_pos -> 1)
            - w7 * edge_penalty(eps_pos -> 0)
            + w8 * k_pref(num_clusters)
        
        EN
        ----
        D : np.ndarray
            DTW EN
        eps : float
            DBSCAN EN eps EN
        minPts : int
            DBSCAN EN minPts EN
        w1 : float
            silhouette score EN（EN 1.0）
        w2 : float
            EN（EN 0.5）
        w3 : float
            EN（EN 0.5）
        
        EN
        ----
        reward : float
            EN
        """
        D = constrain_distance_matrix(D)

        valid = D[(D > 0) & np.isfinite(D) & (D < 1e6)]
        eps_min = float(np.quantile(valid, 0.01)) if valid.size else 0.0
        eps_max = float(np.quantile(valid, 0.95)) if valid.size else 1.0
        denom = float(eps_max - eps_min)
        if np.isfinite(denom) and denom > 1e-12:
            eps_pos = float((float(eps) - eps_min) / denom)
        else:
            eps_pos = 0.0
        eps_pos = float(np.clip(eps_pos, 0.0, 1.0))

        db = DBSCAN(eps=eps, min_samples=minPts, metric="precomputed", n_jobs=1)
        labels = db.fit_predict(D)

        N = int(D.shape[0])
        noise_ratio = float(np.mean(labels == -1)) if N > 0 else 1.0
        try:
            core_ratio = float(len(getattr(db, "core_sample_indices_", [])) / max(N, 1))
        except Exception:
            core_ratio = 0.0
        unique = set(int(x) for x in labels.tolist()) if N > 0 else set()
        num_clusters = int(len(unique) - (1 if -1 in unique else 0))
        
        if num_clusters >= 2 and N > num_clusters:
            try:
                mask = labels != -1
                if np.sum(mask) >= 3 and len(set(labels[mask].tolist())) >= 2:
                    silhouette = float(silhouette_score(D[np.ix_(mask, mask)], labels[mask], metric="precomputed"))
                else:
                    silhouette = 0.0
            except:
                silhouette = 0.0
        else:
            silhouette = 0.0
        

        edge_pen_mid = 0.0
        if np.isfinite(eps_pos):
            center = 0.5 * (float(eps_lo) + float(eps_hi))
            half_width = max(1e-12, 0.5 * (float(eps_hi) - float(eps_lo)))
            d = abs(float(eps_pos) - center) / half_width
            if d < 1.0:
                mid_pref = 1.0 - d * d
            else:
                mid_pref = 0.0
            edge_pen_mid = 1.0 - mid_pref

        reward = (w1 * silhouette) - (w2 * noise_ratio) + (w4 * core_ratio) - (w6 * edge_pen_mid)

        if num_clusters < 1:
            reward -= w3

        if noise_ratio > 0.80:
            reward -= 2.0 * float(noise_ratio - 0.80) / 0.20
        if noise_ratio > 0.95:
            reward -= 2.0

        if num_clusters == 1:
            reward -= w5

        k_pref = 0.0
        if num_clusters >= 2:
            d = abs(float(num_clusters) - float(k_center)) / max(1e-12, float(k_half_width))
            if d < 1.0:
                k_pref = 1.0 - (d * d)
            else:
                k_pref = 0.0

        reward += float(w8) * float(k_pref)

        self.last_metrics = {
            "silhouette": float(silhouette),
            "noise_ratio": float(noise_ratio),
            "num_clusters": int(num_clusters),
            "core_ratio": float(core_ratio),
        }
        
        return float(reward)
    
    def update(self):
        """
        EN PPO EN Actor EN Critic EN
        """
        if len(self.states) == 0:
            return
        
        states = torch.FloatTensor(np.array(self.states)).to(self.device)
        z_eps_actions = torch.FloatTensor(self.latent_actions).to(self.device)
        old_log_probs = torch.stack(self.log_probs).detach().to(self.device)
        
        rewards = []
        discounted_reward = 0
        for reward, is_terminal in zip(reversed(self.rewards), reversed(self.is_terminals)):
            if is_terminal:
                discounted_reward = 0
            discounted_reward = reward + (self.gamma * discounted_reward)
            rewards.insert(0, discounted_reward)
        
        rewards = torch.FloatTensor(rewards).to(self.device)
        rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-8)
        rewards = torch.nan_to_num(rewards, nan=0.0, posinf=0.0, neginf=0.0)
        
        for _ in range(self.k_epochs):
            eps_mean, eps_std = self.actor(states)

            eps_mean = torch.nan_to_num(eps_mean.squeeze(-1), nan=0.0, posinf=0.0, neginf=0.0)
            eps_std = torch.nan_to_num(eps_std.squeeze(-1), nan=0.5, posinf=1.0, neginf=1.0)
            eps_std = eps_std.clamp(min=1e-3, max=10.0)

            eps_dist = Normal(eps_mean, eps_std)

            new_log_probs = eps_dist.log_prob(z_eps_actions)
            new_log_probs = new_log_probs.squeeze()
            
            values = self.critic(states).squeeze()
            advantages = rewards - values.detach()
            
            ratio = torch.exp(new_log_probs - old_log_probs)
            surr1 = ratio * advantages
            surr2 = torch.clamp(ratio, 1 - self.eps_clip, 1 + self.eps_clip) * advantages
            actor_loss = -torch.min(surr1, surr2).mean()
            
            critic_loss = nn.MSELoss()(values, rewards)
            
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 0.5)
            self.actor_optimizer.step()
            
            self.critic_optimizer.zero_grad()
            critic_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 0.5)
            self.critic_optimizer.step()
        
        self.reset_buffer()
    
    def save(self, filepath: str):
        """EN"""
        torch.save({
            'actor_state_dict': self.actor.state_dict(),
            'critic_state_dict': self.critic.state_dict(),
            'actor_optimizer_state_dict': self.actor_optimizer.state_dict(),
            'critic_optimizer_state_dict': self.critic_optimizer.state_dict(),
        }, filepath)
    
    def load(self, filepath: str):
        """EN"""
        setattr(self, "_loaded_ok", False)
        checkpoint = torch.load(filepath, map_location=self.device)
        try:
            self.actor.load_state_dict(checkpoint["actor_state_dict"])
            self.critic.load_state_dict(checkpoint["critic_state_dict"])
            self.actor_optimizer.load_state_dict(checkpoint["actor_optimizer_state_dict"])
            self.critic_optimizer.load_state_dict(checkpoint["critic_optimizer_state_dict"])
            setattr(self, "_loaded_ok", True)
        except Exception as e:
            print(f"⚠️ EN，EN：{filepath}")
            print(f"   EN：EN。EN：{e}")
            return



def train_drl_dbscan(
    D: np.ndarray,
    num_episodes: int = 100,
    max_steps_per_episode: int = 1,
    model_save_path: Optional[str] = None,
    model_load_path: Optional[str] = None,
    log_path: Optional[str] = None,
    minpts_stat: str = "mean",
    verbose: bool = True,
) -> PPOAgent:
    """
    EN DRL EN (eps, minPts) EN
    
    EN
    ----
    D : np.ndarray, shape=(N, N)
        DTW EN
    num_episodes : int
        EN（EN 100）
    max_steps_per_episode : int
        EN（EN 1，EN）
    model_save_path : str, optional
        EN（EN，EN）
    model_load_path : str, optional
        EN（EN，EN）
        EN，EN
    verbose : bool
        EN（EN True）
    
    EN
    ----
    agent : PPOAgent
        EN
    """
    D = constrain_distance_matrix(D)

    agent = PPOAgent()
    
    if model_load_path and os.path.exists(model_load_path):
        agent.load(model_load_path)
        if verbose:
            if getattr(agent, "_loaded_ok", False):
                print(f"📦 EN: {model_load_path}")
                print(f"   EN {num_episodes} EN")
            else:
                print(f"⚠️ EN，EN: {model_load_path}")
    elif model_load_path:
        if verbose:
            print(f"⚠️ EN: {model_load_path}，EN")
    elif verbose:
        print("🆕 EN")
    
    state = extract_state_features(D)
    
    valid = D[(D > 0) & np.isfinite(D) & (D < 1e6)]
    N = int(D.shape[0]) if isinstance(D, np.ndarray) and D.ndim == 2 else 0
    eps_min = float(np.quantile(valid, 0.01)) if valid.size else 0.0
    eps_max = float(np.quantile(valid, 0.95)) if valid.size else 1.0
    minpts_min = 1
    minpts_max = max(4, min(50, max(1, N // 2)))
    
    episode_rewards = []
    episode_logs: List[Dict[str, float]] = []
    for episode in range(num_episodes):
        episode_reward = 0.0
        last_eps = None
        last_eps_pos = None
        last_minpts = None
        last_metrics: Dict[str, float] = {}
        
        for step in range(max_steps_per_episode):
            eps, minPts = agent.select_action(
                state,
                D_for_minpts=D,
                eps_min=eps_min,
                eps_max=eps_max,
                minpts_stat=str(minpts_stat),
                minpts_min=minpts_min,
                minpts_max=minpts_max,
                deterministic=False,
            )
            last_eps = float(eps)
            denom = float(eps_max - eps_min)
            if np.isfinite(denom) and denom > 1e-12:
                last_eps_pos = float((last_eps - float(eps_min)) / denom)
            else:
                last_eps_pos = 0.0
            last_eps_pos = float(np.clip(last_eps_pos, 0.0, 1.0))
            last_minpts = int(minPts)
            
            reward = agent.compute_reward(D, eps, minPts)
            episode_reward += reward
            
            agent.rewards.append(reward)
            agent.is_terminals.append(True)

            if hasattr(agent, "last_metrics") and isinstance(agent.last_metrics, dict):
                last_metrics = {
                    "silhouette": float(agent.last_metrics.get("silhouette", 0.0)),
                    "noise_ratio": float(agent.last_metrics.get("noise_ratio", 0.0)),
                    "num_clusters": int(agent.last_metrics.get("num_clusters", 0)),
                    "core_ratio": float(agent.last_metrics.get("core_ratio", 0.0)),
                }
        
        episode_rewards.append(episode_reward)
        if verbose:
            print(
                "[Check A/B][Train] "
                f"ep={episode + 1}/{num_episodes} "
                f"eps={float(last_eps) if last_eps is not None else float('nan'):.4f} "
                f"pos={(float(last_eps_pos) if last_eps_pos is not None else float('nan')):.2%} "
                f"(eps_min={eps_min:.4f}, eps_max={eps_max:.4f}) "
                f"minPts={int(last_minpts) if last_minpts is not None else -1} "
                f"reward={float(episode_reward):.4f} "
                f"sil={float(last_metrics.get('silhouette', 0.0)):.4f} "
                f"noise={float(last_metrics.get('noise_ratio', 0.0)):.2%} "
                f"core={float(last_metrics.get('core_ratio', 0.0)):.2%} "
                f"K={int(last_metrics.get('num_clusters', 0))}"
            )
        episode_logs.append(
            {
                "episode": int(episode + 1),
                "eps": float(last_eps) if last_eps is not None else float("nan"),
                "eps_min": float(eps_min),
                "eps_max": float(eps_max),
                "eps_pos": float(last_eps_pos) if last_eps_pos is not None else float("nan"),
                "minPts": int(last_minpts) if last_minpts is not None else -1,
                "reward": float(episode_reward),
                "silhouette": float(last_metrics.get("silhouette", 0.0)),
                "noise_ratio": float(last_metrics.get("noise_ratio", 0.0)),
                "core_ratio": float(last_metrics.get("core_ratio", 0.0)),
                "num_clusters": int(last_metrics.get("num_clusters", 0)),
            }
        )
        
        agent.update()
        
        if verbose and (episode + 1) % 10 == 0:
            avg_reward = np.mean(episode_rewards[-10:])
            print(f"Episode {episode + 1}/{num_episodes}, Avg Reward: {avg_reward:.4f}")
    
    if model_save_path:
        agent.save(model_save_path)
        if verbose:
            print(f"✅ EN: {model_save_path}")

    if log_path:
        try:
            os.makedirs(os.path.dirname(log_path), exist_ok=True)
        except Exception:
            pass
        try:
            with open(log_path, "w", newline="", encoding="utf-8") as f:
                fieldnames = [
                    "episode",
                    "eps",
                    "eps_min",
                    "eps_max",
                    "eps_pos",
                    "minPts",
                    "reward",
                    "silhouette",
                    "noise_ratio",
                    "core_ratio",
                    "num_clusters",
                ]
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                for row in episode_logs:
                    writer.writerow(row)
            if verbose:
                print(f"✅ DRL EN: {log_path}")
        except Exception as e:
            if verbose:
                print(f"⚠️ EN DRL EN {log_path}: {e}")
    
    return agent


def train_drl_dbscan_multi(
    Ds: List[np.ndarray],
    num_episodes: int = 300,
    max_steps_per_episode: int = 1,
    model_save_path: Optional[str] = None,
    model_load_path: Optional[str] = None,
    log_path: Optional[str] = None,
    minpts_stat: str = "mean",
    verbose: bool = True,
    seed: int = 0,
) -> PPOAgent:
    """
    EN **EN** EN DRL（EN“EN”EN）。

    EN：
    - EN episode EN D_k
    - EN D_k EN constrain_distance_matrix，EN eps EN
    - EN extract_state_features(D_k) EN D EN，EN MMSI

    EN：
    - EN contextual bandit EN（EN episode EN eps/minPts），EN D EN。
    """
    if not Ds:
        raise ValueError("Ds is empty")
    rng = np.random.default_rng(int(seed))

    agent = PPOAgent()

    if model_load_path and os.path.exists(model_load_path):
        agent.load(model_load_path)
        if verbose:
            if getattr(agent, "_loaded_ok", False):
                print(f"📦 EN: {model_load_path}")
                print(f"   EN {num_episodes} EN（multi-dataset）")
            else:
                print(f"⚠️ EN，EN: {model_load_path}")
    elif model_load_path and verbose:
        print(f"⚠️ EN: {model_load_path}，EN")
    elif verbose:
        print("🆕 EN（multi-dataset）")

    episode_logs: List[Dict[str, float]] = []
    episode_rewards: List[float] = []

    for episode in range(int(num_episodes)):
        k = int(rng.integers(0, len(Ds)))
        D = constrain_distance_matrix(Ds[k])
        state = extract_state_features(D)
    
        valid = D[(D > 0) & np.isfinite(D) & (D < 1e6)]
        N = int(D.shape[0]) if isinstance(D, np.ndarray) and D.ndim == 2 else 0
        eps_min = float(np.quantile(valid, 0.01)) if valid.size else 0.0
        eps_max = float(np.quantile(valid, 0.95)) if valid.size else 1.0
        minpts_min = 1
        minpts_max = min(10, max(4, int(N // 2)))

        episode_reward = 0.0
        last_eps = float("nan")
        last_eps_pos = float("nan")
        last_minpts = -1
        last_metrics: Dict[str, float] = {}

        for _ in range(int(max_steps_per_episode)):
            eps, minPts = agent.select_action(
                state,
                D_for_minpts=D,
                eps_min=eps_min,
                eps_max=eps_max,
                minpts_stat=str(minpts_stat),
                minpts_min=minpts_min,
                minpts_max=minpts_max,
                deterministic=False,
            )
            last_eps = float(eps)
            denom = float(eps_max - eps_min)
            if np.isfinite(denom) and denom > 1e-12:
                last_eps_pos = float((last_eps - float(eps_min)) / denom)
            else:
                last_eps_pos = 0.0
            last_eps_pos = float(np.clip(last_eps_pos, 0.0, 1.0))
            last_minpts = int(minPts)

            reward = agent.compute_reward(D, eps, minPts)
            episode_reward += float(reward)
            agent.rewards.append(float(reward))
            agent.is_terminals.append(True)

            if hasattr(agent, "last_metrics") and isinstance(agent.last_metrics, dict):
                last_metrics = {
                    "silhouette": float(agent.last_metrics.get("silhouette", 0.0)),
                    "noise_ratio": float(agent.last_metrics.get("noise_ratio", 0.0)),
                    "num_clusters": int(agent.last_metrics.get("num_clusters", 0)),
                    "core_ratio": float(agent.last_metrics.get("core_ratio", 0.0)),
                }

        episode_rewards.append(float(episode_reward))

        if verbose:
            print(
                "[Train-Multi] "
                f"ep={episode+1}/{num_episodes} ds={k}/{len(Ds)} "
                f"eps={last_eps:.4f} pos={last_eps_pos:.2%} "
                f"(eps_min={eps_min:.4f}, eps_max={eps_max:.4f}) "
                f"minPts={last_minpts} reward={episode_reward:.4f} "
                f"sil={float(last_metrics.get('silhouette', 0.0)):.4f} "
                f"noise={float(last_metrics.get('noise_ratio', 0.0)):.2%} "
                f"core={float(last_metrics.get('core_ratio', 0.0)):.2%} "
                f"K={int(last_metrics.get('num_clusters', 0))}"
            )

        episode_logs.append(
            {
                "episode": int(episode + 1),
                "dataset_idx": int(k),
                "eps": float(last_eps),
                "eps_min": float(eps_min),
                "eps_max": float(eps_max),
                "eps_pos": float(last_eps_pos),
                "minPts": int(last_minpts),
                "reward": float(episode_reward),
                "silhouette": float(last_metrics.get("silhouette", 0.0)),
                "noise_ratio": float(last_metrics.get("noise_ratio", 0.0)),
                "core_ratio": float(last_metrics.get("core_ratio", 0.0)),
                "num_clusters": int(last_metrics.get("num_clusters", 0)),
            }
        )

        agent.update()

        if verbose and (episode + 1) % 25 == 0:
            avg_reward = float(np.mean(episode_rewards[-25:]))
            print(f"[Train-Multi] Episode {episode+1}/{num_episodes}, AvgReward(last25)={avg_reward:.4f}")

    if model_save_path:
        agent.save(model_save_path)
        if verbose:
            print(f"✅ EN: {model_save_path}")

    if log_path:
        try:
            os.makedirs(os.path.dirname(log_path), exist_ok=True)
        except Exception:
            pass
        try:
            with open(log_path, "w", newline="", encoding="utf-8") as f:
                fieldnames = [
                    "episode",
                    "dataset_idx",
                    "eps",
                    "eps_min",
                    "eps_max",
                    "eps_pos",
                    "minPts",
                    "reward",
                    "silhouette",
                    "noise_ratio",
                    "core_ratio",
                    "num_clusters",
                ]
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                for row in episode_logs:
                    writer.writerow(row)
            if verbose:
                print(f"✅ DRL EN: {log_path}")
        except Exception as e:
            if verbose:
                print(f"⚠️ EN DRL EN {log_path}: {e}")

    return agent



def drl_select_eps_minpts(
    D: np.ndarray,
    model_path: Optional[str] = None,
    agent: Optional[PPOAgent] = None,
    minpts_stat: str = "mean",
    verbose: bool = False,
) -> Tuple[float, int]:
    """
    EN DRL EN (eps, minPts)
    
    EN
    ----
    D : np.ndarray, shape=(N, N)
        DTW EN
    model_path : str, optional
        EN（EN，EN）
    agent : PPOAgent, optional
        EN（EN，EN）
    
    EN
    ----
    eps : float
        EN eps EN
    minPts : int
        EN minPts EN
    """
    D = constrain_distance_matrix(D)

    if agent is None:
        agent = PPOAgent()
        if model_path and os.path.exists(model_path):
            agent.load(model_path)
            if verbose and (not getattr(agent, "_loaded_ok", False)):
                import warnings
                warnings.warn("EN（EN），EN。EN。")
        else:
            if verbose:
                import warnings
                warnings.warn("EN，EN。EN。")
    
    state = extract_state_features(D)
    
    valid = D[(D > 0) & np.isfinite(D) & (D < 1e6)]
    if valid.size == 0:
        return 1.0, 4
    
    eps_min = float(np.quantile(valid, 0.01))
    eps_max = float(np.quantile(valid, 0.95))
    
    with torch.no_grad():
        eps, minPts = agent.select_action(
            state,
            D_for_minpts=D,
            eps_min=eps_min,
            eps_max=eps_max,
            minpts_stat=str(minpts_stat),
            minpts_min=1,
            minpts_max=min(10, max(1, int(D.shape[0]) // 2)),
            deterministic=True,
        )

    denom = float(eps_max - eps_min)
    if np.isfinite(denom) and denom > 1e-12:
        eps_pos = float((float(eps) - float(eps_min)) / denom)
    else:
        eps_pos = 0.0
    eps_pos = float(np.clip(eps_pos, 0.0, 1.0))
    if verbose:
        near = ""
        if eps_pos >= 0.95:
            near = " (⚠️ near eps_max)"
        elif eps_pos <= 0.05:
            near = " (⚠️ near eps_min)"
        print(
            "[Check A][Infer] "
            f"eps={float(eps):.4f}, minPts={int(minPts)} "
            f"eps_min={float(eps_min):.4f}, eps_max={float(eps_max):.4f}, pos={eps_pos:.2%}"
            f"{near}"
        )

    return eps, minPts



if __name__ == "__main__":
    np.random.seed(42)
    N = 50
    D = np.random.rand(N, N) * 5.0
    D = (D + D.T) / 2
    np.fill_diagonal(D, 0)
    
    print("=" * 60)
    print("DRL DBSCAN EN")
    print("=" * 60)
    
    print("\n1. EN...")
    state = extract_state_features(D)
    print(f"   EN: {state.shape}")
    print(f"   EN: {state}")
    
    print("\n2. EN（EN）...")
    eps, minPts = drl_select_eps_minpts(D)
    print(f"   EN: eps={eps:.4f}, minPts={minPts}")
    
    print("\n3. EN（5 EN）...")
    agent = train_drl_dbscan(D, num_episodes=5, verbose=True)
    
    print("\n4. EN...")
    eps, minPts = drl_select_eps_minpts(D, agent=agent)
    print(f"   EN: eps={eps:.4f}, minPts={minPts}")
    
    print("\n✅ EN！")

