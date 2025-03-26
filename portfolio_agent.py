import os
from time import sleep
from collections import deque, namedtuple

import numpy as np
import torch
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
import torch.nn.functional as F
from tensorboardX import SummaryWriter
import matplotlib.pyplot as plt
import seaborn as sns
from IPython import display

# Import dei moduli locali
from memory import Memory, PrioritizedMemory, Node
from portfolio_models import PortfolioCritic, EnhancedPortfolioActor, AssetEncoder

# Definizione di un namedtuple per le transizioni
Transition = namedtuple("Transition", ("state", "action", "reward", "next_state", "dones"))

# Parametri globali
GAMMA = 0.99                 
TAU_ACTOR = 1e-1             
TAU_CRITIC = 1e-3            
LR_ACTOR = 1e-3              
LR_CRITIC = 1e-4             
WEIGHT_DECAY_ACTOR = 0       
WEIGHT_DECAY_CRITIC = 1e-2   
BATCH_SIZE = 512             
BUFFER_SIZE = int(1e6)       
PRETRAIN = 256               
MAX_STEP = 100               
WEIGHTS = "portfolio_weights/"  

# Dimensioni dei layer nelle reti
FC1_UNITS_ACTOR = 128        
FC2_UNITS_ACTOR = 64       
FC3_UNITS_ACTOR = 32       

FC1_UNITS_CRITIC = 512       
FC2_UNITS_CRITIC = 256       
FC3_UNITS_CRITIC = 128       

DECAY_RATE = 1e-6            
EXPLORE_STOP = 0.1           

# Modify in portfolio_agent.py

class MultiAssetOUNoise:
    """
    Ornstein-Uhlenbeck process for generating temporally correlated exploration noise
    for multiple assets simultaneously.
    """
    def __init__(self, action_size, mu=0.0, theta=0.1, sigma=0.2):
        self.action_size = action_size
        self.mu = mu * np.ones(action_size)
        self.theta = theta
        self.sigma = sigma
        self.reset()

    def reset(self):
        self.state = np.copy(self.mu)

    def sample(self, truncate=False, max_pos=2.0, positions=None, actions=None):
        x = self.state
        if truncate:
            assert positions is not None, "positions required when truncate=True"
            assert actions is not None, "actions required when truncate=True"
            from scipy.stats import truncnorm
            noise = np.zeros(self.action_size)
            for i in range(self.action_size):
                m = -max_pos - positions[i] - actions[i] - (1 - self.theta) * x[i]
                M = max_pos - positions[i] - actions[i] - (1 - self.theta) * x[i]
                x_a, x_b = m / self.sigma, M / self.sigma
                X = truncnorm(x_a, x_b, scale=self.sigma)
                dx = self.theta * (self.mu[i] - x[i]) + X.rvs()
                self.state[i] = x[i] + dx
                noise[i] = self.state[i]
        else:
            dx = self.theta * (self.mu - x) + self.sigma * np.random.standard_normal(size=self.action_size)
            self.state = x + dx
            noise = self.state
        return noise

class AdaptiveExploration(MultiAssetOUNoise):
    def __init__(self, action_size, mu=0.0, theta=0.1, sigma=0.2, min_sigma=0.05):
        super().__init__(action_size, mu, theta, sigma)
        self.min_sigma = min_sigma
        self.original_sigma = sigma
        
    def adapt_sigma(self, performance_metric, target_metric=0.2):
        """Adapt exploration noise based on performance."""
        # Increase exploration when underperforming
        ratio = target_metric / (performance_metric + 1e-8)
        self.sigma = max(self.min_sigma, self.original_sigma * min(ratio, 3.0))
        return self.sigma


class PortfolioAgent:
    def __init__(
        self,
        num_assets,
        gamma=GAMMA,
        max_size=BUFFER_SIZE,
        max_step=MAX_STEP,
        memory_type="prioritized",
        alpha=0.6,
        beta0=0.4,
        epsilon=1e-8,
        sliding="oldest",
        batch_size=BATCH_SIZE,
        theta=0.1,
        sigma=0.2,
        use_enhanced_actor=False,
        use_batch_norm=True
    ):
        assert 0 <= gamma <= 1, "Gamma must be in [0,1]"
        assert memory_type in ["uniform", "prioritized"], "Invalid memory type"
        
        self.num_assets = num_assets
        self.gamma = gamma
        self.max_size = max_size
        self.memory_type = memory_type
        self.epsilon = epsilon
        self.use_enhanced_actor = use_enhanced_actor
        self.use_batch_norm = use_batch_norm

        # Device management: utilizza GPU se disponibile
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        if memory_type == "uniform":
            self.memory = Memory(max_size=max_size)
        elif memory_type == "prioritized":
            self.memory = PrioritizedMemory(max_size=max_size, sliding=sliding)

        self.max_step = max_step
        self.alpha = alpha
        self.beta0 = beta0
        self.batch_size = batch_size
        
        # Inizializza il processo OU multi-dimensionale per l'esplorazione
        self.noise = MultiAssetOUNoise(num_assets, theta=theta, sigma=sigma)

        # I modelli verranno creati in fase di training
        self.actor_local = None
        self.actor_target = None
        self.critic_local = None
        self.critic_target = None

    def reset(self):
        self.noise.reset()

    def save_checkpoint(self, file_path, episode, iteration, metrics=None):
        print(f"Preparazione checkpoint per episodio {episode}...")
        checkpoint_metrics = {}
        if metrics:
            for key, value in metrics.items():
                if isinstance(value, deque):
                    checkpoint_metrics[key] = list(value)
                else:
                    checkpoint_metrics[key] = value
        checkpoint = {
            'episode': episode,
            'iteration': iteration,
            'actor_state_dict': self.actor_local.state_dict(),
            'critic_state_dict': self.critic_local.state_dict(),
            'actor_target_state_dict': self.actor_target.state_dict(),
            'critic_target_state_dict': self.critic_target.state_dict(),
            'metrics': checkpoint_metrics
        }
        torch.save(checkpoint, file_path)
        print(f"Checkpoint salvato: {file_path} (episodio {episode})")

    def load_checkpoint(self, path):
        checkpoint = torch.load(path, map_location=self.device)
        if self.actor_local:
            self.actor_local.load_state_dict(checkpoint['actor_state_dict'])
        if self.critic_local:
            self.critic_local.load_state_dict(checkpoint['critic_state_dict'])
        if self.actor_target and checkpoint['actor_target_state_dict']:
            self.actor_target.load_state_dict(checkpoint['actor_target_state_dict'])
        if self.critic_target and checkpoint['critic_target_state_dict']:
            self.critic_target.load_state_dict(checkpoint['critic_target_state_dict'])
        return checkpoint['episode'], checkpoint['metrics']

    def step(self, state, action, reward, next_state, done, pretrain=False):
        state_mb = torch.tensor([state], dtype=torch.float).to(self.device)
        action_mb = torch.tensor([action], dtype=torch.float).to(self.device)
        reward_mb = torch.tensor([[reward]], dtype=torch.float).to(self.device)
        next_state_mb = torch.tensor([next_state], dtype=torch.float).to(self.device)
        not_done_mb = torch.tensor([[not done]], dtype=torch.float).to(self.device)
        if self.memory_type == "uniform":
            self.memory.add((state_mb, action_mb, reward_mb, next_state_mb, not_done_mb))
        elif self.memory_type == "prioritized":
            priority = (abs(reward) + self.epsilon) ** self.alpha if pretrain else self.memory.highest_priority()
            self.memory.add((state_mb, action_mb, reward_mb, next_state_mb, not_done_mb), priority)

    def act(self, state, noise=True, explore_probability=1.0, truncate=False, max_pos=2.0):
        positions = state[-self.num_assets:]
        state_tensor = torch.from_numpy(state).float().unsqueeze(0).to(self.device)
        self.actor_local.eval()
        with torch.no_grad():
            actions = self.actor_local(state_tensor).cpu().data.numpy()[0]
        self.actor_local.train()
        if noise:
            noise_sample = self.noise.sample(truncate=truncate, max_pos=max_pos, positions=positions, actions=actions)
            actions += explore_probability * noise_sample
             # Aggiungi questo:
            if np.random.random() < 0.30:  # 5% delle volte
                asset_idx = np.random.randint(0, len(actions))
                actions[asset_idx] += np.random.choice([-0.2, 0.2])  # Forza un trade significativo

            # Forza un cambio di direzione periodicamente per evitare accumulazione persistente
            if np.random.random() < 0.05:  # 5% delle volte
                # Inverti la direzione delle azioni per tutti gli asset
                actions = -actions * 0.1  # Scalati per non essere troppo estremi
        actions = actions * 5
        return actions

    def soft_update(self, local_model, target_model, tau):
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau * local_param.data + (1.0 - tau) * target_param.data)

    def pretrain(self, env, total_steps=PRETRAIN):
        env.reset()
        with torch.no_grad():
            for i in range(total_steps):
                state = env.get_state()
                if self.actor_local is None:
                    actions = np.random.uniform(-0.1, 0.1, self.num_assets)
                else:
                    actions = self.act(state, truncate=(not env.squared_risk), max_pos=env.max_pos_per_asset, explore_probability=2.0)
                reward = env.step(actions)
                next_state = env.get_state()
                done = env.done
                self.step(state, actions, reward, next_state, done, pretrain=True)
                if done:
                    env.reset()

    def train(
        self,
        env,
        total_episodes=100,
        tau_actor=TAU_ACTOR,
        tau_critic=TAU_CRITIC,
        lr_actor=LR_ACTOR,
        lr_critic=LR_CRITIC,
        weight_decay_actor=WEIGHT_DECAY_ACTOR,
        weight_decay_critic=WEIGHT_DECAY_CRITIC,
        total_steps=PRETRAIN,
        weights=WEIGHTS,
        freq=10,
        fc1_units_actor=FC1_UNITS_ACTOR,
        fc2_units_actor=FC2_UNITS_ACTOR,
        fc3_units_actor=FC3_UNITS_ACTOR,
        fc1_units_critic=FC1_UNITS_CRITIC,
        fc2_units_critic=FC2_UNITS_CRITIC,
        fc3_units_critic=FC3_UNITS_CRITIC,
        decay_rate=DECAY_RATE,
        explore_stop=EXPLORE_STOP,
        tensordir="runs/portfolio/",
        learn_freq=10,
        plots=False,
        progress="tqdm",
        features_per_asset=0,  
        encoding_size=0,       
        clip_grad_norm=1.0,
        checkpoint_path=None,
        checkpoint_freq=10,
        resume_from=None,
        early_stop_patience=20   # Numero di episodi senza miglioramento per early stopping
    ):
        self.last_checkpoint_episode = -1
        if not os.path.isdir(weights):
            os.makedirs(weights, exist_ok=True)
        if checkpoint_path is None:
            checkpoint_path = weights

        writer = SummaryWriter(log_dir=tensordir)

        checkpoint = None
        asset_encoder_dim = None
        actor_dict = None
        
        if resume_from and os.path.exists(resume_from):
            print(f"Analisi del checkpoint {resume_from} per determinare le dimensioni corrette...")
            checkpoint = torch.load(resume_from, map_location=self.device)
            if 'actor_state_dict' in checkpoint:
                actor_dict = checkpoint['actor_state_dict']
                if 'asset_encoder.fc2.weight' in actor_dict:
                    encoder_shape = actor_dict['asset_encoder.fc2.weight'].shape
                    if len(encoder_shape) >= 1:
                        asset_encoder_dim = encoder_shape[0]
                        print(f"Dimensione dell'encoder degli asset: {asset_encoder_dim}")
                if 'attention.weight' in actor_dict:
                    attention_shape = actor_dict['attention.weight'].shape
                    if len(attention_shape) > 1:
                        encoding_size = attention_shape[1]
                        print(f"Adattamento encoding_size a {encoding_size} dal checkpoint")

        attention_dim = None
        encoder_output_size = None
        
        if actor_dict is not None:
            if 'attention.weight' in actor_dict:
                attention_shape = actor_dict['attention.weight'].shape
                if len(attention_shape) > 1:
                    attention_dim = attention_shape[1]
                    print(f"Dimensione attention: {attention_dim}")
            if 'asset_encoder.fc2.weight' in actor_dict:
                encoder_shape = actor_dict['asset_encoder.fc2.weight'].shape
                if len(encoder_shape) >= 1:
                    encoder_output_size = encoder_shape[0]
                    print(f"Dimensione encoder output: {encoder_output_size}")

        self.actor_local = EnhancedPortfolioActor(
            env.state_size, 
            self.num_assets, 
            features_per_asset,
            fc1_units=fc1_units_actor,
            fc2_units=fc2_units_actor,
            encoding_size=encoding_size,
            use_attention=True,
            attention_size=attention_dim,
            encoder_output_size=encoder_output_size
        )

        if self.use_enhanced_actor and features_per_asset > 0:
            if encoding_size == 0:
                encoding_size = 16
            if asset_encoder_dim is not None:
                print(f"Utilizzando dimensione encoder dal checkpoint: {asset_encoder_dim}")
                encoding_size = asset_encoder_dim
            print(f"Inizializzazione EnhancedPortfolioActor con encoding_size={encoding_size}")
            asset_encoder = AssetEncoder(features_per_asset=features_per_asset, encoding_size=encoding_size, seed=0)
            self.actor_local = EnhancedPortfolioActor(
                env.state_size, 
                self.num_assets, 
                features_per_asset,
                fc1_units=fc1_units_actor,
                fc2_units=fc2_units_actor,
                encoding_size=encoding_size,
                use_attention=True,
                attention_size=attention_dim,
                encoder_output_size=encoder_output_size
            )
            self.actor_target = EnhancedPortfolioActor(
                env.state_size, 
                self.num_assets, 
                features_per_asset,
                fc1_units=fc1_units_actor,
                fc2_units=fc2_units_actor,
                encoding_size=encoding_size,
                use_attention=True
            )
        else:
            pass

        self.critic_local = PortfolioCritic(
            env.state_size, 
            self.num_assets, 
            fcs1_units=fc1_units_critic, 
            fc2_units=fc2_units_critic,
            fc3_units=fc3_units_critic,
            use_batch_norm=self.use_batch_norm
        )
        self.critic_target = PortfolioCritic(
            env.state_size, 
            self.num_assets, 
            fcs1_units=fc1_units_critic, 
            fc2_units=fc2_units_critic,
            fc3_units=fc3_units_critic,
            use_batch_norm=self.use_batch_norm
        )
        
        actor_optimizer = optim.Adam(self.actor_local.parameters(), lr=lr_actor, weight_decay=weight_decay_actor)
        actor_lr_scheduler = lr_scheduler.StepLR(actor_optimizer, step_size=100, gamma=0.5)
        
        critic_optimizer = optim.Adam(self.critic_local.parameters(), lr=lr_critic, weight_decay=weight_decay_critic)
        critic_lr_scheduler = lr_scheduler.StepLR(critic_optimizer, step_size=100, gamma=0.5)

        if not resume_from:
            model_file = os.path.join(weights, "portfolio_actor_initial.pth")
            torch.save(self.actor_local.state_dict(), model_file)

        mean_rewards = deque(maxlen=10)
        cum_rewards = []
        actor_losses = deque(maxlen=10)
        critic_losses = deque(maxlen=10)
        portfolio_values = deque(maxlen=10)
        sharpe_ratios = deque(maxlen=10)

        i = 0
        start_episode = 0
        N_train = total_episodes * env.T // learn_freq
        beta = self.beta0
        self.reset()
        n_train = 0

        # Variabili per early stopping
        best_mean_reward = -np.inf
        no_improvement_count = 0

        if checkpoint is not None:
            print(f"Riprendendo l'addestramento da {resume_from}")
            try:
                self.actor_local.load_state_dict(checkpoint['actor_state_dict'])
                self.critic_local.load_state_dict(checkpoint['critic_state_dict'])
                self.actor_target.load_state_dict(checkpoint['actor_target_state_dict'])
                self.critic_target.load_state_dict(checkpoint['critic_target_state_dict'])
                if 'metrics' in checkpoint:
                    metrics = checkpoint['metrics']
                    if 'mean_rewards' in metrics and metrics['mean_rewards']:
                        mean_rewards = deque(metrics['mean_rewards'], maxlen=10)
                    if 'cum_rewards' in metrics and metrics['cum_rewards']:
                        cum_rewards = metrics['cum_rewards']
                    if 'portfolio_values' in metrics and metrics['portfolio_values']:
                        portfolio_values = deque(metrics['portfolio_values'], maxlen=10)
                    if 'sharpe_ratios' in metrics and metrics['sharpe_ratios']:
                        sharpe_ratios = deque(metrics['sharpe_ratios'], maxlen=10)
                    if 'actor_losses' in metrics and metrics['actor_losses']:
                        actor_losses = deque(metrics['actor_losses'], maxlen=10)
                    if 'critic_losses' in metrics and metrics['critic_losses']:
                        critic_losses = deque(metrics['critic_losses'], maxlen=10)
                start_episode = checkpoint['episode']
                i = checkpoint.get('iteration', start_episode * env.T)
                n_train = checkpoint.get('n_train', start_episode * env.T // learn_freq)
                self.last_checkpoint_episode = start_episode - 1
                print(f"Addestramento ripreso dall'episodio {start_episode}")
            except Exception as e:
                print(f"Errore durante il caricamento del checkpoint: {e}")
                print("Iniziando nuovo addestramento...")
                start_episode = 0

        if start_episode == 0:
            Node.reset_count()
            self.pretrain(env, total_steps=total_steps)

        if progress == "tqdm_notebook":
            from tqdm import tqdm_notebook
            range_total_episodes = tqdm_notebook(range(start_episode, total_episodes))
            progress_bar = range_total_episodes
        elif progress == "tqdm":
            from tqdm import tqdm
            range_total_episodes = tqdm(range(start_episode, total_episodes))
            progress_bar = range_total_episodes
        else:
            range_total_episodes = range(start_episode, total_episodes)
            progress_bar = None

        for episode in range_total_episodes:
            episode_rewards = []
            env.reset()
            state = env.get_state()
            done = env.done
            train_iter = 0
            episode_complete = False
            # Aggiungi questo per forzare cambi di direzione regolari
            force_reset_counter = 0

            while not done:
                # Incrementa il contatore a ogni step, non solo durante l'apprendimento
                force_reset_counter += 1
                if force_reset_counter >= 50:  # Ogni 100 step
                    # Reset parziale delle posizioni per forzare il trading
                    env.positions *= 0.5  # Riduci tutte le posizioni del 50%
                    force_reset_counter = 0
                    #print(f"Forzato reset parziale delle posizioni al timestep {i}")
                explore_probability = explore_stop + (1 - explore_stop) * np.exp(-decay_rate * i)
                actions = self.act(state, truncate=(not env.squared_risk), max_pos=env.max_pos_per_asset, explore_probability=explore_probability)
                reward = env.step(actions)
                for j, ticker in enumerate(env.tickers):
                    writer.add_scalar(f"Portfolio/Position/{ticker}", env.positions[j], i)
                    writer.add_scalar(f"Portfolio/Action/{ticker}", actions[j], i)
                writer.add_scalar("Portfolio/TotalValue", env.get_portfolio_value(), i)
                writer.add_scalar("Portfolio/Cash", env.cash, i)
                writer.add_scalar("Portfolio/Reward", reward, i)
                next_state = env.get_state()
                done = env.done
                self.step(state, actions, reward, next_state, done)
                state = next_state
                episode_rewards.append(reward)
                i += 1
                train_iter += 1
                if done:
                    episode_complete = True
                    self.reset()
                    total_reward = np.sum(episode_rewards)
                    mean_rewards.append(total_reward)
                    portfolio_metrics = env.get_real_portfolio_metrics()
                    portfolio_values.append(portfolio_metrics['final_portfolio_value'])
                    sharpe_ratios.append(portfolio_metrics['sharpe_ratio'])
                    
                    # Early stopping logging
                    current_mean_reward = np.mean(mean_rewards)
                    print(f"Episodio {episode}: Ricompensa media = {current_mean_reward:.4f}, Best = {best_mean_reward:.4f}, No Improvement Count = {no_improvement_count}")
                    if current_mean_reward > best_mean_reward:
                        best_mean_reward = current_mean_reward
                        no_improvement_count = 0
                    else:
                        no_improvement_count += 1
                    if no_improvement_count >= early_stop_patience:
                        print(f"Early stopping: nessun miglioramento per {early_stop_patience} episodi.")
                        break

                    if (episode > 0) and (episode % 5 == 0):
                        mean_r = np.mean(mean_rewards)
                        cum_rewards.append(mean_r)
                        mean_portfolio_value = np.mean(portfolio_values)
                        mean_sharpe = np.mean(sharpe_ratios)
                        writer.add_scalar("Portfolio/AvgReward", mean_r, episode)
                        writer.add_scalar("Portfolio/AvgPortfolioValue", mean_portfolio_value, episode)
                        writer.add_scalar("Portfolio/AvgSharpeRatio", mean_sharpe, episode)
                        writer.add_scalar("Loss/ActorLoss", np.mean(actor_losses), episode)
                        writer.add_scalar("Loss/CriticLoss", np.mean(critic_losses), episode)
                if train_iter % learn_freq == 0:
                    n_train += 1
                    if self.memory_type == "uniform":
                        transitions = self.memory.sample(self.batch_size)
                        batch = Transition(*zip(*transitions))
                        states_mb = torch.cat(batch.state)
                        actions_mb = torch.cat(batch.action)
                        rewards_mb = torch.cat(batch.reward)
                        next_states_mb = torch.cat(batch.next_state)
                        dones_mb = torch.cat(batch.dones)
                    elif self.memory_type == "prioritized":
                        transitions, indices = self.memory.sample(self.batch_size)
                        batch = Transition(*zip(*transitions))
                        states_mb = torch.cat(batch.state)
                        actions_mb = torch.cat(batch.action)
                        rewards_mb = torch.cat(batch.reward)
                        next_states_mb = torch.cat(batch.next_state)
                        dones_mb = torch.cat(batch.dones)
                    actions_next = self.actor_target(next_states_mb)
                    Q_targets_next = self.critic_target(next_states_mb, actions_next)
                    Q_targets = rewards_mb + (self.gamma * Q_targets_next * dones_mb)
                    Q_expected = self.critic_local(states_mb, actions_mb)
                    td_errors = F.l1_loss(Q_expected, Q_targets, reduction="none")
                    if self.memory_type == "prioritized":
                        sum_priorities = self.memory.sum_priorities()
                        probabilities = (self.memory.retrieve_priorities(indices) / sum_priorities).reshape((-1, 1))
                        is_weights = torch.tensor(1 / ((self.max_size * probabilities) ** beta), dtype=torch.float)
                        is_weights /= is_weights.max()
                        beta = (1 - self.beta0) * (n_train / N_train) + self.beta0
                        for i_enum, index in enumerate(indices):
                            self.memory.update(index, (abs(float(td_errors[i_enum].data)) + self.epsilon) ** self.alpha)
                        critic_loss = (is_weights * (td_errors ** 2)).mean() / 2
                    else:
                        critic_loss = (td_errors ** 2).mean() / 2
                    critic_losses.append(critic_loss.data.item())
                    critic_optimizer.zero_grad()
                    critic_loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.critic_local.parameters(), clip_grad_norm)
                    critic_optimizer.step()
                    critic_lr_scheduler.step()
                    actions_pred = self.actor_local(states_mb)
                    actor_loss = -self.critic_local(states_mb, actions_pred).mean()
                    actor_losses.append(actor_loss.data.item())
                    actor_optimizer.zero_grad()
                    actor_loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.actor_local.parameters(), clip_grad_norm)
                    actor_optimizer.step()
                    actor_lr_scheduler.step()
                    self.soft_update(self.critic_local, self.critic_target, tau_critic)
                    self.soft_update(self.actor_local, self.actor_target, tau_actor)


            if episode_complete and ((episode % freq) == 0 or episode == total_episodes - 1):
                actor_file = os.path.join(weights, f"portfolio_actor_{episode}.pth")
                critic_file = os.path.join(weights, f"portfolio_critic_{episode}.pth")
                torch.save(self.actor_local.state_dict(), actor_file)
                torch.save(self.critic_local.state_dict(), critic_file)
                if episode != self.last_checkpoint_episode and ((episode % checkpoint_freq == 0) or episode == total_episodes - 1):
                    self.last_checkpoint_episode = episode
                    checkpoint_file = os.path.join(checkpoint_path, f"checkpoint_ep{episode}.pt")
                    metrics = {
                        'mean_rewards': list(mean_rewards),
                        'cum_rewards': cum_rewards,
                        'portfolio_values': list(portfolio_values),
                        'sharpe_ratios': list(sharpe_ratios),
                        'actor_losses': list(actor_losses),
                        'critic_losses': list(critic_losses)
                    }
                    checkpoint = {
                        'episode': episode + 1,
                        'iteration': i,
                        'n_train': n_train,
                        'actor_state_dict': self.actor_local.state_dict(),
                        'critic_state_dict': self.critic_local.state_dict(),
                        'actor_target_state_dict': self.actor_target.state_dict(),
                        'critic_target_state_dict': self.critic_target.state_dict(),
                        'metrics': metrics
                    }
                    torch.save(checkpoint, checkpoint_file)
                    print(f"Checkpoint salvato: {checkpoint_file}")

        writer.export_scalars_to_json("./portfolio_scalars.json")
        writer.close()
        
        return {
            'final_rewards': mean_rewards,
            'cum_rewards': cum_rewards,
            'final_portfolio_values': portfolio_values,
            'final_sharpe_ratios': sharpe_ratios
        }
    
    def load_models(self, actor_path, critic_path=None):
        if self.actor_local is not None:
            checkpoint = torch.load(actor_path, map_location=self.device)
            model_dict = self.actor_local.state_dict()
            # Filtra solo i parametri che hanno le stesse dimensioni
            filtered_dict = {k: v for k, v in checkpoint.items() 
                        if k in model_dict and v.size() == model_dict[k].size()}
            
            # Carica i parametri filtrati
            self.actor_local.load_state_dict(filtered_dict, strict=False)
            
            if self.actor_target is not None:
                self.actor_target.load_state_dict(filtered_dict, strict=False)