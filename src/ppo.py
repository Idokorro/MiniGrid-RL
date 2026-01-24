import torch as th
import numpy as np
import gymnasium as gym
import random
import logging
import os
import pickle

from socket import gethostname

from typing import Callable
from omegaconf import OmegaConf

from stable_baselines3 import PPO, DQN
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv, VecFrameStack
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.vec_env import VecTransposeImage

from sb3_contrib import RecurrentPPO

from environment import make_env
from policies import CustomPPOPolicy, CustomDQNPolicy, CustomRecurrentPPOPolicy, CustomMOEPolicy


LOG = logging.getLogger(__name__)

th.backends.cuda.matmul.allow_tf32 = True
th.backends.cudnn.allow_tf32 = True
th.set_default_dtype(th.float32)
th.set_float32_matmul_precision('high')


def linear_schedule(initial_value: float, final_value: float) -> Callable[[float], float]:

    def func(progress_remaining: float) -> float:
        return max(progress_remaining * initial_value, final_value)

    return func


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    th.manual_seed(seed)
    if th.cuda.is_available():
        th.cuda.manual_seed_all(seed)


def train(cfg, render_mode, single_run, testing, time):
    
    save_path = f'models/{time}-{gethostname()}/PPO_{cfg.job_num}' if cfg.save_model else None

    if cfg.algorithm.name == 'dqn':
        algo = DQN
        policy = CustomDQNPolicy

        config = {
            'policy_kwargs': cfg.network,
            'learning_rate': linear_schedule(cfg.algorithm.model_kwargs.initial_learning_rate, cfg.algorithm.model_kwargs.final_learning_rate),
            'verbose': cfg.verbose,
            'batch_size': cfg.algorithm.model_kwargs.batch_size,
            'gradient_steps': cfg.algorithm.model_kwargs.gradient_steps,
            'gamma': cfg.algorithm.model_kwargs.gamma,
            'tau': cfg.algorithm.model_kwargs.tau,
            'train_freq': cfg.algorithm.model_kwargs.train_freq,
            'exploration_fraction': cfg.algorithm.model_kwargs.exploration_fraction,
            'exploration_initial_eps': cfg.algorithm.model_kwargs.exploration_initial_eps,
            'exploration_final_eps': cfg.algorithm.model_kwargs.exploration_final_eps,
            'max_grad_norm': cfg.algorithm.model_kwargs.max_grad_norm,
            'seed': cfg.seed,
            'buffer_size': cfg.algorithm.model_kwargs.buffer_size,
            'target_update_interval': cfg.algorithm.model_kwargs.target_update_interval,
            'tensorboard_log': f'logs/{time}-{cfg.env_name}/DQN_{cfg.job_num}'
        }
    else:
        algo = RecurrentPPO if cfg.algorithm.recurrent else PPO
        
        if cfg.algorithm.recurrent:
            policy = CustomRecurrentPPOPolicy
        elif hasattr(cfg.network, 'moe') and cfg.network.moe:
            policy = CustomMOEPolicy
            del cfg.network.moe
        else:
            policy = CustomPPOPolicy
    
        if not cfg.algorithm.recurrent:
            del cfg.network.shared_lstm
            del cfg.network.enable_critic_lstm
            del cfg.network.lstm_hidden_size
            del cfg.network.n_lstm_layers

        config = {
            'policy_kwargs': cfg.network,
            'learning_rate': linear_schedule(cfg.algorithm.model_kwargs.initial_learning_rate, cfg.algorithm.model_kwargs.final_learning_rate),
            'verbose': cfg.verbose,
            'batch_size': cfg.algorithm.model_kwargs.batch_size,
            'n_epochs': cfg.algorithm.model_kwargs.n_epochs,
            'n_steps': cfg.algorithm.model_kwargs.horizon,
            'gamma': cfg.algorithm.model_kwargs.gamma,
            'gae_lambda': cfg.algorithm.model_kwargs.gae_lambda,
            'clip_range': cfg.algorithm.model_kwargs.clip_range,
            'clip_range_vf': cfg.algorithm.model_kwargs.clip_range_vf if cfg.algorithm.model_kwargs.clip_range_vf > 0 else None, 
            'normalize_advantage': cfg.algorithm.model_kwargs.normalize_advantage,
            'ent_coef': cfg.algorithm.model_kwargs.ent_coef,
            'vf_coef': cfg.algorithm.model_kwargs.vf_coef,
            'max_grad_norm': cfg.algorithm.model_kwargs.max_grad_norm,
            'seed': cfg.seed,
            'use_sde': cfg.algorithm.model_kwargs.use_sde,
            'sde_sample_freq': cfg.algorithm.model_kwargs.sde_sample_freq,
            'tensorboard_log': f'logs/{time}-{cfg.env_name}/PPO_{cfg.job_num}'
        }
    
    if cfg.seed is not None:
        set_seed(cfg.seed)

    vec_env = make_vec_env(make_env,
                           n_envs=cfg.algorithm.n_envs,
                           seed=cfg.seed,
                           vec_env_cls=SubprocVecEnv if single_run else DummyVecEnv,
                           env_kwargs={ 'cfg':cfg, 'env_name': cfg.env_name, 'render_mode': render_mode })

    if cfg.algorithm.n_frames_stack > 1 and not cfg.algorithm.recurrent:
        vec_env = VecTransposeImage(vec_env)
        vec_env = VecFrameStack(vec_env, cfg.algorithm.n_frames_stack, channels_order='first')

    if cfg.load_path is not None:
        model = algo.load(path=cfg.load_path,
                          policy=policy,
                          env=vec_env,
                          **config)
    else:
        model = algo(policy=policy,
                     env=vec_env,
                     **config)
    
    checkpoint_callback = None
    mean_reward = None

    # th.save(model.policy, 'trained_models/moe/pkp.pth')

    if not testing:
        if save_path is not None:
            checkpoint_callback = EvalCallback(vec_env,
                                               best_model_save_path=save_path,
                                               eval_freq=10000 // cfg.algorithm.n_envs,
                                               n_eval_episodes=cfg.algorithm.n_eval_episodes,
                                               deterministic=True,
                                               render=False)
        
        if not single_run:
            path = f'{config["tensorboard_log"]}/config_{cfg.job_num}-{gethostname()}.yaml'
            os.makedirs(os.path.dirname(path), exist_ok=True)

            with open(path, 'w') as f:
                OmegaConf.save(cfg, f)
        
        model.learn(total_timesteps=cfg.algorithm.total_timesteps, progress_bar=single_run, callback=checkpoint_callback)

        mean_reward, _ = evaluate_policy(
            model,
            vec_env,
            n_eval_episodes=cfg.algorithm.n_eval_episodes
        )

        LOG.info(f'Average reward in {cfg.algorithm.n_eval_episodes} episodes: {mean_reward}')
    
    vec_env.close()

    return model, mean_reward


def test(model, cfg):

    collecting = False
    if hasattr(cfg, 'collect_rollouts') and cfg.collect_rollouts:
        LOG.info('Collecting rollouts...')
        collecting = True
        images = None
        directions = None
        missions = None
        policies = None
    else:
        LOG.info(f'Testing {cfg.env_name}...')
    
    vec_env = make_vec_env(make_env,
                           n_envs=1,
                           seed=cfg.seed,
                           vec_env_cls=DummyVecEnv,
                           env_kwargs={ 'cfg':cfg, 'env_name': cfg.env_name, 'render_mode': 'human' if cfg.render else 'rgb_array' })

    if cfg.algorithm.n_frames_stack > 1 and not cfg.algorithm.recurrent:
        vec_env = VecTransposeImage(vec_env)
        vec_env = VecFrameStack(vec_env, cfg.algorithm.n_frames_stack, channels_order='first')
    
    rewards = []

    for i in range(cfg.algorithm.n_test_episodes):
        episode_starts = np.ones((1,), dtype=bool)
        dones = np.zeros((1,), dtype=bool)
        lstm_states = None
        rewards.append(0)
        ep_images = None
        ep_directions = None
        ep_missions = None
        ep_policies = None
        deterministic = cfg.deterministic if hasattr(cfg, 'deterministic') else True

        obs = vec_env.reset()
        
        while np.all(dones == 0):
            if collecting:
                obs_t = model.policy.obs_to_tensor(obs)[0]
                dis = model.policy.get_distribution(obs_t)
                policy = dis.distribution.probs.cpu().detach().numpy()
                action = dis.get_actions(deterministic=deterministic).cpu().detach().numpy()
            else:
                action, lstm_states = model.predict(obs,
                                                    state=lstm_states,
                                                    episode_start=episode_starts,
                                                    deterministic=deterministic)
            
            if collecting:
                if ep_images is None:
                    ep_images = obs['image']
                else:
                    ep_images = np.append(ep_images, obs['image'], axis=0)
                if ep_directions is None:
                    ep_directions = obs['direction']
                else:
                    ep_directions = np.append(ep_directions, obs['direction'], axis=0)
                if ep_missions is None:
                    ep_missions = obs['mission']
                else:
                    ep_missions = np.append(ep_missions, obs['mission'], axis=0)
                if ep_policies is None:
                    ep_policies = policy
                else:
                    ep_policies = np.append(ep_policies, policy, axis=0)

            obs, reward, dones, _ = vec_env.step(action)

            if collecting and any(reward):
                if images is None:
                    images = ep_images
                else:
                    images = np.append(images, ep_images, axis=0)
                if directions is None:
                    directions = ep_directions
                else:
                    directions = np.append(directions, ep_directions, axis=0)
                if missions is None:
                    missions = ep_missions
                else:
                    missions = np.append(missions, ep_missions, axis=0)
                if policies is None:
                    policies = ep_policies
                else:
                    policies = np.append(policies, ep_policies, axis=0)

            episode_starts = dones
            rewards[i] += reward[0]
            if cfg.render:
                vec_env.render()
        
        if not collecting:
            LOG.info(f'Episode {i} reward: {rewards[i]}')
        elif i % 100 == 0:
            LOG.info(f"Collected {i} episodes")
    
    if collecting:
        LOG.info(f'Collected {len(images)} samples.')
        
        path = 'rollouts/data.pkl'

        os.makedirs(os.path.dirname(path), exist_ok=True)

        observations = []
        for i in range(len(images)):
            observations.append({
                'image': images[i].tolist(),
                'direction': directions[i].tolist(),
                'mission': missions[i].tolist(),
                'policy': policies[i].tolist()
            })
        
        with open(path, 'wb') as f:
            pickle.dump(observations, f)
    
    LOG.info(f'Average reward: {sum(rewards) / len(rewards)}')
    vec_env.close()


def distilling(model, cfg, time):
    LOG.info('Distilling model...')

    save_path = f'models/{time}-{gethostname()}/distilling_{cfg.job_num}/model.zip' if cfg.save_model else None
    tb_writer = th.utils.tensorboard.SummaryWriter(f'logs/{time}-{cfg.env_name}/distilling_{gethostname()}')

    device = 'cuda' if th.cuda.is_available() else 'cpu'

    network = model.policy

    policies = []

    images = []
    directions = []
    missions = []

    for file_name in os.listdir('rollouts'):
        try:
            with open(os.path.join('rollouts', file_name), 'rb') as f:
                data = pickle.load(f)
                LOG.info(f'Loaded rollouts: {file_name}')
        except Exception as e:
            LOG.error(f'Error loading rollouts: {e}')
            return

        for d in data:
            policies.append(d['policy'])
            images.append(d['image'])
            directions.append(d['direction'])
            missions.append(d['mission'])
    
    policies = np.array(policies)
    images = np.array(images)
    directions = np.array(directions)
    missions = np.array(missions)

    indices = np.arange(len(policies))
    np.random.shuffle(indices)

    policies = policies[indices]
    images = images[indices]
    directions = directions[indices]
    missions = missions[indices]

    policies_bached = None
    obs_batched = []

    for i in range(len(policies) // cfg.dist_batch_size):
        start = i * cfg.dist_batch_size
        end = (i + 1) * cfg.dist_batch_size
        
        if policies_bached is None:
            policies_bached = np.expand_dims(policies[start:end], axis=0)
        else:
            policies_bached = np.append(policies_bached, np.expand_dims(policies[start:end], axis=0), axis=0)
        
        obs_batched.append({
            'image': images[start:end],
            'direction': directions[start:end],
            'mission': missions[start:end]
        })
    
    optimizer = th.optim.Adam(network.parameters(), lr=cfg.dist_learning_rate)
    lr_scheduler = th.optim.lr_scheduler.ExponentialLR(optimizer, gamma=cfg.dist_lr_decay)

    for epoch in range(cfg.dist_epochs):
        epoch_loss = 0
        last_loss = float('inf')

        for batch in range(len(obs_batched)):
            obs = obs_batched[batch]
            target_policy = th.tensor(policies_bached[batch]).to(device)

            obs_t = network.obs_to_tensor(obs)[0]
            dis = network.get_distribution(obs_t)
            policy = dis.distribution.probs

            loss = th.nn.functional.kl_div(policy.log(), target_policy, reduction='batchmean')

            epoch_loss += loss.item()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
        tb_writer.add_scalar('loss', epoch_loss / len(obs_batched), epoch)
        tb_writer.add_scalar('learning_rate', lr_scheduler.get_last_lr()[0], epoch)

        lr_scheduler.step()
            
        if save_path is not None and (epoch + 1) % 100 == 0 and epoch_loss < last_loss:
            model.save(save_path)
            last_loss = epoch_loss
            LOG.info('Model saved.')
            
        LOG.info(f'Epoch {epoch + 1}/{cfg.dist_epochs}, Loss: {epoch_loss / len(obs_batched)}')
    
    tb_writer.close()
