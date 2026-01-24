
#! Ovo ide u main funkciju za generisanje podataka za trening Gating mreze

# from environment import make_env
# from stable_baselines3.common.env_util import make_vec_env
# from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack

# obss = []
# ls = []

# for m in [0, 1, 2, 5]:
#     cfg.env.mission = m
#     vec_env = make_vec_env(make_env,
#                         n_envs=1,
#                         seed=cfg.seed,
#                         vec_env_cls=DummyVecEnv,
#                         env_kwargs={ 'cfg':cfg, 'env_name': cfg.env_name, 'render_mode': 'human' if cfg.render else 'rgb_array' })

#     if cfg.algorithm.n_frames_stack > 1 and not cfg.algorithm.recurrent:
#         vec_env = VecFrameStack(vec_env, cfg.algorithm.n_frames_stack, channels_order='first')

#     for _ in range(1000):
#         obs = vec_env.reset()
#         obss.append(obs['mission'].copy())
#         ls.append(m if m != 5 else 3)

# import numpy as np
# obss = np.array(obss).squeeze()
# ls = np.array(ls)

# np.save('rollouts/missions.npy', obss)
# np.save('rollouts/labels.npy', ls)

# return 0


#! Ovo ide iznad main funkcije za trening Gating mreze

# vocab = [chr(x) for x in range(ord('a'), ord('z') + 1)]
# vocab.insert(0, '.')
# vocab.insert(0, ',')
# vocab.insert(0, ':')
# vocab.insert(0, '-')
# vocab.insert(0, '\n')
# vocab.insert(0, ' ')


# def decode_mission(msn):
#     tmp = ""

#     import numpy as np

#     if len(msn.shape) == 2:
#         msn = np.squeeze(msn)

#     for token in msn:
#         tmp += vocab[token]

#     return tmp.strip()


#! Ovo ide u main funkciju za trening Gating mreze

# import numpy as np
# import torch as th
# from policies import GatingNetwork
# from environment import make_env
# from stable_baselines3.common.env_util import make_vec_env
# from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack

# def custom_init_weights(m):
#     if isinstance(m, th.nn.Linear):
#         m.weight.data.normal_(0, 1)
#         m.weight.data *= 1 / th.sqrt(m.weight.data.pow(2).sum(1, keepdim=True))
#         if m.bias is not None:
#             m.bias.data.fill_(0)

# num_epochs = 2000
# batch_size = 512
# lr = 3e-3

# obss = np.load('rollouts/missions.npy')
# ls = np.load('rollouts/labels.npy')

# ind = np.random.permutation(len(obss))

# obss = obss[ind]
# ls = ls[ind]

# net = GatingNetwork(num_experts=4,
#                     arch=cfg.network.features_extractor_kwargs.arch,
#                     head=cfg.network.features_extractor_kwargs.head)
# net.apply(custom_init_weights)
# net = net.to('cuda')

# loss = th.nn.CrossEntropyLoss()
# opt = th.optim.Adam(net.parameters(), lr=lr)

# for epoch in range(num_epochs):
#     epoch_loss = 0

#     for b in range(0, len(obss), batch_size):
#         batch_obs = th.tensor(obss[b:b+batch_size], dtype=th.int64, device='cuda')
#         batch_labels = th.tensor(ls[b:b+batch_size], dtype=th.long, device='cuda')

#         opt.zero_grad()

#         preds = net(batch_obs)
#         l = loss(preds, batch_labels)

#         l.backward()
#         opt.step()

#         epoch_loss += l.item()
    
#     print(f'Epoch {epoch}: {epoch_loss}')

#     if epoch_loss == 0:
#         break

# th.save(net, 'models/gating.pth')

# vec_env = make_vec_env(make_env,
#                     n_envs=1,
#                     seed=cfg.seed,
#                     vec_env_cls=DummyVecEnv,
#                     env_kwargs={ 'cfg':cfg, 'env_name': cfg.env_name, 'render_mode': 'human' if cfg.render else 'rgb_array' })

# if cfg.algorithm.n_frames_stack > 1 and not cfg.algorithm.recurrent:
#     vec_env = VecFrameStack(vec_env, cfg.algorithm.n_frames_stack, channels_order='first')

# hits = 0
# for _ in range(100):
#     obs = vec_env.reset()
#     dec_msn = decode_mission(obs['mission'])
    
#     out = net(th.tensor(obs['mission'], dtype=th.int64, device='cuda'))
#     pred = th.argmax(th.nn.functional.softmax(out, dim=1), dim=1).item()
    
#     if dec_msn == 'go to goal' and pred == 3:
#         hits += 1
#     elif 'go to' in dec_msn and pred == 0:
#         hits += 1
#     elif 'toggle' in dec_msn and pred == 1:
#         hits += 1
#     elif 'pick up' in dec_msn and pred == 2:
#         hits += 1
    
# print(f'Accuracy: {hits/100:.2%}')
    
# return 0


#! Ovo ide u main funkciju za imitaciju

# if config_name == 'imitation':
#     model, mean_reward = imitate(cfg=cfg,
#                                  render_mode='rgb_array',
#                                  time=time)
# else:


#! Ovo ide u ppo.py za imitaciju

# from imitation.algorithms.bc import BC
# from imitation.algorithms.adversarial.gail import GAIL
# from imitation.algorithms.adversarial.airl import AIRL
# from imitation.data import rollout
# from imitation.data.wrappers import RolloutInfoWrapper
# from imitation.rewards.reward_nets import BasicRewardNet, BasicShapedRewardNet
# from imitation.util import logger as imit_logger

# from experts import Expert


# def imitate(cfg, render_mode, time):

#     save_path = f'models/{time}-{gethostname()}/imitation_{cfg.algo}_{cfg.env.problem}' if cfg.save_model else None

#     config = {
#         'policy_kwargs': cfg.network,
#         'learning_rate': linear_schedule(cfg.algorithm.model_kwargs.initial_learning_rate, cfg.algorithm.model_kwargs.final_learning_rate),
#         'verbose': cfg.verbose,
#         'batch_size': cfg.algorithm.model_kwargs.batch_size,
#         'n_epochs': cfg.algorithm.model_kwargs.n_epochs,
#         'n_steps': cfg.algorithm.model_kwargs.horizon,
#         'gamma': cfg.algorithm.model_kwargs.gamma,
#         'gae_lambda': cfg.algorithm.model_kwargs.gae_lambda,
#         'clip_range': cfg.algorithm.model_kwargs.clip_range,
#         'clip_range_vf': cfg.algorithm.model_kwargs.clip_range_vf if cfg.algorithm.model_kwargs.clip_range_vf > 0 else None, 
#         'normalize_advantage': cfg.algorithm.model_kwargs.normalize_advantage,
#         'ent_coef': cfg.algorithm.model_kwargs.ent_coef,
#         'vf_coef': cfg.algorithm.model_kwargs.vf_coef,
#         'max_grad_norm': cfg.algorithm.model_kwargs.max_grad_norm,
#         'seed': cfg.seed,
#         'use_sde': cfg.algorithm.model_kwargs.use_sde,
#         'sde_sample_freq': cfg.algorithm.model_kwargs.sde_sample_freq,
#         'tensorboard_log': f'logs/{time}-{cfg.env_name}/imitation_{cfg.algo}_{cfg.env.problem}'
#     }

#     rng = np.random.default_rng(cfg.seed)
#     set_seed(cfg.seed)

#     vec_env = make_vec_env(make_env,
#                            n_envs=cfg.algorithm.n_envs,
#                            seed=cfg.seed,
#                            vec_env_cls=SubprocVecEnv,
#                            wrapper_class=RolloutInfoWrapper,
#                            env_kwargs={ 'env_name': cfg.env_name, 'render_mode': render_mode, 'cfg': cfg })
    
#     imitation_logger = imit_logger.configure(f'logs/{time}-{gethostname()}/{cfg.algo}_{cfg.job_num}', ["stdout", "tensorboard"])

#     expert = Expert(cfg)

#     if cfg.algo == 'bc':
#         vec_env = VecTransposeImage(vec_env)
#         cfg.rollout_timesteps = None
#     else:
#         cfg.rollout_episodes = None
    
#     rollouts = rollout.rollout(
#         expert,
#         vec_env,
#         rollout.make_sample_until(min_timesteps=cfg.rollout_timesteps, min_episodes=cfg.rollout_episodes),
#         rng=rng
#     )

#     model = PPO(policy=CustomPPOPolicy,
#                 env=vec_env,
#                 **config)

#     match cfg.algo:
#         case 'bc':
#             print('Training BC...')
#             BC(observation_space=vec_env.observation_space,
#                action_space=vec_env.action_space,
#                demonstrations=rollouts,
#                rng=rng,
#                policy=model.policy,
#                batch_size=cfg.batch_size,
#                device=th.device('cuda'),
#                custom_logger=imitation_logger).train(n_epochs=cfg.n_epochs,
#                                                      log_interval=500,
#                                                      progress_bar=True,
#                                                      reset_tensorboard=False)

#         case 'gail':
#             print('Training GAIL...')
#             reward_net = BasicRewardNet(
#                 observation_space=vec_env.observation_space,
#                 action_space=vec_env.action_space,
#                 normalize_input_layer=None  # RunningNorm
#             )

#             GAIL(demonstrations=rollouts,
#                  demo_batch_size=cfg.demo_batch_size,
#                  gen_replay_buffer_capacity=cfg.gen_replay_buffer_capacity,
#                  n_disc_updates_per_round=cfg.n_disc_updates_per_round,
#                  venv=vec_env,
#                  gen_algo=model,
#                  reward_net=reward_net,
#                  allow_variable_horizon=True,
#                  custom_logger=imitation_logger).train(total_timesteps=cfg.n_steps)

#         case 'airl':
#             print('Training AIRL...')
#             reward_net = BasicShapedRewardNet(
#                 observation_space=vec_env.observation_space,
#                 action_space=vec_env.action_space,
#                 normalize_input_layer=None  # RunningNorm
#             )

#             AIRL(demonstrations=rollouts,
#                  demo_batch_size=cfg.demo_batch_size,
#                  gen_replay_buffer_capacity=cfg.gen_replay_buffer_capacity,
#                  n_disc_updates_per_round=cfg.n_disc_updates_per_round,
#                  venv=vec_env,
#                  gen_algo=model,
#                  reward_net=reward_net,
#                  allow_variable_horizon=True,
#                  custom_logger=imitation_logger).train(total_timesteps=cfg.n_steps)
    
#     if save_path is not None:
#         model.save(save_path)

#     print('Evaluating model...')
#     mean_reward, _ = evaluate_policy(
#         model,
#         vec_env,
#         n_eval_episodes=cfg.algorithm.n_eval_episodes
#     )

#     LOG.info(f'Average reward in {cfg.algorithm.n_eval_episodes} episodes: {mean_reward}')
    
#     vec_env.close()

#     return model, mean_reward