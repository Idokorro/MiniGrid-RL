import numpy as np
import gymnasium as gym
from minigrid.wrappers import FullyObsWrapper, RGBImgObsWrapper
from custom_env import PlaygroundEnv


gym.logger.min_level = gym.logger.ERROR


def make_env(env_name, render_mode, cfg=None, manual=False):
    if env_name == 'custom':
        env = PlaygroundEnv(render_mode=render_mode, cfg=cfg, manual=manual)
    else:
        env = gym.make(env_name, render_mode=render_mode)
    
    # env = RGBImgObsWrapper(env, tile_size=32)  # Za single trening sa RGB slikom
    # env = FullyObsWrapper(env)
    
    if manual:
        env = LLMDescriptionWrapper(env)
    else:
        env = TokenizeVocabWrapper(env)
    
    # env = RGBWithSymbolsWrapper(env)  # Za imitation trening sa RGB slikom
    # env = ObservationPaddingWrapper(env)
    # env = ImageNormalizationWrapper(env)

    if cfg.algorithm.n_frames_stack > 1 and not cfg.algorithm.recurrent:
        env = Discrete2BoxWrapper(env)
    
    if cfg is not None and 'algo' in cfg.keys() and cfg.algo != 'bc':
        env = ObservationFlattenWrapper(env)
    
    return env


class ImageNormalizationWrapper(gym.ObservationWrapper):
    def __init__(self, env):
        gym.ObservationWrapper.__init__(self, env)

        (w, h, c) = env.observation_space['image'].shape

        self.observation_space['image'] = gym.spaces.Box(low=0, high=2, shape=(c, w, h), dtype=np.float16)
        return None

    def observation(self, obs):
        obs['image'] = np.transpose(obs['image'].astype(np.float16) / 32, (2, 0, 1))

        return obs


class RGBWithSymbolsWrapper(RGBImgObsWrapper):
    def __init__(self, env, tile_size=32):
        symbols_space = env.observation_space['image']
        super().__init__(env, tile_size=tile_size)

        self.observation_space['symbols'] = symbols_space

        x = 3
    
    def observation(self, obs):
        symbols = obs['image'].copy()
        obs = super().observation(obs)
        obs['symbols'] = symbols

        return obs


class TokenizeVocabWrapper(gym.ObservationWrapper):
    def __init__(self, env):
        gym.ObservationWrapper.__init__(self, env)

        # self.vocab = env.observation_space['mission'].ordered_placeholders
        self.vocab = [chr(x) for x in range(ord('a'), ord('z') + 1)]
        self.vocab.insert(0, '.')
        self.vocab.insert(0, ',')
        self.vocab.insert(0, ':')
        self.vocab.insert(0, '-')
        self.vocab.insert(0, '\n')
        self.vocab.insert(0, ' ')

        self.msn_len = 32  # len(self.vocab)
        
        self.observation_space = gym.spaces.Dict({
            'direction': env.observation_space['direction'],
            'image': env.observation_space['image']
        })

        self.observation_space['mission'] = gym.spaces.Box(low=0, high=32, shape=(self.msn_len,), dtype=np.int64)
    
    def _calculate_indexes(self, msn):
        if self.vocab is None:
            return None
        
        tmp = np.zeros(self.msn_len, dtype=np.int64)

        for i, token in enumerate([*msn.lower()]):
            tmp[i] = self.vocab.index(token)

        return tmp


    def observation(self, obs):
        tmp = {
            'direction': obs['direction'],
            'image': obs['image']
        }
        
        if self.vocab is not None:
            tmp['mission'] = self._calculate_indexes(obs['mission'])
            
        return tmp


class ObservationPaddingWrapper(gym.ObservationWrapper):
    def __init__(self, env):
        gym.ObservationWrapper.__init__(self, env)

        self.original_shape = env.observation_space['image'].shape
        self.observation_space['image'] = gym.spaces.Box(low=0, high=255, shape=(22,22,3), dtype=np.uint8)

    def observation(self, obs):
        width_odd = 0
        height_odd = 0
        if self.original_shape[0] % 2 != 0:
            width_odd = 1
        if self.original_shape[1] % 2 != 0:
            height_odd = 1

        hWidth = (22 - self.original_shape[0]) // 2
        hHeight = (22 - self.original_shape[1]) // 2
        
        obs['image'] = np.pad(obs['image'], ((hWidth, hWidth + width_odd), (hHeight, hHeight + height_odd), (0, 0)), mode='constant', constant_values=0)

        return obs


class Discrete2BoxWrapper(gym.ObservationWrapper):
    def __init__(self, env):
        gym.ObservationWrapper.__init__(self, env)

        self.observation_space['direction'] = gym.spaces.Box(low=0, high=1, shape=(4,), dtype=np.uint8)

    def observation(self, obs):
        d = obs['direction']
        obs['direction'] = np.zeros((4,), dtype=np.uint8)
        obs['direction'][d] = 1

        return obs


class LLMDescriptionWrapper(gym.ObservationWrapper):
    def __init__(self, env):
        gym.ObservationWrapper.__init__(self, env)

        self.msn_len = 4096

        self.vocab = [chr(x) for x in range(ord('a'), ord('z') + 1)]
        self.vocab.insert(0, '.')
        self.vocab.insert(0, ',')
        self.vocab.insert(0, ':')
        self.vocab.insert(0, '-')
        self.vocab.insert(0, '\n')
        self.vocab.insert(0, ' ')

        self.observation_space['mission'] = gym.spaces.Box(low=0, high=32, shape=(self.msn_len,), dtype=np.int64)
    
    def _calculate_indexes(self, msn):
        if self.vocab is None:
            return None
        
        tmp = np.zeros(self.msn_len, dtype=np.int64)

        for i, token in enumerate([*msn.lower()]):
            tmp[i] = self.vocab.index(token)

        return tmp
    
    def observation(self, obs):
        e = self.env

        while not isinstance(e, PlaygroundEnv):
            e = e.env
    
        tmp = {
            'direction': obs['direction'],
            'image': obs['image']
        }
        
        tmp['mission'] = e.llm_description + obs['mission']

        if self.vocab is not None:
            tmp['mission'] = self._calculate_indexes(tmp['mission'])
            
        return tmp


class ObservationFlattenWrapper(gym.ObservationWrapper):
    def __init__(self, env):
        gym.ObservationWrapper.__init__(self, env)

        self.observation_space = gym.spaces.Box(low=0, high=255, shape=(225,), dtype=np.int64)

    def observation(self, obs):
        new_obs = obs['image'].flatten()
        new_obs = np.concatenate((new_obs, obs['mission']))
        new_obs = np.concatenate((new_obs, np.expand_dims(obs['direction'], axis=0)))

        return new_obs