from custom_env import PlaygroundEnv
from minigrid.wrappers import FullyObsWrapper
from environment import TokenizeVocabWrapper
from experts import Expert
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import DummyVecEnv


class Cfg:
    def __init__(self, algo, env):
        self.algo = algo
        self.env = env
    
    def __getitem__(self, x):
        return getattr(self, x)

class Env:
    def __init__(self, problem, size, num_objects, see_through_walls, obstacles):
        self.problem = problem
        self.size = size
        self.num_objects = num_objects
        self.see_through_walls = see_through_walls
        self.obstacles = obstacles

cfg = Cfg('test', Env('gtg', 11, 6, True, True))

def make_env(cfg):
    env = PlaygroundEnv(render_mode='human', cfg=cfg, manual=False)
    env = FullyObsWrapper(env)
    return TokenizeVocabWrapper(env)

env = make_vec_env(
    make_env,
    n_envs=1,
    seed=1337,
    vec_env_cls=DummyVecEnv,
    env_kwargs={ 'cfg': cfg }
)

for _ in range(10):
    expert = Expert(cfg)

    obs = env.reset()
    done = False
    while not done:
        action = expert(obs, None, False)
        obs, reward, done, info = env.step(action[0])
    print(reward)