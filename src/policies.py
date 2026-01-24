from typing import Callable, Dict, List, Optional, Tuple, Union

import os
import logging

import numpy as np
import torch as th
from torch import nn
from gymnasium import spaces

from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.dqn.policies import DQNPolicy
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

from sb3_contrib.common.recurrent.policies import RecurrentMultiInputActorCriticPolicy


LOG = logging.getLogger(__name__)


class CustomExtractor(BaseFeaturesExtractor):
    def __init__(self,
                 observation_space: spaces.Dict,
                 arch: Dict[str, List[int]],
                 n_frames_stack: int):
        # We do not know features-dim here before going over all the items,
        # so put something dummy for now. PyTorch requires calling
        # nn.Module.__init__ before adding modules
        super().__init__(observation_space, features_dim=1)

        self.n_frames_stack = n_frames_stack

        if type(observation_space) is spaces.Box:
            observation_space = spaces.Dict({
                'image': spaces.Box(low=0, high=255, shape=(3, 8, 8), dtype=np.uint8),
                'mission': spaces.Box(low=0, high=32, shape=(32,), dtype=np.int64),
                'direction': spaces.Discrete(4)
            })

        self.GRU = False
        extractors = {}
        total_concat_size = 0
        # We need to know size of the output of this extractor,
        # so go over all the spaces and compute output feature sizes
        for key, subspace in observation_space.spaces.items():
            if key not in arch:
                continue
            
            extractors[key] = nn.Sequential()

            first_layer = True

            for i, layer in enumerate(arch[key]):
                class_ = getattr(nn, layer[0])
                if layer[1]:
                    params = layer[1].copy()
                    if first_layer and layer[0] != 'Embedding':
                        params[0] = params[0] * self.n_frames_stack
                extractors[key].add_module(f'{key}_{layer[0]}_{i}', class_(*params) if layer[1] else class_())

                if layer[0] == 'GRU':
                    self.GRU = True
                
                first_layer = False
            
            total_concat_size += self.get_output_shape(extractors[key], subspace, key)[0]

        self.extractors = nn.ModuleDict(extractors)

        # Update the features dim manually
        self._features_dim = total_concat_size

    def forward(self, observations) -> th.Tensor:
        encoded_tensor_list = []

        if type(observations) is not dict:
            observations = {
                'image': observations[:, :192].reshape(-1, 8, 8, 3).permute(0, 3, 1, 2),
                'mission': observations[:, 192:224],
                'direction': nn.functional.one_hot(observations[:, 224].type(th.cuda.LongTensor), 4).type(th.cuda.FloatTensor)
            }

        for key, extractor in self.extractors.items():
            if key == 'mission':
                obs = observations[key].type(th.int64)

                if self.GRU:
                    _, hdn = extractor(obs)
                    out = hdn[-1]
                else:
                    out = extractor(obs)
            elif key == 'image' and observations[key].shape[1] != 3 * self.n_frames_stack:
                out = extractor(observations[key].permute(0, 3, 1, 2))
            else:
                out = extractor(observations[key])

            if len(out.shape) > 2:
                out = th.squeeze(out)
            
            encoded_tensor_list.append(out)

        return th.cat(encoded_tensor_list, dim=1)
    
    def get_output_shape(self, model, space, key):
        if key == 'direction':
            if isinstance(space, spaces.Box):
                n = space.shape[0]
            else:
                n = space.n
            return model(th.rand(n)).shape
        
        if key == 'mission':
            if self.GRU:
                _, hdn = model(th.randint(space.low[0], space.high[0], (1, *space.shape)))
                return [hdn[-1].shape[-1]]
            else:
                out = model(th.randint(space.low[0], space.high[0], (1, *space.shape)))
                return th.squeeze(out).shape

        return model(th.rand(*(space.shape))).view((-1, 1)).shape


class GatingNetwork(nn.Module):
    def __init__(self,
                 num_experts,
                 arch: Dict[str, List[int]],
                 head: Dict[str, List[int]]):
        super().__init__()

        self.num_experts = num_experts

        self.gating = nn.Sequential()
        
        for i, layer in enumerate(arch):
            class_ = getattr(nn, layer[0])
            self.gating.add_module(f'{layer[0]}_{i}', class_(*layer[1]) if layer[1] else class_())
        
        self.head = nn.Sequential()

        head[-1][1][-1] = num_experts
        
        for i, layer in enumerate(head):
            class_ = getattr(nn, layer[0])
            if layer[1]:
                params = layer[1].copy()
            self.head.add_module(f'{layer[0]}_{i}', class_(*params) if layer[1] else class_())
    
    def forward(self, observations) -> th.Tensor:
        _, hdn = self.gating(observations[:,-32:])
        gt_out = hdn[-1]

        return self.head(gt_out)


class MoeExtractor(BaseFeaturesExtractor):
    def __init__(self,
                 observation_space: spaces.Dict,
                 arch: Dict[str, List[int]],
                 head: Dict[str, List[int]]):
        # We do not know features-dim here before going over all the items,
        # so put something dummy for now. PyTorch requires calling
        # nn.Module.__init__ before adding modules
        super().__init__(observation_space, features_dim=1)

        self.experts = []
        
        gto_expert = th.load(os.path.join('trained_models', 'moe', 'gto.pth'), map_location='cuda', weights_only=False)
        self.experts.append(gto_expert)
        LOG.info(f'Loaded expert: gto.pth')
        
        tgl_expert = th.load(os.path.join('trained_models', 'moe', 'tgl.pth'), map_location='cuda', weights_only=False)
        self.experts.append(tgl_expert)
        LOG.info(f'Loaded expert: tgl.pth')
        
        pkp_expert = th.load(os.path.join('trained_models', 'moe', 'pkp.pth'), map_location='cuda', weights_only=False)
        self.experts.append(pkp_expert)
        LOG.info(f'Loaded expert: pkp.pth')
        
        gtg_expert = th.load(os.path.join('trained_models', 'moe', 'gtg.pth'), map_location='cuda', weights_only=False)
        self.experts.append(gtg_expert)
        LOG.info(f'Loaded expert: gtg.pth')

        for expert in self.experts:
            for param in expert.parameters():
                param.requires_grad = False
        
        self.num_experts = len(self.experts)

        self.gating_network = th.load(os.path.join('trained_models', 'moe', 'gating.pth'), map_location='cuda', weights_only=False)
        
        self._features_dim = 1

    def forward(self, obs: th.Tensor, deterministic: bool = False) -> tuple[th.Tensor, th.Tensor, th.Tensor]:
        msn_t = th.tensor(obs['mission'], dtype=th.int64, device='cuda')
        gt_logits = self.gating_network(msn_t)

        gt_sm = nn.functional.softmax(gt_logits, dim=1)
        gt_idx = th.argmax(gt_sm, dim=1)
        weights =  nn.functional.one_hot(gt_idx, num_classes=self.num_experts)

        acts = []
        for expert in self.experts:
            a, _ = expert.predict(obs, deterministic=deterministic)

            acts.append(a)

        acts = th.tensor(acts, device='cuda').view(-1,)

        acts = th.sum(acts * weights, dim=1)

        return acts


class MoeNetwork(nn.Module):
    def __init__(self):
        super().__init__()

        # IMPORTANT:
        # Save output dimensions, used to create the distributions
        self.latent_dim_pi = 7
        self.latent_dim_vf = 1

    def forward(self, features: th.Tensor) -> Tuple[th.Tensor, th.Tensor]:
        return features


class CustomPPOPolicy(ActorCriticPolicy):
    def __init__(self,
                 observation_space: spaces.Space,
                 action_space: spaces.Space,
                 lr_schedule: Callable[[float], float],
                 *args,
                 **kwargs):
        
        self.policy_kwargs = kwargs.pop('policy_kwargs', None)
        self.n_envs = kwargs.pop('n_envs', None)

        super().__init__(observation_space,
                         action_space,
                         lr_schedule,
                         features_extractor_class=CustomExtractor,
                         optimizer_kwargs={'eps': kwargs.pop('optim_eps')},
                         *args,
                         **kwargs)
    
    @staticmethod
    def init_weights(module: nn.Module, gain: float = 1) -> None:
        if isinstance(module, nn.Conv2d):
            nn.init.orthogonal_(module.weight, gain=gain)
            if module.bias is not None:
                module.bias.data.fill_(0.0)

        if isinstance(module, nn.Linear):
            module.weight.data.normal_(0, 1)
            module.weight.data *= 1 / th.sqrt(module.weight.data.pow(2).sum(1, keepdim=True))
            if module.bias is not None:
                module.bias.data.fill_(0)


class CustomMOEPolicy(ActorCriticPolicy):
    def __init__(self,
                 observation_space: spaces.Space,
                 action_space: spaces.Space,
                 lr_schedule: Callable[[float], float],
                 *args,
                 **kwargs):
        
        self.policy_kwargs = kwargs.pop('policy_kwargs', None)
        self.n_envs = kwargs.pop('n_envs', None)

        super().__init__(observation_space,
                         action_space,
                         lr_schedule,
                         features_extractor_class=MoeExtractor,
                         optimizer_kwargs={'eps': kwargs.pop('optim_eps')},
                         *args,
                         **kwargs)
    
    @staticmethod
    def init_weights(module: nn.Module, gain: float = 1) -> None:
        return
    
    def _build_mlp_extractor(self) -> None:
        self.mlp_extractor = MoeNetwork()
    
    def predict(
        self,
        observation: Union[np.ndarray, dict[str, np.ndarray]],
        state: Optional[tuple[np.ndarray, ...]] = None,
        episode_start: Optional[np.ndarray] = None,
        deterministic: bool = False,
    ) -> tuple[np.ndarray, Optional[tuple[np.ndarray, ...]]]:
        
        with th.no_grad():
            actions = self.features_extractor(observation, deterministic)
        
        return actions.cpu().numpy(), state


class CustomRecurrentPPOPolicy(RecurrentMultiInputActorCriticPolicy):
    def __init__(self,
                 observation_space: spaces.Space,
                 action_space: spaces.Space,
                 lr_schedule: Callable[[float], float],
                 *args,
                 **kwargs):
        
        self.policy_kwargs = kwargs.pop('policy_kwargs')

        super().__init__(observation_space,
                         action_space,
                         lr_schedule,
                         features_extractor_class=CustomExtractor,
                         optimizer_kwargs={'eps': kwargs.pop('optim_eps')},
                         *args,
                         **kwargs)
    
    @staticmethod
    def init_weights(module: nn.Module, gain: float = 1) -> None:
        if isinstance(module, nn.Conv2d):
            nn.init.orthogonal_(module.weight, gain=gain)
            if module.bias is not None:
                module.bias.data.fill_(0.0)

        if isinstance(module, nn.Linear):
            module.weight.data.normal_(0, 1)
            module.weight.data *= 1 / th.sqrt(module.weight.data.pow(2).sum(1, keepdim=True))
            if module.bias is not None:
                module.bias.data.fill_(0)


class CustomDQNPolicy(DQNPolicy):
    def __init__(self,
                 observation_space: spaces.Space,
                 action_space: spaces.Space,
                 lr_schedule: Callable[[float], float],
                 *args,
                 **kwargs) -> None:
        self.policy_kwargs = kwargs.pop('policy_kwargs')

        super().__init__(observation_space,
                         action_space,
                         lr_schedule,
                         features_extractor_class=CustomExtractor,
                         optimizer_kwargs={'eps': kwargs.pop('optim_eps')},
                         *args,
                         **kwargs)