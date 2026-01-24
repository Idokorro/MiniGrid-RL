import warnings
import hydra

from omegaconf import DictConfig
from hydra.core.hydra_config import HydraConfig
from hydra.types import RunMode

from datetime import datetime

from ppo import train, test, distilling
from manual import ManualControl


@hydra.main(config_path='hydra_configs', config_name='manual', version_base=None)
def main(cfg: DictConfig):
    warnings.filterwarnings('ignore')

    time = datetime.now().strftime('%Y-%m-%d_%H-%M')

    config_name = HydraConfig.get().job.config_name
    
    model, mean_reward = train(cfg=cfg,
                               render_mode='rgb_array',
                               single_run=HydraConfig.get().mode == RunMode.RUN,
                               testing=config_name in ['testing', 'distilling', 'manual'],
                               time=time)
    
    if config_name == 'manual':
        man = ManualControl(cfg=cfg, model=model)
        man.run()

    elif config_name == 'distilling':
        distilling(model=model, cfg=cfg, time=time)

    if HydraConfig.get().mode == RunMode.RUN and config_name != 'manual':
        test(model=model, cfg=cfg)
    
    return mean_reward


if __name__ == '__main__':
    main()
