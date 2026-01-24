import random
import logging
import numpy as np
import torch as th

import pygame as pg
import pygame_widgets as pgw
from pygame_widgets.dropdown import Dropdown
from pygame_widgets.button import Button
from pygame_widgets.toggle import Toggle

from multiprocessing import Process, Pipe

from minigrid.core.actions import Actions
from minigrid.core.constants import COLOR_NAMES

from stable_baselines3 import PPO
from policies import CustomPPOPolicy, CustomRecurrentPPOPolicy
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack
from stable_baselines3.common.vec_env import VecTransposeImage

from sb3_contrib import RecurrentPPO

from environment import make_env

from ollama import chat


LOG = logging.getLogger(__name__)
LOG.setLevel(logging.DEBUG)

SYSTEM_PROMPT = """
You are giving instructions to a robot.
Available instructions are: "go to", "pick up", "toggle", "go to goal".
Available objects are: "key", "door", "ball", "box".
Available colors are: "red", "green", "yellow", "blue", "purple", "grey".

Give a simple answer, consisting only of available instructions, colors and objects.
Don't look for alternative routes.
Give your answer in a step by step format. Try to have as little as possible steps in your answer.
If the door is locked, a key of the same color is needed. Doors are unlocked by toogling them while holding the key.
Keys can only toggle locked doors of the same color. Robot must first pick up a key before toggling the door.
Boxes do not need keys to be toggled.
Consider everything directly accessible unless it is in another room.
Box can contain other objects with the same color. If the object you are looking for is not present but there is a box, try opening the box.
Make sure to reference only objects present in the scene.
Do not tell the robot to go into the room or to open the unlocked door.

Examples:

The scene contains:
- green ball
- yellow ball
- blue key
- purple door
- yellow door
- green key
- goal
Mission: go to goal'

Answer:
1. go to goal

The scene contains:
- green ball
- purple ball
- red box
- purple box
- green box
- blue key
Mission: pick up purple box'

Answer:
1. pick up purple box

The scene contains:
Two rooms. Left and right.
There is a locked grey door between the rooms
Left room contains:
- goal
- green ball
- yellow key
Right room contains:
- robot
- grey box
- red key
- grey ball
Mission: go to goal

Answer:
1. toggle grey box
2. pick up grey key
3. toggle grey door
4. go to goal
"""

def game_proc(conn, cfg):
    vec_env = make_vec_env(make_env,
                           n_envs=1,
                           seed=cfg.seed,
                           vec_env_cls=DummyVecEnv,
                           env_kwargs={ 'cfg':cfg, 'env_name': 'custom', 'render_mode': 'human', 'manual': True })

    if cfg.algorithm.n_frames_stack > 1 and not cfg.algorithm.recurrent:
        vec_env = VecTransposeImage(vec_env)
        vec_env = VecFrameStack(vec_env, cfg.algorithm.n_frames_stack, channels_order='first')

    obs = vec_env.reset()
    done = np.zeros((1,), dtype=bool)

    while True:
        conn.send({ 'state': obs, 'done': done})
        done = np.zeros((1,), dtype=bool)

        match conn.recv():
            case 'quit':
                vec_env.close()
                conn.close()
                break
            case 'reset':
                obs = vec_env.reset()
            case action:
                obs, rew, done, _ = vec_env.step([action])

                if np.all(done == 1):
                    LOG.info(f'Episode reward: {rew}')
                    obs = vec_env.reset()

        vec_env.render()


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    th.manual_seed(seed)
    if th.cuda.is_available():
        th.cuda.manual_seed_all(seed)


class ManualControl:
    def __init__(self, cfg, model):
        self.cfg = cfg
        self.model = model
        self.instructions = []
        self.game_loop_running = True
        self.auto_running = False
        self.gui = {}

        self.vocab = [chr(x) for x in range(ord('a'), ord('z') + 1)]
        self.vocab.insert(0, '.')
        self.vocab.insert(0, ',')
        self.vocab.insert(0, ':')
        self.vocab.insert(0, '-')
        self.vocab.insert(0, '\n')
        self.vocab.insert(0, ' ')
        self.msn_len = 32

        self.key_to_action = {
            'left': Actions.left,
            'right': Actions.right,
            'up': Actions.forward,
            'space': Actions.toggle,
            'pageup': Actions.pickup,
            'pagedown': Actions.drop,
            'tab': Actions.pickup,
            'left shift': Actions.drop,
            'right shift': Actions.done,
        }

        self.p_conn, c_conn = Pipe()
        self.game = Process(target=game_proc, args=(c_conn, cfg))
        self.game.start()

        pg.init()
        res = (450, 370)
        self.screen = pg.display.set_mode(res)
        pg.display.set_caption('Greatest AI in the world!!!')

        self.setup_gui()
    
    def encode_mission(self, msn):
        if self.vocab is None:
            return None
        
        tmp = np.zeros(self.msn_len, dtype=np.int64)

        for i, token in enumerate([*msn.lower()]):
            tmp[i] = self.vocab.index(token)

        tmp = np.expand_dims(tmp, axis=0)

        return tmp
    
    def decode_mission(self, msn):
        if self.vocab is None:
            return None
        
        tmp = ""

        for token in msn[0]:
            tmp += self.vocab[token]

        return tmp.strip()

    def toggle_auto_running(self, new_state):
        self.auto_running = new_state
    
    def get_instructions(self, env_mission=None):
        decoded_mission = self.decode_mission(env_mission)

        if self.gui['GPT'].getValue():

            LOG.debug(decoded_mission)

            stream = chat(
                model=self.cfg.llm.model,
                options={
                    'seed': self.cfg.seed,
                    'num_ctx': self.cfg.llm.num_ctx,
                    'repeat_last_n': self.cfg.llm.repeat_last_n,
                    'repeat_penalty': self.cfg.llm.repeat_penalty,
                    'temperature': self.cfg.llm.temperature,
                    'top_k': self.cfg.llm.top_k,
                    'top_p': self.cfg.llm.top_p,
                    'min_p': self.cfg.llm.min_p
                },
                stream=True,
                messages=[
                    {
                        'role': 'system',
                        'content': SYSTEM_PROMPT
                    },
                    {
                        'role': 'user',
                        'content': "".join(decoded_mission)
                    }
                ]
            )

            msg = ""
            for chunk in stream:
                txt = chunk['message']['content']
                print(txt, end='', flush=True)
                msg += txt
            print('\n')

            lines = msg[msg.find("</think>\n\n") + 10:].splitlines()
            mission = []

            for line in lines:
                line = line.strip()
                if len(line) > 0 and line[0].isdigit():
                    mission.append(line.split(' ', 1)[1])
            
            LOG.debug(f"Missions from LLM: {mission}")

        else:
            instruction = self.gui['mission'].getSelected()
                    
            if instruction in ['go to', 'toggle', 'pick up']:
                color = self.gui['color'].getSelected()
                obj = self.gui['obj'].getSelected()
                mission = [f'{instruction} {color} {obj}']
            elif instruction == 'move':
                direction = self.gui['direction'].getSelected()
                mission = [f'{instruction} {direction}']
            else:
                mission = [instruction]
            
        return mission

    def close_others(self, name):
        for key in self.gui.keys():
            if key != name and type(self.gui[key]) is Dropdown:
                self.gui[key].setDropped(False)


    def setup_gui(self):
        instructions = [
            '"left": Move Left',
            '"right": Move Right',
            '"up": Move Forward',
            '"space": Toggle',
            '"tab": Pick Up',
            '"left shift": Drop',
            '"right shift": Done',
            '"backspace": Reset Environment',
            '"escape": Quit'
        ]

        font = pg.font.Font(None, 32)
        font_height = font.get_height()

        self.gui['mode'] = Dropdown(
            self.screen, 10, 10, 100, 50, name='Select Mode',
            choices=['manual', 'auto'],
            borderRadius=3, colour=pg.Color('green'), direction='down', textHAlign='left',
            onClick=self.close_others, onClickParams=['mode']
        )
        
        self.gui['mission'] = Dropdown(
            self.screen, 120, 10, 100, 50, name='Mission',
            choices=['go to', 'toggle', 'pick up', 'drop', 'move', 'go to goal'],
            borderRadius=3, colour=pg.Color('green'), direction='down', textHAlign='left',
            onClick=self.close_others, onClickParams=['mission']
        )
        
        self.gui['color'] = Dropdown(
            self.screen, 230, 10, 100, 50, name='Color',
            choices=COLOR_NAMES,
            borderRadius=3, colour=pg.Color('green'), direction='down', textHAlign='left',
            onClick=self.close_others, onClickParams=['color']
        )
        
        self.gui['direction'] = Dropdown(
            self.screen, 230, 10, 100, 50, name='Direction',
            choices=['left', 'right', 'up', 'down'],
            borderRadius=3, colour=pg.Color('green'), direction='down', textHAlign='left',
            onClick=self.close_others, onClickParams=['direction']
        )

        self.gui['obj'] = Dropdown(
            self.screen, 340, 10, 100, 50, name='Object',
            choices=['key', 'ball', 'box', 'door'],
            borderRadius=3, colour=pg.Color('green'), direction='down', textHAlign='left',
            onClick=self.close_others, onClickParams=['obj']
        )

        gptl = font.render('LLM', True, pg.Color('white'))
        self.gui['GPT_label'] = gptl
        self.gui['GPT_label_rect'] = gptl.get_rect(topleft=(35, 70))

        self.gui['GPT'] = Toggle(
            self.screen, 35, 100, 50, 15, text='GPT',
            onColour=pg.Color('gray'), offColour=pg.Color('gray'),
            handleOnColour=pg.Color('red'), handleOffColour=pg.Color('green'),
        )

        self.gui['start'] = Button(
            self.screen, 10, 130, 100, 50, text='Start',
            radius=3, colour=pg.Color('green'),
            onClick=self.toggle_auto_running, onClickParams=[True]
        )
        
        self.gui['stop'] = Button(
            self.screen, 10, 190, 100, 50, text='Stop',
            radius=3, colour=pg.Color('green'),
            onClick=self.toggle_auto_running, onClickParams=[False]
        )
        
        self.gui['man_instr'] = []
        self.gui['man_instr_rect'] = []

        for i, inst in enumerate(instructions):
            ins = font.render(inst, True, pg.Color('white'))
            self.gui['man_instr'].append(ins)
            self.gui['man_instr_rect'].append(ins.get_rect(topleft=(10, 70 + i * font_height)))


    def run(self):
        obs = None
        episode_start = True
        hidden_state = None
        current_instruction = None

        while self.game_loop_running:
            if self.p_conn.poll():
                obs = self.p_conn.recv()

            events = pg.event.get()
            for event in events:
                if event.type == pg.QUIT:
                    self.game_loop_running = False
                    self.p_conn.send('quit')
                if event.type == pg.KEYDOWN:
                    key = pg.key.name(int(event.key))

                    if key == 'escape':
                        self.game_loop_running = False
                        self.p_conn.send('quit')
                    
                    if self.gui['mode'].getSelected() == 'manual':
                        if key == 'backspace':
                            self.p_conn.send('reset')
                        else:
                            if key in self.key_to_action.keys():
                                self.p_conn.send(self.key_to_action[key])
                                obs = None
            
            if obs is not None and self.gui['mode'].getSelected() == 'auto' and self.auto_running:
                if np.all(obs['done'] == 1):
                    self.toggle_auto_running(False)
                    episode_start = np.ones((1,), dtype=bool)
                    hidden_state = None
                    self.instructions = []
                    current_instruction = None
                    obs['done'] = np.zeros((1,), dtype=bool)
                    continue

                new_obs = obs['state']

                if len(self.instructions) == 0 and current_instruction is None:
                    self.instructions = self.get_instructions(new_obs['mission'])
                
                if len(self.instructions) > 0 and current_instruction is None:
                    current_instruction = self.instructions.pop(0)
                    LOG.debug(f'Current instruction: {current_instruction}')

                new_obs['mission'] = self.encode_mission(current_instruction)

                if self.cfg.algorithm.n_frames_stack > 1 and not self.cfg.algorithm.recurrent:
                    new_obs['mission'] = np.concatenate((new_obs['mission'],) * self.cfg.algorithm.n_frames_stack, axis=1)

                if self.model is None:
                    self.load_model()

                if current_instruction is not None:
                    if 'None' in current_instruction:
                        LOG.error('Invalid instruction')
                        self.toggle_auto_running(False)
                        episode_start = np.ones((1,), dtype=bool)
                        hidden_state = None
                        self.instructions = []
                        current_instruction = None
                        continue

                    action, hidden_state = self.model.predict(new_obs, hidden_state, episode_start, deterministic=True)
                    LOG.debug(f'Action: {action}')
                    episode_start = np.zeros((1,), dtype=bool)

                    self.p_conn.send(action)

                    if action == Actions.done:
                        current_instruction = None
                        episode_start = np.ones((1,), dtype=bool)
                        hidden_state = None
                        if not self.gui['GPT'].getValue():
                            self.toggle_auto_running(False)

                obs = None

            self.screen.fill((0, 0, 0))
            if self.gui['mode'].getSelected() == None:
                self.gui['mission'].hide()
                self.gui['mission'].reset()
                self.gui['mission'].setDropped(False)
                self.gui['color'].hide()
                self.gui['color'].reset()
                self.gui['color'].setDropped(False)
                self.gui['direction'].hide()
                self.gui['direction'].reset()
                self.gui['direction'].setDropped(False)
                self.gui['obj'].hide()
                self.gui['obj'].reset()
                self.gui['obj'].setDropped(False)
                self.gui['GPT'].hide()
                self.gui['start'].hide()
                self.gui['stop'].hide()

            if self.gui['mode'].getSelected() == 'manual':
                self.gui['mission'].hide()
                self.gui['mission'].reset()
                self.gui['mission'].setDropped(False)
                self.gui['color'].hide()
                self.gui['color'].reset()
                self.gui['color'].setDropped(False)
                self.gui['direction'].hide()
                self.gui['direction'].reset()
                self.gui['direction'].setDropped(False)
                self.gui['obj'].hide()
                self.gui['obj'].reset()
                self.gui['obj'].setDropped(False)
                self.gui['GPT'].hide()
                if self.gui['GPT'].getValue():
                    self.gui['GPT'].toggle()
                self.gui['start'].hide()
                self.gui['stop'].hide()
                
                for i in range(len(self.gui['man_instr'])):
                    self.screen.blit(self.gui['man_instr'][i], self.gui['man_instr_rect'][i])
            
            elif self.gui['mode'].getSelected() == 'auto':
                self.gui['mission'].show()
                self.screen.blit(self.gui['GPT_label'], self.gui['GPT_label_rect'])
                self.gui['GPT'].show()
                self.gui['start'].show()
                self.gui['stop'].show()

                if self.auto_running:
                    self.gui['mode'].disable()
                    self.gui['mission'].disable()
                    self.gui['obj'].disable()
                    self.gui['color'].disable()
                    self.gui['direction'].disable()
                    self.gui['GPT'].disable()
                    self.gui['start'].disable()
                    self.gui['stop'].enable()
                else:
                    self.gui['mode'].enable()
                    self.gui['mission'].enable()
                    self.gui['obj'].enable()
                    self.gui['color'].enable()
                    self.gui['direction'].enable()
                    self.gui['GPT'].enable()
                    self.gui['start'].enable()
                    self.gui['stop'].disable()

                if self.gui['mission'].getSelected() in ['go to', 'toggle', 'pick up']:
                    self.gui['color'].show()
                    self.gui['obj'].show()
                    self.gui['direction'].hide()
                    self.gui['direction'].reset()
                if self.gui['mission'].getSelected() in ['drop', 'go to goal']:
                    self.gui['color'].hide()
                    self.gui['color'].reset()
                    self.gui['direction'].hide()
                    self.gui['direction'].reset()
                    self.gui['obj'].hide()
                    self.gui['obj'].reset()
                if self.gui['mission'].getSelected() == 'move':
                    self.gui['direction'].show()
                    self.gui['color'].hide()
                    self.gui['color'].reset()
                    self.gui['obj'].hide()
                    self.gui['obj'].reset()
                
                if self.gui['GPT'].getValue():
                    self.gui['mission'].disable()
                    self.gui['mission'].reset()
                    self.gui['mission'].setDropped(False)
                    self.gui['color'].hide()
                    self.gui['color'].reset()
                    self.gui['color'].setDropped(False)
                    self.gui['obj'].hide()
                    self.gui['obj'].reset()
                    self.gui['obj'].setDropped(False)

            self.gui['mode'].hide()
            self.gui['mode'].show()
            pgw.update(events)
            pg.display.update()
        
        self.p_conn.close()
        self.game.join()
