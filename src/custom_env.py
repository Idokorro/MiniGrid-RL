from __future__ import annotations
from math import floor
import random
from random import choice, randint

from minigrid.core.constants import COLOR_NAMES
from minigrid.core.grid import Grid
from minigrid.core.mission import MissionSpace
from minigrid.core.world_object import Ball, Box, Key, Door, Goal, Lava, Wall
from minigrid.minigrid_env import MiniGridEnv
from minigrid.core.actions import Actions


MISSION_TO_ACTION = {
        'go to goal': None,
        'go to': Actions.done,
        'open': Actions.toggle,
        'pick up': Actions.pickup,
        'drop': Actions.drop,
        'move': None
    }


class PlaygroundEnv(MiniGridEnv):
    """
    ## Description

    This environment is a room with all objects of all colors. Goal is to learn
    all possible missions.

    ## Mission Space

    Missions: full, gto, gtg, opn, pkp, drp, mov

    'go to the {color} {obj_type}'

    {color} is the color of the object. Can be 'red', 'green', 'blue', 'purple',
    'yellow' or 'grey'.
    {obj_type} is the type of the object. Can be 'key', 'ball', 'box'.

    ## Action Space

    | Num | Name         | Action               |
    |-----|--------------|----------------------|
    | 0   | left         | Turn left            |
    | 1   | right        | Turn right           |
    | 2   | forward      | Move forward         |
    | 3   | pickup       | Pick up object       |
    | 4   | drop         | Drop object          |
    | 5   | toggle       | Unlock door          |
    | 6   | done         | Done completing task |

    ## Observation Encoding

    - Each tile is encoded as a 3 dimensional tuple:
        `(OBJECT_IDX, COLOR_IDX, STATE)`
    - `OBJECT_TO_IDX` and `COLOR_TO_IDX` mapping can be found in
        [minigrid/core/constants.py](minigrid/core/constants.py)
    - `STATE` refers to the door state with 0=open, 1=closed and 2=locked

    ## Rewards

    A reward of '1 - 0.9 * (step_count / max_steps)' is given for success, and '0' for failure.

    ## Termination

    The episode ends if any one of the following conditions is met:

    1. The agent completes the mission.
    2. Timeout (see `max_steps`).

    """

    def __init__(self, **kwargs):
        c = kwargs.pop('cfg')
        self.cfg = c['env']
        self.manual = kwargs.pop('manual')
        self.reward = None
        self.mission_done = False
        self.cfg_cmd = self.cfg.mission if hasattr(self.cfg, 'mission') else None

        random.seed(c.seed)

        # Types of objects to be generated
        self.obj_types = ['key', 'ball', 'box', 'door']

        self.msn_commands = [
            'go to',      # any
            'toggle',     # box, door
            'pick up',    # key, ball, box
            'drop',       # empty
            'move',       # direction
            'go to goal'  # empty
        ]

        self.msn_directions = [
            'left',
            'right',
            'up',
            'down'
        ]

        mission_space = MissionSpace(
            mission_func=self._gen_mission,
            ordered_placeholders=[self.msn_commands, COLOR_NAMES, self.obj_types, self.msn_directions],
        )

        super().__init__(
            mission_space=mission_space,
            width=self.cfg.size,
            height=self.cfg.size,
            # Set this to True for maximum speed
            see_through_walls=self.cfg.see_through_walls,
            max_steps=self.cfg.size**2,
            **kwargs,
        )

    @staticmethod
    def _gen_mission(command: str, color: str, obj_type: str, direction: str):
        return f'{command} {color} {obj_type} {direction}'

    def _gen_grid(self, width, height):
        self.grid = Grid(width, height)

        self.target_action = None
        self.target_pos = None
        self.target_range = []
        self.mission = None
        self.llm_description = None

        # Generate the surrounding walls
        self.grid.wall_rect(0, 0, width, height)

        match self.cfg.problem:
            case 'full':
                objs, target_cmd = self._generate_full_map()
            case 'gto':
                objs, target_cmd = self._generate_gto_map()
            case 'gtg':
                objs, target_cmd = self._generate_gtg_map()
            case 'opn':
                objs, target_cmd = self._generate_open_map()
            case 'pkp':
                objs, target_cmd = self._generate_pkp_map()
            case 'drp':
                objs, target_cmd = self._generate_drop_map()
            case 'mov':
                objs, target_cmd = self._generate_move_map()
            case 'multi':
                objs, target_cmd = self._generate_multi_map()
            case _:
                raise ValueError(f'Invalid problem type given: {self.cfg.problem}')
        
        # Place obstacles
        if self.cfg.obstacles:
            for i in range(floor((self.cfg.size - 2)**2 * self.cfg.percent_obstacles)):
                if self.cfg.problem == 'multi':
                    while True:
                        obj_pos = (randint(1, self.cfg.size - 2), randint(1, self.cfg.size - 2))
                        if obj_pos[0] == (self.cfg.size // 2) or obj_pos[1] == (self.cfg.size // 2):
                            continue

                        for o in objs:
                            if obj_pos == o[2]:
                                break
                        else:
                            if obj_pos != self.agent_pos and not self.next2door(obj_pos):
                                break
                    
                    self.put_obj(Lava(), obj_pos[0], obj_pos[1])
                else:
                    self.place_obj(choice([Lava(), Wall()]))
        
        match target_cmd:
            case 'go to':
                while True:
                    obj_i = self.np_random.integers(0, len(objs))

                    target_type = objs[obj_i][0]
                    target_color = objs[obj_i][1]

                    if target_type != 'goal':
                        break

                self.mission = f'{target_cmd} {target_color} {target_type}'

                self.target_pos = objs[obj_i][2]
                self.target_action = Actions.done

            case 'toggle':
                while True:
                    obj = choice(objs)

                    if obj[0] in ['box', 'door']:
                        break

                self.mission = f'{target_cmd} {obj[1]} {obj[0]}'
                self.target_pos = obj[2]
                self.target_action = Actions.toggle

            case 'pick up':
                while True:
                    obj = choice(objs)

                    if obj[0] in ['box', 'key', 'ball']:
                        break
                
                self.mission = f'{target_cmd} {obj[1]} {obj[0]}'
                self.target_pos = obj[2]
                self.target_action = Actions.pickup

            case 'drop':
                self.mission = target_cmd
                self.target_action = Actions.drop

            case 'move':  # direction
                target_dir = self.np_random.choice(self.msn_directions)

                match target_dir:
                    case 'left':
                        for y in range(1, self.cfg.size - 1):
                            x = 1
                            while x < self.cfg.size - 1 and self.grid.get(x, y) is not None:
                                x += 1
                            
                            if x < self.cfg.size - 1:
                                self.target_range.append((x, y))
                    
                    case 'right':
                        for y in range(1, self.cfg.size - 1):
                            x = self.cfg.size - 2
                            while x > 0 and self.grid.get(x, y) is not None:
                                x -= 1

                            if x > 0:
                                self.target_range.append((x, y))
                    
                    case 'up':
                        for x in range(1, self.cfg.size - 1):
                            y = 1
                            while y < self.cfg.size - 1 and self.grid.get(x, y) is not None:
                                y += 1
                            
                            if y < self.cfg.size - 1:
                                self.target_range.append((x, y))
                    
                    case 'down':
                        for x in range(1, self.cfg.size - 1):
                            y = self.cfg.size - 2
                            while y > 0 and self.grid.get(x, y) is not None:
                                y -= 1

                            if y > 0:
                                self.target_range.append((x, y))
                
                self.mission = f'{target_cmd} {target_dir}'

            case 'go to goal':
                self.mission = target_cmd

                for o in objs:
                    if o[0] == 'goal':
                        self.target_pos = o[2]
                        self.target_action = None
                        break
                else:
                    raise ValueError(f'Invalid mission generated: {self.mission}')

    def step(self, action):
        obs, reward, terminated, truncated, info = super().step(action)

        if terminated:
            if self.mission != 'go to goal':
                self.mission_done = False
                self.reward = None
                reward = 0.
            return obs, float(reward), terminated, truncated, info
        
        if action == Actions.toggle:
            fwd_cell = self.grid.get(*self.front_pos)
            if isinstance(fwd_cell, Door) and self.carrying is not None:
                if fwd_cell.color == self.carrying.color:
                    self.carrying = None
            
        ax, ay = self.agent_pos
        arrived = False
        
        if not self.mission_done:
            if self.target_pos:
                tx, ty = self.target_pos

                if self.target_action:
                    if (ax == tx and ay - ty == -1 and obs['direction'] == 1) or \
                       (ax == tx and ay - ty == 1  and obs['direction'] == 3) or \
                       (ax - tx == 1  and ay == ty and obs['direction'] == 2) or \
                       (ax - tx == -1 and ay == ty and obs['direction'] == 0):
                        arrived = True
                else:
                    if ax == tx and ay == ty:
                        if self.reward is None:
                            self.reward = float(self._reward())
                        self.mission_done = True

            if arrived and action == self.target_action:
                if self.reward is None:
                    self.reward = float(self._reward())
                self.mission_done = True
            
            if self.target_pos is None and action == self.target_action:
                if self.reward is None:
                    self.reward = float(self._reward())
                self.mission_done = True
            
            if self.agent_pos in self.target_range:
                if self.reward is None:
                    self.reward = float(self._reward())
                self.mission_done = True

        if action == Actions.done:
            if self.mission_done:
                self.mission_done = False
                tmp_rew = self.reward
                self.reward = None
                return obs, tmp_rew, True, truncated, info
            elif not self.manual:
                self.mission_done = False
                self.reward = None
                return obs, float(0), True, truncated, info

        return obs, float(reward), terminated, truncated, info
    
    def _generate_full_map(self):
        objs = []
        self.llm_description = 'The scene contains:\nOnly one room.\n'

        # Until we have generated all the objects
        for objType in self.obj_types:
            for objColor in COLOR_NAMES:

                match objType:
                    case 'key':
                        obj = Key(objColor)
                    case 'ball':
                        obj = Ball(objColor)
                    case 'box':
                        obj = Box(objColor)
                    case 'door':
                        obj = Door(objColor)
                    case _:
                        raise ValueError(f'{objType} object type given. Object type can only be of values key, ball, box and door.')
                
                pos = self.place_obj(obj)
                objs.append((objType, objColor, pos))
                self.llm_description += f'- {objColor} {objType}\n'
        
        # Place a goal square
        pos = self.place_obj(Goal())
        objs.append(('goal', None, pos))
        self.llm_description += '- goal\n'

        # Randomize the agent start position and orientation
        self.place_agent()

        # Setup mission
        target_cmd = self.np_random.choice(self.msn_commands)

        self.llm_description += 'Mission: '

        return objs, target_cmd

    def _generate_gto_map(self):
        objs = []
        self.llm_description = 'The scene contains:\nOnly one room.\n'

        # Until we have generated all the objects
        obj_choice = [(obj, clr) for obj in self.obj_types for clr in COLOR_NAMES]
        assert len(obj_choice) >= self.cfg.num_objects, 'Number of objects to be generated is more than the available objects.'

        for _ in range(self.cfg.num_objects):
            (objType, objColor) = choice(obj_choice)
            obj_choice.remove((objType, objColor))

            match objType:
                case 'key':
                    obj = Key(objColor)
                case 'ball':
                    obj = Ball(objColor)
                case 'box':
                    obj = Box(objColor)
                case 'door':
                    obj = Door(objColor)
                case _:
                    raise ValueError(f'{objType} object type given. Object type can only be of values key, ball, box and door.')
                
            pos = self.place_obj(obj)
            objs.append((objType, objColor, pos))
            self.llm_description += f'- {objColor} {objType}\n'
        
        # Randomize the agent start position and orientation
        self.place_agent()

        # Setup mission
        target_cmd = self.msn_commands[0]

        self.llm_description += 'Mission: '
        
        return objs, target_cmd
    
    def _generate_gtg_map(self):
        objs = []
        self.llm_description = 'The scene contains:\nOnly one room.\n'

        # Until we have generated all the objects
        obj_choice = [(obj, clr) for obj in ["box", "door", "key", "ball"] for clr in COLOR_NAMES]
        assert len(obj_choice) >= self.cfg.num_objects, 'Number of objects to be generated is more than the available objects.'

        for _ in range(self.cfg.num_objects):
            (objType, objColor) = choice(obj_choice)
            obj_choice.remove((objType, objColor))
            
            match objType:
                case 'key':
                    obj = Key(objColor)
                case 'ball':
                    obj = Ball(objColor)
                case 'box':
                    obj = Box(objColor)
                case 'door':
                    obj = Door(objColor)
                case _:
                    raise ValueError(f'{objType} object type given. Object type can only be of values key, ball, box and door.')
            
            pos = self.place_obj(obj)
            objs.append((objType, objColor, pos))
            self.llm_description += f'- {objColor} {objType}\n'
                
        pos = self.place_obj(Goal())
        objs.append(('goal', None, pos))
        self.llm_description += '- goal\n'
        
        # Randomize the agent start position and orientation
        self.place_agent()

        # Setup mission
        target_cmd = self.msn_commands[5]

        self.llm_description += 'Mission: '
        
        return objs, target_cmd
    
    def _generate_open_map(self):
        objs = []
        self.llm_description = 'The scene contains:\nOnly one room.\n'

        # Until we have generated all the objects
        obj_choice = [(obj, clr) for obj in ["box", "door"] for clr in COLOR_NAMES]
        assert len(obj_choice) >= self.cfg.num_objects, 'Number of objects to be generated is more than the available objects.'

        for _ in range(self.cfg.num_objects):
            (objType, objColor) = choice(obj_choice)
            obj_choice.remove((objType, objColor))
            
            if objType == "box":
                obj = Box(objColor)
            else:
                obj = Door(objColor)
            
            pos = self.place_obj(obj)
            objs.append((objType, objColor, pos))
            self.llm_description += f'- {objColor} {objType}\n'

        # Randomize the agent start position and orientation
        self.place_agent()

        # Setup mission
        target_cmd = self.msn_commands[1]

        self.llm_description += 'Mission: '
        
        return objs, target_cmd
    
    def _generate_pkp_map(self):
        objs = []
        self.llm_description = 'The scene contains:\nOnly one room.\n'

        # Until we have generated all the objects
        obj_choice = [(obj, clr) for obj in ["key", "box", "ball"] for clr in COLOR_NAMES]
        assert len(obj_choice) >= self.cfg.num_objects, 'Number of objects to be generated is more than the available objects.'

        for _ in range(self.cfg.num_objects):
            (objType, objColor) = choice(obj_choice)
            obj_choice.remove((objType, objColor))

            if objType == "key":
                obj = Key(objColor)
            elif objType == "box":
                obj = Box(objColor)
            else:
                obj = Ball(objColor)
            
            pos = self.place_obj(obj)
            objs.append((objType, objColor, pos))
            self.llm_description += f'- {objColor} {objType}\n'

        # Randomize the agent start position and orientation
        self.place_agent()

        # Setup mission
        target_cmd = self.msn_commands[2]

        self.llm_description += 'Mission: '
        
        return objs, target_cmd
    
    def _generate_drop_map(self):
        objs = []
        self.llm_description = 'The scene contains:\nOnly one room.\n'

        # Until we have generated all the objects
        obj_choice = [(obj, clr) for obj in self.obj_types for clr in COLOR_NAMES]
        assert len(obj_choice) >= self.cfg.num_objects, 'Number of objects to be generated is more than the available objects.'

        for _ in range(self.cfg.num_objects):
            (objType, objColor) = choice(obj_choice)
            obj_choice.remove((objType, objColor))

            match objType:
                case 'key':
                    obj = Key(objColor)
                case 'ball':
                    obj = Ball(objColor)
                case 'box':
                    obj = Box(objColor)
                case 'door':
                    obj = Door(objColor)
                case _:
                    raise ValueError(f'{objType} object type given. Object type can only be of values key, ball, box and door.')
            
            pos = self.place_obj(obj)
            objs.append((objType, objColor, pos))
            self.llm_description += f'- {objColor} {objType}\n'
        
        # Place a goal square
        pos = self.place_obj(Goal())
        objs.append(('goal', None, pos))

        # Randomize the agent start position and orientation
        self.place_agent()

        # Setup mission
        target_cmd = self.msn_commands[3]

        self.llm_description += 'Mission: '
        
        return objs, target_cmd
    
    def _generate_move_map(self):
        objs = []
        self.llm_description = 'The scene contains:\nOnly one room.\n'

        # Until we have generated all the objects
        obj_choice = [(obj, clr) for obj in self.obj_types for clr in COLOR_NAMES]
        assert len(obj_choice) >= self.cfg.num_objects, 'Number of objects to be generated is more than the available objects.'

        for _ in range(self.cfg.num_objects):
            (objType, objColor) = choice(obj_choice)
            obj_choice.remove((objType, objColor))

            match objType:
                case 'key':
                    obj = Key(objColor)
                case 'ball':
                    obj = Ball(objColor)
                case 'box':
                    obj = Box(objColor)
                case 'door':
                    obj = Door(objColor)
                case _:
                    raise ValueError(f'{objType} object type given. Object type can only be of values key, ball, box and door.')
                
            pos = self.place_obj(obj)
            objs.append((objType, objColor, pos))
            self.llm_description += f'- {objColor} {objType}\n'

        # Randomize the agent start position and orientation
        self.place_agent()

        # Setup mission
        target_cmd = self.msn_commands[4]

        self.llm_description += 'Mission: '
        
        return objs, target_cmd
    
    def _generate_multi_map(self):
        objs = []
        middle = self.cfg.size // 2
        self.llm_description = 'The scene contains:\n'
        
        # Setup mission
        cmd = self.cfg.mission if hasattr(self.cfg, 'mission') and self.cfg.mission is not None else choice([0, 1, 2, 5])
        target_cmd = self.msn_commands[cmd]

        # Generate rooms
        match randint(2, 4):
            case 2:
                objs = self._generate_2_rooms(middle)
            case 3:
                objs = self._generate_3_rooms(middle)
            case 4:
                objs = self._generate_4_rooms(middle)

        self.llm_description += 'Mission: '

        return objs, target_cmd
    
    def _generate_2_rooms(self, middle):
        objs = []
        num_left_objects = self.cfg.num_objects // 2
        num_right_objects = self.cfg.num_objects - num_left_objects

        door_colors = COLOR_NAMES.copy()

        self.llm_description += 'Two rooms. Left and right.\n'

        # Generated all the combinations of objects and colors
        obj_choice = [(obj, clr) for obj in ['key', 'ball', 'box'] for clr in COLOR_NAMES]
        assert len(obj_choice) >= self.cfg.num_objects, 'Number of objects to be generated is more than the available objects.'

        # Build walls
        for i in range(1, self.cfg.size - 1):
            self.grid.set(middle, i, Wall())
        
        # Setup key and / or box
        door_color = choice(COLOR_NAMES)
        door_colors.remove(door_color)
        locked = False if hasattr(self.cfg, 'all_doors_open') and self.cfg.all_doors_open else choice([True, False])
        key_in_box = choice([True, False])

        if locked:
            obj_choice.remove(('key', door_color))
            if key_in_box:
                obj_choice.remove(('box', door_color))

        # Add doors
        j = randint(1, self.cfg.size - 2)
        self.grid.set(middle,
                      j,
                      Door(door_color, is_locked=locked, is_open=choice([True, False]) if hasattr(self.cfg, 'all_doors_open') and self.cfg.all_doors_open else False))
        objs.append(('door', door_color, (middle, j)))
        self.llm_description += f'There is {"a locked" if locked else "an unlocked"} {door_color} door between the rooms\n'
        
        # Place a goal square
        while True:
            goal_pos = self.place_obj(Goal())
        
            if self.next2door(goal_pos):
                self.grid.set(goal_pos[0], goal_pos[1], None)
                continue
        
            break

        goal_left = goal_pos[0] < middle
        objs.append(('goal', None, goal_pos))
        
        # Randomize the agent start position and orientation
        self.place_agent()
        agent_left = self.agent_pos[0] < middle

        self.llm_description += 'Left room contains:\n'
        
        if agent_left:
            self.llm_description += '- robot\n'
            
            # Place a key or box with the key
            if locked:
                num_left_objects -= 1

                while True:
                    key_pos = (randint(1, middle - 1), randint(1, self.cfg.size - 2))
                    if key_pos != goal_pos and key_pos != self.agent_pos and not self.next2door(key_pos):
                        break

                if key_in_box:
                    self.grid.set(key_pos[0], key_pos[1], Box(door_color, Key(door_color)))
                    objs.append(('box', door_color, key_pos))

                    self.llm_description += f'- {door_color} box\n'
                else:
                    self.grid.set(key_pos[0], key_pos[1], Key(door_color))
                    objs.append(('key', door_color, key_pos))
                    
                    self.llm_description += f'- {door_color} key\n'

        if goal_left:
            num_left_objects -= 1
            self.llm_description += '- goal\n'
        
        # Place other objects in left room
        for _ in range(num_left_objects):
            (objType, objColor) = choice(obj_choice)
            obj_choice.remove((objType, objColor))

            match objType:
                case 'key':
                    obj = Key(objColor)
                case 'ball':
                    obj = Ball(objColor)
                case 'box':
                    obj = Box(objColor)
                case _:
                    raise ValueError(f'{objType} object type given. Object type can only be of values key, ball, box and door.')
                
            while True:
                obj_pos = (randint(1, middle - 1), randint(1, self.cfg.size - 2))
                for o in objs:
                    if obj_pos == o[2]:
                        break
                else:
                    if obj_pos != self.agent_pos and not self.next2door(obj_pos):
                        break

            self.put_obj(obj, obj_pos[0], obj_pos[1])
            objs.append((objType, objColor, obj_pos))
            self.llm_description += f'- {objColor} {objType}\n'
        
        # if self.cfg_cmd == 1:
        #     for (objType, objColor) in obj_choice:
        #         if objType == 'box':
        #             obj_choice.remove((objType, objColor))
        #             obj = Box(objColor)
                    
        #             while True:
        #                 obj_pos = (randint(1, middle - 1), randint(1, self.cfg.size - 2))
        #                 for o in objs:
        #                     if obj_pos == o[2]:
        #                         break
        #                 else:
        #                     if obj_pos != self.agent_pos and not self.next2door(obj_pos):
        #                         break

        #             self.put_obj(obj, obj_pos[0], obj_pos[1])
        #             objs.append((objType, objColor, obj_pos))
        #             self.llm_description += f'- {objColor} {objType}\n'

        #             break
        
        # if self.cfg_cmd is not None:
        #     obj = Door(choice(door_colors), is_locked=True, is_open=False)
            
        #     while True:
        #         obj_pos = (randint(1, middle - 1), randint(1, self.cfg.size - 2))
        #         for o in objs:
        #             if obj_pos == o[2]:
        #                 break
        #         else:
        #             if obj_pos != self.agent_pos and not self.next2door(obj_pos):
        #                 break

        #     self.put_obj(obj, obj_pos[0], obj_pos[1])

        self.llm_description += 'Right room contains:\n'

        if not agent_left:
            self.llm_description += '- robot\n'

            # Place a key or box with the key
            if locked:
                num_right_objects -= 1

                while True:
                    key_pos = (randint(middle + 1, self.cfg.size - 2), randint(1, self.cfg.size - 2))
                    if key_pos != goal_pos and key_pos != self.agent_pos and not self.next2door(key_pos):
                        break

                if key_in_box:
                    self.grid.set(key_pos[0], key_pos[1], Box(door_color, Key(door_color)))
                    objs.append(('box', door_color, key_pos))

                    self.llm_description += f'- {door_color} box\n'
                else:
                    self.grid.set(key_pos[0], key_pos[1], Key(door_color))
                    objs.append(('key', door_color, key_pos))
                    
                    self.llm_description += f'- {door_color} key\n'

        if not goal_left:
            num_right_objects -= 1

            self.llm_description += '- goal\n'
        
        # Place other objects in right room
        for _ in range(num_right_objects):
            (objType, objColor) = choice(obj_choice)
            obj_choice.remove((objType, objColor))

            match objType:
                case 'key':
                    obj = Key(objColor)
                case 'ball':
                    obj = Ball(objColor)
                case 'box':
                    obj = Box(objColor)
                case _:
                    raise ValueError(f'{objType} object type given. Object type can only be of values key, ball, box and door.')
            
            while True:
                obj_pos = (randint(middle + 1, self.cfg.size - 2), randint(1, self.cfg.size - 2))
                for o in objs:
                    if obj_pos == o[2]:
                        break
                else:
                    if obj_pos != self.agent_pos and not self.next2door(obj_pos):
                        break

            self.put_obj(obj, obj_pos[0], obj_pos[1])
            objs.append((objType, objColor, obj_pos))
            self.llm_description += f'- {objColor} {objType}\n'
        
        # if self.cfg_cmd == 1:
        #     for (objType, objColor) in obj_choice:
        #         if objType == 'box':
        #             obj_choice.remove((objType, objColor))
        #             obj = Box(objColor)
                    
        #             while True:
        #                 obj_pos = (randint(middle + 1, self.cfg.size - 2), randint(1, self.cfg.size - 2))
        #                 for o in objs:
        #                     if obj_pos == o[2]:
        #                         break
        #                 else:
        #                     if obj_pos != self.agent_pos and not self.next2door(obj_pos):
        #                         break

        #             self.put_obj(obj, obj_pos[0], obj_pos[1])
        #             objs.append((objType, objColor, obj_pos))
        #             self.llm_description += f'- {objColor} {objType}\n'

        #             break
        
        # if self.cfg_cmd is not None:
        #     obj = Door(choice(door_colors), is_locked=True, is_open=False)
            
        #     while True:
        #         obj_pos = (randint(middle + 1, self.cfg.size - 2), randint(1, self.cfg.size - 2))
        #         for o in objs:
        #             if obj_pos == o[2]:
        #                 break
        #         else:
        #             if obj_pos != self.agent_pos and not self.next2door(obj_pos):
        #                 break

        #     self.put_obj(obj, obj_pos[0], obj_pos[1])
        
        return objs
    
    def _generate_3_rooms(self, middle):
        objs = []
        num_left_objects = self.cfg.num_objects // 2
        num_left_upper_objects = num_left_objects // 2
        num_left_lower_objects = num_left_objects - num_left_upper_objects
        num_right_objects = self.cfg.num_objects - num_left_objects

        self.llm_description += 'Three rooms. Upper left, lower left and right.\n'

        # Generated all the combinations of objects and colors
        obj_choice = [(obj, clr) for obj in ['key', 'ball', 'box'] for clr in COLOR_NAMES]
        assert len(obj_choice) >= self.cfg.num_objects, 'Number of objects to be generated is more than the available objects.'

        # Build walls
        for i in range(1, self.cfg.size - 1):
            self.grid.set(middle, i, Wall())
        
        for i in range(1, middle):
            self.grid.set(i, middle, Wall())

        # Setup keys and / or boxes
        door_colors = COLOR_NAMES.copy()

        h_door_color = choice(door_colors)
        door_colors.remove(h_door_color)
        h_locked = False if hasattr(self.cfg, 'all_doors_open') and self.cfg.all_doors_open else choice([True, False])
        h_key_in_box = choice([True, False])

        if h_locked:
            obj_choice.remove(('key', h_door_color))
            if h_key_in_box:
                obj_choice.remove(('box', h_door_color))
        
        vu_door_color = choice(door_colors)
        door_colors.remove(vu_door_color)
        vu_locked = False if hasattr(self.cfg, 'all_doors_open') and self.cfg.all_doors_open else choice([True, False])
        vu_key_in_box = choice([True, False])

        if vu_locked:
            obj_choice.remove(('key', vu_door_color))
            if vu_key_in_box:
                obj_choice.remove(('box', vu_door_color))

        vl_door_color = choice(door_colors)
        door_colors.remove(vl_door_color)
        vl_locked = False if hasattr(self.cfg, 'all_doors_open') and self.cfg.all_doors_open else choice([True, False])
        vl_key_in_box = choice([True, False])

        if vl_locked:
            obj_choice.remove(('key', vl_door_color))
            if vl_key_in_box:
                obj_choice.remove(('box', vl_door_color))

        # Add doors
        h_i = randint(1, middle - 1)
        self.grid.set(h_i,
                      middle,
                      Door(h_door_color, is_locked=h_locked, is_open=choice([True, False]) if hasattr(self.cfg, 'all_doors_open') and self.cfg.all_doors_open else False))
        objs.append(('door', h_door_color, (h_i, middle)))
        self.llm_description += f'There is {"a locked" if h_locked else "an unlocked"} {h_door_color} door between the upper left and lower left rooms.\n'
        
        vu_j = randint(1, middle - 1)
        self.grid.set(middle,
                      vu_j,
                      Door(vu_door_color, is_locked=vu_locked, is_open=choice([True, False]) if hasattr(self.cfg, 'all_doors_open') and self.cfg.all_doors_open else False))
        objs.append(('door', vu_door_color, (middle, vu_j)))
        self.llm_description += f'There is {"a locked" if vu_locked else "an unlocked"} {vu_door_color} door between the upper left and right rooms.\n'
        
        vl_j = randint(middle + 1, self.cfg.size - 2)
        self.grid.set(middle,
                      vl_j,
                      Door(vl_door_color, is_locked=vl_locked, is_open=choice([True, False]) if hasattr(self.cfg, 'all_doors_open') and self.cfg.all_doors_open else False))
        objs.append(('door', vl_door_color, (middle, vl_j)))
        self.llm_description += f'There is {"a locked" if vl_locked else "an unlocked"} {vl_door_color} door between the lower left and right rooms.\n'
        
        # Place a goal square
        while True:
            goal_pos = self.place_obj(Goal())
        
            if self.next2door(goal_pos):
                self.grid.set(goal_pos[0], goal_pos[1], None)
                continue
        
            break

        goal_left = goal_pos[0] < middle
        goal_upper = goal_pos[1] < middle
        objs.append(('goal', None, goal_pos))
        
        # Randomize the agent start position and orientation
        self.place_agent()
        agent_left = self.agent_pos[0] < middle
        agent_upper = self.agent_pos[1] < middle

        self.llm_description += 'Upper left room contains:\n'
        
        if agent_left and agent_upper:
            self.llm_description += '- robot\n'
            
            # Place a key or box with the key
            vu_key_pos = None
            if vu_locked:
                num_left_upper_objects -= 1

                while True:
                    vu_key_pos = (randint(1, middle - 1), randint(1, middle - 1))
                    if vu_key_pos != goal_pos and vu_key_pos != self.agent_pos and not self.next2door(vu_key_pos):
                        break

                if vu_key_in_box:
                    self.grid.set(vu_key_pos[0], vu_key_pos[1], Box(vu_door_color, Key(vu_door_color)))
                    objs.append(('box', vu_door_color, vu_key_pos))

                    self.llm_description += f'- {vu_door_color} box\n'
                else:
                    self.grid.set(vu_key_pos[0], vu_key_pos[1], Key(vu_door_color))
                    objs.append(('key', vu_door_color, vu_key_pos))
                    
                    self.llm_description += f'- {vu_door_color} key\n'
            
            if h_locked:
                num_left_upper_objects -= 1

                while True:
                    h_key_pos = (randint(1, middle - 1), randint(1, middle - 1))
                    if h_key_pos != goal_pos and \
                       h_key_pos != self.agent_pos and \
                       (vu_key_pos is None or h_key_pos != vu_key_pos) and \
                       not self.next2door(h_key_pos):
                        break

                if h_key_in_box:
                    self.grid.set(h_key_pos[0], h_key_pos[1], Box(h_door_color, Key(h_door_color)))
                    objs.append(('box', h_door_color, h_key_pos))

                    self.llm_description += f'- {h_door_color} box\n'
                else:
                    self.grid.set(h_key_pos[0], h_key_pos[1], Key(h_door_color))
                    objs.append(('key', h_door_color, h_key_pos))
                    
                    self.llm_description += f'- {h_door_color} key\n'

        if goal_left and goal_upper:
            num_left_upper_objects -= 1
            self.llm_description += '- goal\n'
        
        # Place other objects in left room
        for _ in range(num_left_upper_objects):
            (objType, objColor) = choice(obj_choice)
            obj_choice.remove((objType, objColor))

            match objType:
                case 'key':
                    obj = Key(objColor)
                case 'ball':
                    obj = Ball(objColor)
                case 'box':
                    obj = Box(objColor)
                case _:
                    raise ValueError(f'{objType} object type given. Object type can only be of values key, ball, box and door.')
                
            while True:
                obj_pos = (randint(1, middle - 1), randint(1, middle - 1))
                for o in objs:
                    if obj_pos == o[2]:
                        break
                else:
                    if obj_pos != self.agent_pos and not self.next2door(obj_pos):
                        break

            self.put_obj(obj, obj_pos[0], obj_pos[1])
            objs.append((objType, objColor, obj_pos))
            self.llm_description += f'- {objColor} {objType}\n'
        
        # if self.cfg_cmd == 1:
        #     for (objType, objColor) in obj_choice:
        #         if objType == 'box':
        #             obj_choice.remove((objType, objColor))
        #             obj = Box(objColor)
                    
        #             while True:
        #                 obj_pos = (randint(1, middle - 1), randint(1, middle - 1))
        #                 for o in objs:
        #                     if obj_pos == o[2]:
        #                         break
        #                 else:
        #                     if obj_pos != self.agent_pos and not self.next2door(obj_pos):
        #                         break

        #             self.put_obj(obj, obj_pos[0], obj_pos[1])
        #             objs.append((objType, objColor, obj_pos))
        #             self.llm_description += f'- {objColor} {objType}\n'

        #             break
        
        # if self.cfg_cmd is not None:
        #     obj = Door(choice(door_colors), is_locked=True, is_open=False)
            
        #     while True:
        #         obj_pos = (randint(1, middle - 1), randint(1, middle - 1))
        #         for o in objs:
        #             if obj_pos == o[2]:
        #                 break
        #         else:
        #             if obj_pos != self.agent_pos and not self.next2door(obj_pos):
        #                 break

        #     self.put_obj(obj, obj_pos[0], obj_pos[1])
        
        self.llm_description += 'Lower left room contains:\n'
        
        if agent_left and not agent_upper:
            self.llm_description += '- robot\n'
            
            # Place a key or box with the key
            vl_key_pos = None
            if vl_locked:
                num_left_lower_objects -= 1

                while True:
                    vl_key_pos = (randint(1, middle - 1), randint(middle + 1, self.cfg.size - 2))
                    if vl_key_pos != goal_pos and vl_key_pos != self.agent_pos and not self.next2door(vl_key_pos):
                        break

                if vl_key_in_box:
                    self.grid.set(vl_key_pos[0], vl_key_pos[1], Box(vl_door_color, Key(vl_door_color)))
                    objs.append(('box', vl_door_color, vl_key_pos))

                    self.llm_description += f'- {vl_door_color} box\n'
                else:
                    self.grid.set(vl_key_pos[0], vl_key_pos[1], Key(vl_door_color))
                    objs.append(('key', vl_door_color, vl_key_pos))
                    
                    self.llm_description += f'- {vl_door_color} key\n'
            
            if h_locked:
                num_left_lower_objects -= 1

                while True:
                    h_key_pos = (randint(1, middle - 1), randint(middle + 1, self.cfg.size - 2))
                    if h_key_pos != goal_pos and \
                       h_key_pos != self.agent_pos and \
                       (vl_key_pos is None or h_key_pos != vl_key_pos) and \
                       not self.next2door(h_key_pos):
                        break

                if h_key_in_box:
                    self.grid.set(h_key_pos[0], h_key_pos[1], Box(h_door_color, Key(h_door_color)))
                    objs.append(('box', h_door_color, h_key_pos))

                    self.llm_description += f'- {h_door_color} box\n'
                else:
                    self.grid.set(h_key_pos[0], h_key_pos[1], Key(h_door_color))
                    objs.append(('key', h_door_color, h_key_pos))
                    
                    self.llm_description += f'- {h_door_color} key\n'

        if goal_left and not goal_upper:
            num_left_lower_objects -= 1
            self.llm_description += '- goal\n'
        
        # Place other objects in left room
        for _ in range(num_left_upper_objects):
            (objType, objColor) = choice(obj_choice)
            obj_choice.remove((objType, objColor))

            match objType:
                case 'key':
                    obj = Key(objColor)
                case 'ball':
                    obj = Ball(objColor)
                case 'box':
                    obj = Box(objColor)
                case _:
                    raise ValueError(f'{objType} object type given. Object type can only be of values key, ball, box and door.')
                
            while True:
                obj_pos = (randint(1, middle - 1), randint(middle + 1, self.cfg.size - 2))
                for o in objs:
                    if obj_pos == o[2]:
                        break
                else:
                    if obj_pos != self.agent_pos and not self.next2door(obj_pos):
                        break

            self.put_obj(obj, obj_pos[0], obj_pos[1])
            objs.append((objType, objColor, obj_pos))
            self.llm_description += f'- {objColor} {objType}\n'
        
        # if self.cfg_cmd == 1:
        #     for (objType, objColor) in obj_choice:
        #         if objType == 'box':
        #             obj_choice.remove((objType, objColor))
        #             obj = Box(objColor)
                    
        #             while True:
        #                 obj_pos = (randint(1, middle - 1), randint(middle + 1, self.cfg.size - 2))
        #                 for o in objs:
        #                     if obj_pos == o[2]:
        #                         break
        #                 else:
        #                     if obj_pos != self.agent_pos and not self.next2door(obj_pos):
        #                         break

        #             self.put_obj(obj, obj_pos[0], obj_pos[1])
        #             objs.append((objType, objColor, obj_pos))
        #             self.llm_description += f'- {objColor} {objType}\n'

        #             break
        
        # if self.cfg_cmd is not None:
        #     obj = Door(choice(door_colors), is_locked=True, is_open=False)
            
        #     while True:
        #         obj_pos = (randint(1, middle - 1), randint(middle + 1, self.cfg.size - 2))
        #         for o in objs:
        #             if obj_pos == o[2]:
        #                 break
        #         else:
        #             if obj_pos != self.agent_pos and not self.next2door(obj_pos):
        #                 break

        #     self.put_obj(obj, obj_pos[0], obj_pos[1])
        
        self.llm_description += 'Right room contains:\n'

        if not agent_left:
            self.llm_description += '- robot\n'

            # Place a key or box with the key
            vl_key_pos = None
            if vl_locked:
                num_right_objects -= 1

                while True:
                    vl_key_pos = (randint(middle + 1, self.cfg.size - 2), randint(1, self.cfg.size - 2))
                    if vl_key_pos != goal_pos and vl_key_pos != self.agent_pos and not self.next2door(vl_key_pos):
                        break

                if vl_key_in_box:
                    self.grid.set(vl_key_pos[0], vl_key_pos[1], Box(vl_door_color, Key(vl_door_color)))
                    objs.append(('box', vl_door_color, vl_key_pos))

                    self.llm_description += f'- {vl_door_color} box\n'
                else:
                    self.grid.set(vl_key_pos[0], vl_key_pos[1], Key(vl_door_color))
                    objs.append(('key', vl_door_color, vl_key_pos))
                    
                    self.llm_description += f'- {vl_door_color} key\n'
            
            if vu_locked:
                num_right_objects -= 1

                while True:
                    vu_key_pos = (randint(middle + 1, self.cfg.size - 2), randint(1, self.cfg.size - 2))
                    if vu_key_pos != goal_pos and \
                       vu_key_pos != self.agent_pos and \
                       (vl_key_pos is None or vu_key_pos != vl_key_pos) and \
                       not self.next2door(vu_key_pos):
                        break

                if vu_key_in_box:
                    self.grid.set(vu_key_pos[0], vu_key_pos[1], Box(vu_door_color, Key(vu_door_color)))
                    objs.append(('box', vu_door_color, vu_key_pos))

                    self.llm_description += f'- {vu_door_color} box\n'
                else:
                    self.grid.set(vu_key_pos[0], vu_key_pos[1], Key(vu_door_color))
                    objs.append(('key', vu_door_color, vu_key_pos))
                    
                    self.llm_description += f'- {vu_door_color} key\n'

        if not goal_left:
            num_right_objects -= 1

            self.llm_description += '- goal\n'
        
        # Place other objects in right room
        for _ in range(num_right_objects):
            (objType, objColor) = choice(obj_choice)
            obj_choice.remove((objType, objColor))

            match objType:
                case 'key':
                    obj = Key(objColor)
                case 'ball':
                    obj = Ball(objColor)
                case 'box':
                    obj = Box(objColor)
                case _:
                    raise ValueError(f'{objType} object type given. Object type can only be of values key, ball, box and door.')
            
            while True:
                obj_pos = (randint(middle + 1, self.cfg.size - 2), randint(1, self.cfg.size - 2))
                for o in objs:
                    if obj_pos == o[2]:
                        break
                else:
                    if obj_pos != self.agent_pos and not self.next2door(obj_pos):
                        break

            self.put_obj(obj, obj_pos[0], obj_pos[1])
            objs.append((objType, objColor, obj_pos))
            self.llm_description += f'- {objColor} {objType}\n'
        
        # if self.cfg_cmd == 1:
        #     for (objType, objColor) in obj_choice:
        #         if objType == 'box':
        #             obj_choice.remove((objType, objColor))
        #             obj = Box(objColor)
                    
        #             while True:
        #                 obj_pos = (randint(middle + 1, self.cfg.size - 2), randint(1, self.cfg.size - 2))
        #                 for o in objs:
        #                     if obj_pos == o[2]:
        #                         break
        #                 else:
        #                     if obj_pos != self.agent_pos and not self.next2door(obj_pos):
        #                         break

        #             self.put_obj(obj, obj_pos[0], obj_pos[1])
        #             objs.append((objType, objColor, obj_pos))
        #             self.llm_description += f'- {objColor} {objType}\n'

        #             break

        # if self.cfg_cmd is not None:
        #     obj = Door(choice(door_colors), is_locked=True, is_open=False)
            
        #     while True:
        #         obj_pos = (randint(middle + 1, self.cfg.size - 2), randint(1, self.cfg.size - 2))
        #         for o in objs:
        #             if obj_pos == o[2]:
        #                 break
        #         else:
        #             if obj_pos != self.agent_pos and not self.next2door(obj_pos):
        #                 break

        #     self.put_obj(obj, obj_pos[0], obj_pos[1])
        
        return objs

    def _generate_4_rooms(self, middle):
        objs = []
        num_left_objects = self.cfg.num_objects // 2
        num_left_upper_objects = num_left_objects // 2
        num_left_lower_objects = num_left_objects - num_left_upper_objects
        num_right_objects = self.cfg.num_objects - num_left_objects
        num_right_upper_objects = num_right_objects // 2
        num_right_lower_objects = num_right_objects - num_right_upper_objects

        self.llm_description += 'Four rooms. Upper left, lower left, upper right and lower right.\n'

        # Generated all the combinations of objects and colors
        obj_choice = [(obj, clr) for obj in ['key', 'ball', 'box'] for clr in COLOR_NAMES]
        assert len(obj_choice) >= self.cfg.num_objects, 'Number of objects to be generated is more than the available objects.'

        # Build walls
        for i in range(1, self.cfg.size - 1):
            self.grid.set(middle, i, Wall())

        for i in range(1, self.cfg.size - 1):
            self.grid.set(i, middle, Wall())

        # Setup keys and / or boxes
        door_colors = COLOR_NAMES.copy()

        hl_door_color = choice(door_colors)
        door_colors.remove(hl_door_color)
        hl_locked = False if hasattr(self.cfg, 'all_doors_open') and self.cfg.all_doors_open else choice([True, False])
        hl_key_in_box = choice([True, False])

        if hl_locked:
            obj_choice.remove(('key', hl_door_color))
            if hl_key_in_box:
                obj_choice.remove(('box', hl_door_color))
        
        hr_door_color = choice(door_colors)
        door_colors.remove(hr_door_color)
        hr_locked = False if hasattr(self.cfg, 'all_doors_open') and self.cfg.all_doors_open else choice([True, False])
        hr_key_in_box = choice([True, False])

        if hr_locked:
            obj_choice.remove(('key', hr_door_color))
            if hr_key_in_box:
                obj_choice.remove(('box', hr_door_color))
        
        vu_door_color = choice(door_colors)
        door_colors.remove(vu_door_color)
        vu_locked = False if hasattr(self.cfg, 'all_doors_open') and self.cfg.all_doors_open else choice([True, False])
        vu_key_in_box = choice([True, False])

        if vu_locked:
            obj_choice.remove(('key', vu_door_color))
            if vu_key_in_box:
                obj_choice.remove(('box', vu_door_color))
        
        vl_door_color = choice(door_colors)
        door_colors.remove(vl_door_color)
        vl_locked = False if hasattr(self.cfg, 'all_doors_open') and self.cfg.all_doors_open else choice([True, False])
        vl_key_in_box = choice([True, False])

        if vl_locked:
            obj_choice.remove(('key', vl_door_color))
            if vl_key_in_box:
                obj_choice.remove(('box', vl_door_color))

        # Add doors
        hl_i = randint(1, middle - 1)
        self.grid.set(hl_i,
                      middle,
                      Door(hl_door_color, is_locked=hl_locked, is_open=choice([True, False]) if hasattr(self.cfg, 'all_doors_open') and self.cfg.all_doors_open else False))
        objs.append(('door', hl_door_color, (hl_i, middle)))
        self.llm_description += f'There is {"a locked" if hl_locked else "an unlocked"} {hl_door_color} door between the upper left and lower left rooms.\n'
        
        hr_i = randint(middle + 1, self.cfg.size - 2)
        self.grid.set(hr_i,
                      middle,
                      Door(hr_door_color, is_locked=hr_locked, is_open=choice([True, False]) if hasattr(self.cfg, 'all_doors_open') and self.cfg.all_doors_open else False))
        objs.append(('door', hr_door_color, (hr_i, middle)))
        self.llm_description += f'There is {"a locked" if hr_locked else "an unlocked"} {hr_door_color} door between the upper right and lower right rooms.\n'
        
        vu_j = randint(1, middle - 1)
        self.grid.set(middle,
                      vu_j,
                      Door(vu_door_color, is_locked=vu_locked, is_open=choice([True, False]) if hasattr(self.cfg, 'all_doors_open') and self.cfg.all_doors_open else False))
        objs.append(('door', vu_door_color, (middle, vu_j)))
        self.llm_description += f'There is {"a locked" if vu_locked else "an unlocked"} {vu_door_color} door between the upper left and upper right rooms.\n'
        
        vl_j = randint(middle + 1, self.cfg.size - 2)
        self.grid.set(middle,
                      vl_j,
                      Door(vl_door_color, is_locked=vl_locked, is_open=choice([True, False]) if hasattr(self.cfg, 'all_doors_open') and self.cfg.all_doors_open else False))
        objs.append(('door', vl_door_color, (middle, vl_j)))
        self.llm_description += f'There is {"a locked" if vl_locked else "an unlocked"} {vl_door_color} door between the lower left and lower right rooms.\n'
        
        # Place a goal square
        while True:
            goal_pos = self.place_obj(Goal())
        
            if self.next2door(goal_pos):
                self.grid.set(goal_pos[0], goal_pos[1], None)
                continue
        
            break
        
        goal_left = goal_pos[0] < middle
        goal_upper = goal_pos[1] < middle
        objs.append(('goal', None, goal_pos))
        
        # Randomize the agent start position and orientation
        self.place_agent()
        agent_left = self.agent_pos[0] < middle
        agent_upper = self.agent_pos[1] < middle

        self.llm_description += 'Upper left room contains:\n'
        
        if agent_left and agent_upper:
            self.llm_description += '- robot\n'
            
            # Place a key or box with the key
            vu_key_pos = None
            if vu_locked:
                num_left_upper_objects -= 1

                while True:
                    vu_key_pos = (randint(1, middle - 1), randint(1, middle - 1))
                    if vu_key_pos != goal_pos and vu_key_pos != self.agent_pos and not self.next2door(vu_key_pos):
                        break

                if vu_key_in_box:
                    self.grid.set(vu_key_pos[0], vu_key_pos[1], Box(vu_door_color, Key(vu_door_color)))
                    objs.append(('box', vu_door_color, vu_key_pos))

                    self.llm_description += f'- {vu_door_color} box\n'
                else:
                    self.grid.set(vu_key_pos[0], vu_key_pos[1], Key(vu_door_color))
                    objs.append(('key', vu_door_color, vu_key_pos))
                    
                    self.llm_description += f'- {vu_door_color} key\n'
            
            if hl_locked:
                num_left_upper_objects -= 1

                while True:
                    hl_key_pos = (randint(1, middle - 1), randint(1, middle - 1))
                    if hl_key_pos != goal_pos and \
                       hl_key_pos != self.agent_pos and \
                       (vu_key_pos is None or hl_key_pos != vu_key_pos) and \
                       not self.next2door(hl_key_pos):
                        break

                if hl_key_in_box:
                    self.grid.set(hl_key_pos[0], hl_key_pos[1], Box(hl_door_color, Key(hl_door_color)))
                    objs.append(('box', hl_door_color, hl_key_pos))

                    self.llm_description += f'- {hl_door_color} box\n'
                else:
                    self.grid.set(hl_key_pos[0], hl_key_pos[1], Key(hl_door_color))
                    objs.append(('key', hl_door_color, hl_key_pos))
                    
                    self.llm_description += f'- {hl_door_color} key\n'
        
        elif agent_left and not agent_upper:
            if vu_locked:
                num_left_upper_objects -= 1

                while True:
                    vu_key_pos = (randint(1, middle - 1), randint(1, middle - 1))
                    if vu_key_pos != goal_pos and not self.next2door(vu_key_pos):
                        break

                if vu_key_in_box:
                    self.grid.set(vu_key_pos[0], vu_key_pos[1], Box(vu_door_color, Key(vu_door_color)))
                    objs.append(('box', vu_door_color, vu_key_pos))

                    self.llm_description += f'- {vu_door_color} box\n'
                else:
                    self.grid.set(vu_key_pos[0], vu_key_pos[1], Key(vu_door_color))
                    objs.append(('key', vu_door_color, vu_key_pos))
                    
                    self.llm_description += f'- {vu_door_color} key\n'
        
        elif not agent_left and agent_upper:
            if hl_locked:
                num_left_upper_objects -= 1

                while True:
                    hl_key_pos = (randint(1, middle - 1), randint(1, middle - 1))
                    if hl_key_pos != goal_pos and not self.next2door(hl_key_pos):
                        break

                if hl_key_in_box:
                    self.grid.set(hl_key_pos[0], hl_key_pos[1], Box(hl_door_color, Key(hl_door_color)))
                    objs.append(('box', hl_door_color, hl_key_pos))

                    self.llm_description += f'- {hl_door_color} box\n'
                else:
                    self.grid.set(hl_key_pos[0], hl_key_pos[1], Key(hl_door_color))
                    objs.append(('key', hl_door_color, hl_key_pos))
                    
                    self.llm_description += f'- {hl_door_color} key\n'

        if goal_left and goal_upper:
            num_left_upper_objects -= 1
            self.llm_description += '- goal\n'
        
        # Place other objects in left room
        for _ in range(num_left_upper_objects):
            (objType, objColor) = choice(obj_choice)
            obj_choice.remove((objType, objColor))

            match objType:
                case 'key':
                    obj = Key(objColor)
                case 'ball':
                    obj = Ball(objColor)
                case 'box':
                    obj = Box(objColor)
                case _:
                    raise ValueError(f'{objType} object type given. Object type can only be of values key, ball, box and door.')
                
            while True:
                obj_pos = (randint(1, middle - 1), randint(1, middle - 1))
                for o in objs:
                    if obj_pos == o[2]:
                        break
                else:
                    if obj_pos != self.agent_pos and not self.next2door(obj_pos):
                        break

            self.put_obj(obj, obj_pos[0], obj_pos[1])
            objs.append((objType, objColor, obj_pos))
            self.llm_description += f'- {objColor} {objType}\n'
        
        # if self.cfg_cmd == 1:
        #     for (objType, objColor) in obj_choice:
        #         if objType == 'box':
        #             obj_choice.remove((objType, objColor))
        #             obj = Box(objColor)
                    
        #             while True:
        #                 obj_pos = (randint(1, middle - 1), randint(1, middle - 1))
        #                 for o in objs:
        #                     if obj_pos == o[2]:
        #                         break
        #                 else:
        #                     if obj_pos != self.agent_pos and not self.next2door(obj_pos):
        #                         break

        #             self.put_obj(obj, obj_pos[0], obj_pos[1])
        #             objs.append((objType, objColor, obj_pos))
        #             self.llm_description += f'- {objColor} {objType}\n'

        #             break
        
        # if self.cfg_cmd is not None:
        #     obj = Door(choice(door_colors), is_locked=True, is_open=False)
            
        #     while True:
        #         obj_pos = (randint(1, middle - 1), randint(1, middle - 1))
        #         for o in objs:
        #             if obj_pos == o[2]:
        #                 break
        #         else:
        #             if obj_pos != self.agent_pos and not self.next2door(obj_pos):
        #                 break

        #     self.put_obj(obj, obj_pos[0], obj_pos[1])
        
        self.llm_description += 'Lower left room contains:\n'
        
        if agent_left and not agent_upper:
            self.llm_description += '- robot\n'
            
            # Place a key or box with the key
            vl_key_pos = None
            if vl_locked:
                num_left_lower_objects -= 1

                while True:
                    vl_key_pos = (randint(1, middle - 1), randint(middle + 1, self.cfg.size - 2))
                    if vl_key_pos != goal_pos and vl_key_pos != self.agent_pos and not self.next2door(vl_key_pos):
                        break

                if vl_key_in_box:
                    self.grid.set(vl_key_pos[0], vl_key_pos[1], Box(vl_door_color, Key(vl_door_color)))
                    objs.append(('box', vl_door_color, vl_key_pos))

                    self.llm_description += f'- {vl_door_color} box\n'
                else:
                    self.grid.set(vl_key_pos[0], vl_key_pos[1], Key(vl_door_color))
                    objs.append(('key', vl_door_color, vl_key_pos))
                    
                    self.llm_description += f'- {vl_door_color} key\n'
            
            if hl_locked:
                num_left_lower_objects -= 1

                while True:
                    hl_key_pos = (randint(1, middle - 1), randint(middle + 1, self.cfg.size - 2))
                    if hl_key_pos != goal_pos and \
                       hl_key_pos != self.agent_pos and \
                       (vl_key_pos is None or hl_key_pos != vl_key_pos) and \
                       not self.next2door(hl_key_pos):
                        break

                if hl_key_in_box:
                    self.grid.set(hl_key_pos[0], hl_key_pos[1], Box(hl_door_color, Key(hl_door_color)))
                    objs.append(('box', hl_door_color, hl_key_pos))

                    self.llm_description += f'- {hl_door_color} box\n'
                else:
                    self.grid.set(hl_key_pos[0], hl_key_pos[1], Key(hl_door_color))
                    objs.append(('key', hl_door_color, hl_key_pos))
                    
                    self.llm_description += f'- {hl_door_color} key\n'
        
        elif not agent_left and not agent_upper:
            if hl_locked:
                num_left_lower_objects -= 1

                while True:
                    hl_key_pos = (randint(1, middle - 1), randint(middle + 1, self.cfg.size - 2))
                    if hl_key_pos != goal_pos and not self.next2door(hl_key_pos):
                        break

                if hl_key_in_box:
                    self.grid.set(hl_key_pos[0], hl_key_pos[1], Box(hl_door_color, Key(hl_door_color)))
                    objs.append(('box', hl_door_color, hl_key_pos))

                    self.llm_description += f'- {hl_door_color} box\n'
                else:
                    self.grid.set(hl_key_pos[0], hl_key_pos[1], Key(hl_door_color))
                    objs.append(('key', hl_door_color, hl_key_pos))
                    
                    self.llm_description += f'- {hl_door_color} key\n'
        
        elif agent_left and agent_upper:
            if vl_locked:
                num_left_lower_objects -= 1

                while True:
                    vl_key_pos = (randint(1, middle - 1), randint(middle + 1, self.cfg.size - 2))
                    if vl_key_pos != goal_pos and not self.next2door(vl_key_pos):
                        break

                if vl_key_in_box:
                    self.grid.set(vl_key_pos[0], vl_key_pos[1], Box(vl_door_color, Key(vl_door_color)))
                    objs.append(('box', vl_door_color, vl_key_pos))

                    self.llm_description += f'- {vl_door_color} box\n'
                else:
                    self.grid.set(vl_key_pos[0], vl_key_pos[1], Key(vl_door_color))
                    objs.append(('key', vl_door_color, vl_key_pos))
                    
                    self.llm_description += f'- {vl_door_color} key\n'

        if goal_left and not goal_upper:
            num_left_lower_objects -= 1
            self.llm_description += '- goal\n'
        
        # Place other objects in left room
        for _ in range(num_left_upper_objects):
            (objType, objColor) = choice(obj_choice)
            obj_choice.remove((objType, objColor))

            match objType:
                case 'key':
                    obj = Key(objColor)
                case 'ball':
                    obj = Ball(objColor)
                case 'box':
                    obj = Box(objColor)
                case _:
                    raise ValueError(f'{objType} object type given. Object type can only be of values key, ball, box and door.')
                
            while True:
                obj_pos = (randint(1, middle - 1), randint(middle + 1, self.cfg.size - 2))
                for o in objs:
                    if obj_pos == o[2]:
                        break
                else:
                    if obj_pos != self.agent_pos and not self.next2door(obj_pos):
                        break

            self.put_obj(obj, obj_pos[0], obj_pos[1])
            objs.append((objType, objColor, obj_pos))
            self.llm_description += f'- {objColor} {objType}\n'
        
        # if self.cfg_cmd == 1:
        #     for (objType, objColor) in obj_choice:
        #         if objType == 'box':
        #             obj_choice.remove((objType, objColor))
        #             obj = Box(objColor)
                    
        #             while True:
        #                 obj_pos = (randint(1, middle - 1), randint(middle + 1, self.cfg.size - 2))
        #                 for o in objs:
        #                     if obj_pos == o[2]:
        #                         break
        #                 else:
        #                     if obj_pos != self.agent_pos and not self.next2door(obj_pos):
        #                         break

        #             self.put_obj(obj, obj_pos[0], obj_pos[1])
        #             objs.append((objType, objColor, obj_pos))
        #             self.llm_description += f'- {objColor} {objType}\n'

        #             break
        
        # if self.cfg_cmd is not None:
        #     obj = Door(choice(door_colors), is_locked=True, is_open=False)
            
        #     while True:
        #         obj_pos = (randint(1, middle - 1), randint(middle + 1, self.cfg.size - 2))
        #         for o in objs:
        #             if obj_pos == o[2]:
        #                 break
        #         else:
        #             if obj_pos != self.agent_pos and not self.next2door(obj_pos):
        #                 break

        #     self.put_obj(obj, obj_pos[0], obj_pos[1])
        
        self.llm_description += 'Upper right room contains:\n'

        if not agent_left and agent_upper:
            self.llm_description += '- robot\n'

            # Place a key or box with the key
            vu_key_pos = None
            if vu_locked:
                num_right_upper_objects -= 1

                while True:
                    vu_key_pos = (randint(middle + 1, self.cfg.size - 2), randint(1, middle - 1))
                    if vu_key_pos != goal_pos and vu_key_pos != self.agent_pos and not self.next2door(vu_key_pos):
                        break

                if vu_key_in_box:
                    self.grid.set(vu_key_pos[0], vu_key_pos[1], Box(vu_door_color, Key(vu_door_color)))
                    objs.append(('box', vu_door_color, vu_key_pos))

                    self.llm_description += f'- {vu_door_color} box\n'
                else:
                    self.grid.set(vu_key_pos[0], vu_key_pos[1], Key(vu_door_color))
                    objs.append(('key', vu_door_color, vu_key_pos))
                    
                    self.llm_description += f'- {vu_door_color} key\n'
            
            if hr_locked:
                num_right_upper_objects -= 1

                while True:
                    hr_key_pos = (randint(middle + 1, self.cfg.size - 2), randint(1, middle - 1))
                    if hr_key_pos != goal_pos and \
                       hr_key_pos != self.agent_pos and \
                       (vu_key_pos is None or hr_key_pos != vu_key_pos) and \
                       not self.next2door(hr_key_pos):
                        break

                if hr_key_in_box:
                    self.grid.set(hr_key_pos[0], hr_key_pos[1], Box(hr_door_color, Key(hr_door_color)))
                    objs.append(('box', hr_door_color, hr_key_pos))

                    self.llm_description += f'- {hr_door_color} box\n'
                else:
                    self.grid.set(hr_key_pos[0], hr_key_pos[1], Key(hr_door_color))
                    objs.append(('key', hr_door_color, hr_key_pos))
                    
                    self.llm_description += f'- {hr_door_color} key\n'
        
        elif not agent_left and not agent_upper:
            if vu_locked:
                num_right_upper_objects -= 1

                while True:
                    vu_key_pos = (randint(middle + 1, self.cfg.size - 2), randint(1, middle - 1))
                    if vu_key_pos != goal_pos and not self.next2door(vu_key_pos):
                        break

                if vu_key_in_box:
                    self.grid.set(vu_key_pos[0], vu_key_pos[1], Box(vu_door_color, Key(vu_door_color)))
                    objs.append(('box', vu_door_color, vu_key_pos))

                    self.llm_description += f'- {vu_door_color} box\n'
                else:
                    self.grid.set(vu_key_pos[0], vu_key_pos[1], Key(vu_door_color))
                    objs.append(('key', vu_door_color, vu_key_pos))
                    
                    self.llm_description += f'- {vu_door_color} key\n'

        elif agent_left and agent_upper:
            if hr_locked:
                num_right_upper_objects -= 1

                while True:
                    hr_key_pos = (randint(middle + 1, self.cfg.size - 2), randint(1, middle - 1))
                    if hr_key_pos != goal_pos and not self.next2door(hr_key_pos):
                        break

                if hr_key_in_box:
                    self.grid.set(hr_key_pos[0], hr_key_pos[1], Box(hr_door_color, Key(hr_door_color)))
                    objs.append(('box', hr_door_color, hr_key_pos))

                    self.llm_description += f'- {hr_door_color} box\n'
                else:
                    self.grid.set(hr_key_pos[0], hr_key_pos[1], Key(hr_door_color))
                    objs.append(('key', hr_door_color, hr_key_pos))
                    
                    self.llm_description += f'- {hr_door_color} key\n'

        if not goal_left and goal_upper:
            num_right_upper_objects -= 1

            self.llm_description += '- goal\n'
        
        # Place other objects in right room
        for _ in range(num_right_upper_objects):
            (objType, objColor) = choice(obj_choice)
            obj_choice.remove((objType, objColor))

            match objType:
                case 'key':
                    obj = Key(objColor)
                case 'ball':
                    obj = Ball(objColor)
                case 'box':
                    obj = Box(objColor)
                case _:
                    raise ValueError(f'{objType} object type given. Object type can only be of values key, ball, box and door.')
            
            while True:
                obj_pos = (randint(middle + 1, self.cfg.size - 2), randint(1, middle - 1))
                for o in objs:
                    if obj_pos == o[2]:
                        break
                else:
                    if obj_pos != self.agent_pos and not self.next2door(obj_pos):
                        break

            self.put_obj(obj, obj_pos[0], obj_pos[1])
            objs.append((objType, objColor, obj_pos))
            self.llm_description += f'- {objColor} {objType}\n'
        
        # if self.cfg_cmd == 1:
        #     for (objType, objColor) in obj_choice:
        #         if objType == 'box':
        #             obj_choice.remove((objType, objColor))
        #             obj = Box(objColor)
                    
        #             while True:
        #                 obj_pos = (randint(middle + 1, self.cfg.size - 2), randint(1, middle - 1))
        #                 for o in objs:
        #                     if obj_pos == o[2]:
        #                         break
        #                 else:
        #                     if obj_pos != self.agent_pos and not self.next2door(obj_pos):
        #                         break

        #             self.put_obj(obj, obj_pos[0], obj_pos[1])
        #             objs.append((objType, objColor, obj_pos))
        #             self.llm_description += f'- {objColor} {objType}\n'

        #             break
        
        # if self.cfg_cmd is not None:
        #     obj = Door(choice(door_colors), is_locked=True, is_open=False)
            
        #     while True:
        #         obj_pos = (randint(middle + 1, self.cfg.size - 2), randint(1, middle - 1))
        #         for o in objs:
        #             if obj_pos == o[2]:
        #                 break
        #         else:
        #             if obj_pos != self.agent_pos and not self.next2door(obj_pos):
        #                 break

        #     self.put_obj(obj, obj_pos[0], obj_pos[1])
        
        self.llm_description += 'Lower right room contains:\n'

        if not agent_left and not agent_upper:
            self.llm_description += '- robot\n'

            # Place a key or box with the key
            vl_key_pos = None
            if vl_locked:
                num_right_lower_objects -= 1

                while True:
                    vl_key_pos = (randint(middle + 1, self.cfg.size - 2), randint(middle + 1, self.cfg.size - 2))
                    if vl_key_pos != goal_pos and vl_key_pos != self.agent_pos and not self.next2door(vl_key_pos):
                        break

                if vl_key_in_box:
                    self.grid.set(vl_key_pos[0], vl_key_pos[1], Box(vl_door_color, Key(vl_door_color)))
                    objs.append(('box', vl_door_color, vl_key_pos))

                    self.llm_description += f'- {vl_door_color} box\n'
                else:
                    self.grid.set(vl_key_pos[0], vl_key_pos[1], Key(vl_door_color))
                    objs.append(('key', vl_door_color, vl_key_pos))
                    
                    self.llm_description += f'- {vl_door_color} key\n'
            
            if hr_locked:
                num_right_lower_objects -= 1

                while True:
                    hr_key_pos = (randint(middle + 1, self.cfg.size - 2), randint(middle + 1, self.cfg.size - 2))
                    if hr_key_pos != goal_pos and \
                       hr_key_pos != self.agent_pos and \
                       (vl_key_pos is None or hr_key_pos != vl_key_pos) and \
                       not self.next2door(hr_key_pos):
                        break

                if hr_key_in_box:
                    self.grid.set(hr_key_pos[0], hr_key_pos[1], Box(hr_door_color, Key(hr_door_color)))
                    objs.append(('box', hr_door_color, hr_key_pos))

                    self.llm_description += f'- {hr_door_color} box\n'
                else:
                    self.grid.set(hr_key_pos[0], hr_key_pos[1], Key(hr_door_color))
                    objs.append(('key', hr_door_color, hr_key_pos))
                    
                    self.llm_description += f'- {hr_door_color} key\n'

        elif agent_left and not agent_upper:
            if hr_locked:
                num_right_lower_objects -= 1

                while True:
                    hr_key_pos = (randint(middle + 1, self.cfg.size - 2), randint(middle + 1, self.cfg.size - 2))
                    if hr_key_pos != goal_pos and not self.next2door(hr_key_pos):
                        break

                if hr_key_in_box:
                    self.grid.set(hr_key_pos[0], hr_key_pos[1], Box(hr_door_color, Key(hr_door_color)))
                    objs.append(('box', hr_door_color, hr_key_pos))

                    self.llm_description += f'- {hr_door_color} box\n'
                else:
                    self.grid.set(hr_key_pos[0], hr_key_pos[1], Key(hr_door_color))
                    objs.append(('key', hr_door_color, hr_key_pos))
                    
                    self.llm_description += f'- {hr_door_color} key\n'

        elif not agent_left and agent_upper:
            if vl_locked:
                num_right_lower_objects -= 1

                while True:
                    vl_key_pos = (randint(middle + 1, self.cfg.size - 2), randint(middle + 1, self.cfg.size - 2))
                    if vl_key_pos != goal_pos and not self.next2door(vl_key_pos):
                        break

                if vl_key_in_box:
                    self.grid.set(vl_key_pos[0], vl_key_pos[1], Box(vl_door_color, Key(vl_door_color)))
                    objs.append(('box', vl_door_color, vl_key_pos))

                    self.llm_description += f'- {vl_door_color} box\n'
                else:
                    self.grid.set(vl_key_pos[0], vl_key_pos[1], Key(vl_door_color))
                    objs.append(('key', vl_door_color, vl_key_pos))
                    
                    self.llm_description += f'- {vl_door_color} key\n'

        if not goal_left and not goal_upper:
            num_right_lower_objects -= 1

            self.llm_description += '- goal\n'
        
        # Place other objects in right room
        for _ in range(num_right_lower_objects):
            (objType, objColor) = choice(obj_choice)
            obj_choice.remove((objType, objColor))

            match objType:
                case 'key':
                    obj = Key(objColor)
                case 'ball':
                    obj = Ball(objColor)
                case 'box':
                    obj = Box(objColor)
                case _:
                    raise ValueError(f'{objType} object type given. Object type can only be of values key, ball, box and door.')
            
            while True:
                obj_pos = (randint(middle + 1, self.cfg.size - 2), randint(middle + 1, self.cfg.size - 2))
                for o in objs:
                    if obj_pos == o[2]:
                        break
                else:
                    if obj_pos != self.agent_pos and not self.next2door(obj_pos):
                        break

            self.put_obj(obj, obj_pos[0], obj_pos[1])
            objs.append((objType, objColor, obj_pos))
            self.llm_description += f'- {objColor} {objType}\n'
        
        # if self.cfg_cmd == 1:
        #     for (objType, objColor) in obj_choice:
        #         if objType == 'box':
        #             obj_choice.remove((objType, objColor))
        #             obj = Box(objColor)
                    
        #             while True:
        #                 obj_pos = (randint(middle + 1, self.cfg.size - 2), randint(middle + 1, self.cfg.size - 2))
        #                 for o in objs:
        #                     if obj_pos == o[2]:
        #                         break
        #                 else:
        #                     if obj_pos != self.agent_pos and not self.next2door(obj_pos):
        #                         break

        #             self.put_obj(obj, obj_pos[0], obj_pos[1])
        #             objs.append((objType, objColor, obj_pos))
        #             self.llm_description += f'- {objColor} {objType}\n'

        #             break
        
        # if self.cfg_cmd is not None:
        #     obj = Door(choice(door_colors), is_locked=True, is_open=False)
            
        #     while True:
        #         obj_pos = (randint(middle + 1, self.cfg.size - 2), randint(middle + 1, self.cfg.size - 2))
        #         for o in objs:
        #             if obj_pos == o[2]:
        #                 break
        #         else:
        #             if obj_pos != self.agent_pos and not self.next2door(obj_pos):
        #                 break

        #     self.put_obj(obj, obj_pos[0], obj_pos[1])
        
        return objs
    
    def next2door(self, pos):
        left = self.grid.get(pos[0] - 1, pos[1])
        right = self.grid.get(pos[0] + 1, pos[1])
        up = self.grid.get(pos[0], pos[1] - 1)
        down = self.grid.get(pos[0], pos[1] + 1)

        if (left is not None and left.type == 'door') or \
           (right is not None and right.type == 'door') or \
           (up is not None and up.type == 'door') or \
           (down is not None and down.type == 'door'):
            return True