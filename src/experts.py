import numpy as np

from minigrid.core.constants import OBJECT_TO_IDX, COLOR_TO_IDX
from minigrid.core.actions import Actions

from custom_env import MISSION_TO_ACTION


class Cell:
    """
    Class cell represents a cell in the world which have the properties:
    position: represented by tuple of x and y coordinates initially set to (0,0).
    parent: Contains the parent cell object visited before we arrived at this cell.
    g, h, f: Parameters used when calling our heuristic function.
    """

    def __init__(self):
        self.position = (0, 0)
        self.parent = None
        self.g = 0
        self.h = 0
        self.f = 0

    """
    Overrides equals method because otherwise cell assign will give
    wrong results.
    """

    def __eq__(self, cell):
        return self.position == cell.position

    def showcell(self):
        print(self.position)


class Gridworld:
    """
    Gridworld class represents the  external world here a grid M*M matrix.
    world_size: create a numpy array with the given world_size default is 5.
    """

    def __init__(self, world_size=(5, 5)):
        self.w = np.zeros(world_size)
        self.world_x_limit = world_size[0]
        self.world_y_limit = world_size[1]
        self.obstacles = []

    def get_neighbours(self, cell):
        """
        Return the neighbours of cell
        """
        neughbour_cord = [
            (-1, 0),
            (0, -1),
            (0, 1),
            (1, 0)
        ]
        current_x = cell.position[0]
        current_y = cell.position[1]
        neighbours = []
        for n in neughbour_cord:
            x = current_x + n[0]
            y = current_y + n[1]
            if 0 <= x < self.world_x_limit and 0 <= y < self.world_y_limit and (x, y) not in self.obstacles:
                c = Cell()
                c.position = (x, y)
                c.parent = cell
                neighbours.append(c)
        return neighbours
    
    def add_obstacles(self, img, target_id, target_clr):
        for x in range(len(img)):
            for y in range(len(img)):
                if img[x][y][0] not in [1, 10, target_id]:
                    self.obstacles.append((x, y))
                elif img[x][y][0] == target_id and img[x][y][1] != target_clr:
                    self.obstacles.append((x, y))


def astar(world, start, goal):
    """
    Implementation of A* algorithm.
    world : Object of the world object.
    start : Object of the cell as  start position.
    stop  : Object of the cell as goal position.
    """
    _open = []
    _closed = []
    _open.append(start)

    while _open:
        min_f = np.argmin([n.f for n in _open])
        current = _open[min_f]
        _closed.append(_open.pop(min_f))
        if current == goal:
            break
        for n in world.get_neighbours(current):
            if n in _closed:
                continue
            n.g = current.g + 1
            x1, y1 = n.position
            x2, y2 = goal.position
            n.h = (y2 - y1) ** 2 + (x2 - x1) ** 2
            n.f = n.h + n.g

            if n in _open:
                i = _open.index(n)
                if n.f <= _open[i].f:
                    _open.pop(i)
                else:
                    continue
            _open.append(n)
    path = []
    while current.parent is not None:
        path.append(current.position)
        current = current.parent
    path.append(current.position)
    return path[::-1]


class Expert:
    def __init__(self, cfg):
        self.cfg = cfg
        self.paths = []
        self.actions = None
        self.dones = None

    def __call__(self, obs, state, _):
        if self.cfg.algo == 'bc':
            if 'symbols' in obs.keys():
                images = np.transpose(obs['symbols'], (0, 2, 3, 1))
            else:
                images = np.transpose(obs['image'], (0, 2, 3, 1))
            mission = obs['mission']
            direction = obs['direction']
        elif self.cfg.algo == 'test':
            images = obs['image']
            mission = obs['mission']
            direction = obs['direction']
        else:
            images = obs[:, :192].reshape(-1, 8, 8, 3)
            mission = obs[:, 192:224]
            direction = obs[:, 224]

        agent_xs, agent_ys = Expert.get_player_positions(images)

        self.paths = []
        self.actions = None
        
        self.actions, obj_is, clr_is = Expert.decode_missions(mission)
        if self.dones is None:
            self.dones = [False] * len(self.actions)
        
        target_xs, target_ys = self.get_target_positions(obj_is, clr_is, agent_xs, agent_ys, images)
        
        for i in range(len(agent_xs)):
            if self.dones[i] or target_xs[i] == None or target_ys[i] == None:
                self.paths.append([])
                continue
            
            world = Gridworld(world_size=(images[i].shape[0], images[i].shape[1]))

            world.add_obstacles(images[i], obj_is[i], clr_is[i])

            start = Cell()
            goal = Cell()
            start.position = (agent_xs[i], agent_ys[i])
            goal.position = (target_xs[i], target_ys[i])
            
            path = astar(world, start, goal)[1:]
            if path != [] and path[-1] != goal.position:
                self.paths.append([])
            
            self.paths.append(path)

        actions = self.calculate_actions(agent_xs, agent_ys, direction)

        return actions, state

    def decode_missions(msns):
        vocab = [chr(x) for x in range(ord('a'), ord('z') + 1)]
        vocab.insert(0, ' ')
        
        acts = []
        obj_i = []
        clr_i = []

        for tkns in msns:
            msn = ''.join([vocab[i] for i in tkns])

            for m in MISSION_TO_ACTION.keys():
                if m in msn:
                    acts.append(MISSION_TO_ACTION[m])
                    break
            
            if acts[-1] == Actions.drop:
                obj_i.append(None)
                clr_i.append(None)
            elif m == 'go to goal':
                obj_i.append(8)
                clr_i.append(1)
            elif m == 'move':
                if 'left' in msn:
                    obj_i.append(-2)
                    clr_i.append(None)
                elif 'right' in msn:
                    obj_i.append(-1)
                    clr_i.append(None)
                elif 'up' in msn:
                    obj_i.append(None)
                    clr_i.append(-2)
                elif 'down' in msn:
                    obj_i.append(None)
                    clr_i.append(-1)
            else:
                for obj in OBJECT_TO_IDX.keys():
                    if obj in msn:
                        obj_i.append(OBJECT_TO_IDX[obj])
                        break
                for clr in COLOR_TO_IDX.keys():
                    if clr in msn:
                        clr_i.append(COLOR_TO_IDX[clr])
                        break
        
        return acts, obj_i, clr_i


    def get_player_positions(imgs):
        agent_x = []
        agent_y = []

        for i, img in enumerate(imgs):
            for x in range(len(img)):
                for y in range(len(img)):
                    if img[x][y][0] == 10:
                        agent_x.append(x)
                        agent_y.append(y)
        
        return agent_x, agent_y

    def get_target_positions(self, obj_is, clr_is, agent_xs, agent_ys, imgs):
        target_x = []
        target_y = []

        for i, img in enumerate(imgs):
            if self.actions[i] == Actions.drop:
                target_x.append(None)
                target_y.append(None)
                continue
            
            if self.actions[i] == None:
                if obj_is[i] == -2:
                    target_x.append(1)
                    target_y.append(self.find_empty(img, agent_xs[i], agent_ys[i], 1, None))
                    continue
                elif obj_is[i] == -1:
                    target_x.append(len(img) - 2)
                    target_y.append(self.find_empty(img, agent_xs[i], agent_ys[i], len(img) - 2, None))
                    continue
                elif clr_is[i] == -2:
                    target_x.append(self.find_empty(img, agent_xs[i], agent_ys[i], None, 1))
                    target_y.append(1)
                    continue
                elif clr_is[i] == -1:
                    target_x.append(self.find_empty(img, agent_xs[i], agent_ys[i], None, len(img) - 2))
                    target_y.append(len(img) - 2)
                    continue

            for x in range(len(img)):
                for y in range(len(img)):
                    if img[x][y][0] == obj_is[i] and img[x][y][1] == clr_is[i]:
                        target_x.append(x)
                        target_y.append(y)
            
            if len(target_x) != i + 1:
                target_x.append(None)
                target_y.append(None)
        
        return target_x, target_y

    def calculate_actions(self, agent_x, agent_y, dirs):
        acts = []

        for i in range(len(agent_x)):

            if self.dones[i]:
                acts.append(6)
                self.dones[i] = False
                continue

            if self.paths[i] == []:
                if self.actions[i] == Actions.drop:
                    acts.append(4)
                    self.dones[i] = True
                else:
                    acts.append(6)
                
                continue

            target = self.paths[i][0]

            dx = target[0] - agent_x[i]
            dy = target[1] - agent_y[i]

            if dx == 0:
                if dy == 1:
                    target_dir = 1
                else:
                    target_dir = 3
            elif dx == 1:
                target_dir = 0
            else:
                target_dir = 2

            match dirs[i]:
                case 0:
                    match target_dir:
                        case 0:
                            acts.append(self.maybe_forward(i))
                        case 1:
                            acts.append(1)
                        case 2:
                            acts.append(1)
                        case 3:
                            acts.append(0)
                case 1:
                    match target_dir:
                        case 0:
                            acts.append(0)
                        case 1:
                            acts.append(self.maybe_forward(i))
                        case 2:
                            acts.append(1)
                        case 3:
                            acts.append(1)
                case 2:
                    match target_dir:
                        case 0:
                            acts.append(1)
                        case 1:
                            acts.append(0)
                        case 2:
                            acts.append(self.maybe_forward(i))
                        case 3:
                            acts.append(1)
                case 3:
                    match target_dir:
                        case 0:
                            acts.append(1)
                        case 1:
                            acts.append(1)
                        case 2:
                            acts.append(0)
                        case 3:
                            acts.append(self.maybe_forward(i))
        
        return acts

    def maybe_forward(self, i):
        # Ako nije na kraju puta
        # idi napred
        if len(self.paths[i]) > 1:
            return 2
        
        # Ako je na kraju puta i akcija je None (move, go to goal)
        # idi napred
        if self.actions[i] is None:
            self.dones[i] = True
            return 2
        
        # Ako je na kraju puta i akcija je definisana
        # zapisi da si gotov
        if self.actions[i] != Actions.done:
            self.dones[i] = True
        
        # Odigraj akciju
        return self.actions[i]
    
    def find_empty(self, img, agent_x=None, agent_y=None, target_x=None, target_y=None):
        if target_x is None:
            diff = 0

            while True:
                right_out = False
                left_out = False
                if agent_x + diff >= len(img):
                    right_out = True
                elif img[agent_x + diff][target_y][0] in [1, 10]:
                    return agent_x + diff

                if agent_x - diff < 0:
                    left_out = True
                
                elif img[agent_x - diff][target_y][0] in [1, 10]:
                    return agent_x - diff
                
                if right_out and left_out:
                    return None
                
                diff += 1
        
        if target_y is None:
            diff = 0

            while True:
                bottom_out = False
                top_out = False
                if agent_y + diff >= len(img):
                    bottom_out = True
                elif img[target_x][agent_y + diff][0] in [1, 10]:
                    return agent_y + diff
                
                if agent_y - diff < 0:
                    top_out = True
                elif img[target_x][agent_y - diff][0] in [1, 10]:
                    return agent_y - diff
                
                if bottom_out and top_out:
                    return None
                
                diff += 1
                