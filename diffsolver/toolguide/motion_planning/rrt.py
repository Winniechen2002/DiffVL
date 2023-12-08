import math
import tqdm
import numpy as np
from typing import List, Tuple, Callable
from numpy.typing import NDArray
from ...config import MPConfig

float_type = np.float64
FloatArray = NDArray[float_type]
COLLISION_CHECK = Callable[[FloatArray], bool]

TRAPPED = 0
ADVANCED = 1
REACHED = 2


def steerTo(start: FloatArray, end: FloatArray, collision_check: COLLISION_CHECK, step_size: float, expand_dis: float= np.inf):
    """
    Returns:
        0: collision
        1: reach
    if no collision, return the new end point
    """
    delta = end - start
    length = float(np.linalg.norm(delta))

    if length > 0:
        delta /= length
        length = min(length, expand_dis)
        for i in range(0, int(math.floor(length/step_size))):
            cur = start + (i * step_size) * delta
            if collision_check(cur):
                return 0, None
        end = start + length * delta
        if collision_check(end):
            return 0, None
    return 1, end


def lvc(path: List[FloatArray], collHandle: COLLISION_CHECK, step_size: float) -> List[FloatArray]:
    for i in range(0, len(path)-1):
        for j in range(len(path)-1, i+1, -1):
            if steerTo(path[i], path[j], collHandle, step_size)[0]:
                return lvc(path[0:i+1] + path[j:len(path)], collHandle, step_size)
    return path


class Tree:
    def __init__(self, collision_check: COLLISION_CHECK, expand_dis: float, step_size: float):
        self.nodes: List[FloatArray] = []
        self.path: List[FloatArray] = []
        self.father: List[int] = []
        self.collision_check = collision_check
        self.expand_dis = expand_dis
        self.step_size = step_size

    def is_reaching_target(self, start1: FloatArray, start2: FloatArray):
        return np.abs(start1 - start2).max() < self.step_size

    def extend(self, q: FloatArray):
        nearest_ind = self.get_nearest_node_index(self.nodes, q)
        nearest_node = self.nodes[nearest_ind]
        flag, new_node = steerTo(nearest_node, q, self.collision_check, step_size=self.step_size, expand_dis=self.expand_dis)
        if flag:
            assert new_node is not None

            self.add_edge(new_node, nearest_ind)
            if self.is_reaching_target(new_node, q):
                return REACHED, new_node
            else:
                return ADVANCED, new_node
        else:
            return TRAPPED, None

    def connect(self, q: FloatArray):
        while True:
            S = self.extend(q)[0]
            if S != ADVANCED:
                break
        return S

    def add_edge(self, q: FloatArray, parent_id: int):
        self.nodes.append(q)
        self.father.append(parent_id)

    def backtrace(self):
        cur = len(self.nodes) - 1
        path = []
        while cur != 0:
            path.append(self.nodes[cur])
            cur = self.father[cur]
        return path


    @staticmethod
    def get_nearest_node_index(node_list: List[FloatArray], rnd_node: FloatArray):
        #TODO K-D tree or OCTTree
        dlist = [math.sqrt(((node - rnd_node) ** 2).sum()) for node in node_list]
        minind = dlist.index(min(dlist))
        return minind


class RRTConnectPlanner:
    def __init__(
        self,
        state_sampler: Callable[[], FloatArray],
        collision_checker: COLLISION_CHECK,
        config: MPConfig,
    ):

        self.state_sampler = state_sampler
        self.collision_checker = collision_checker
        self.config = config

    def __call__(self, start: FloatArray, goal: FloatArray):
        start = np.array(start)
        goal = np.array(goal)
        self.TA = self.TB = None

        # code for single direction
        self.TA = TA = Tree(self.collision_checker, self.config.expand_dis, self.config.step_size)
        TA.add_edge(start, -1)

        self.TB = TB = Tree(self.collision_checker, self.config.expand_dis, self.config.step_size)
        TB.add_edge(goal, -1)

        ran = range if not self.config.info else tqdm.trange
        for i in ran(self.config.max_iter):
            q_rand = np.array(self.state_sampler())
            S, q_new = TA.extend(q_rand)
            if not (S == TRAPPED):
                assert q_new is not None

                if TB.connect(q_new) == REACHED:
                    if i % 2 == 1:
                        TA, TB = TB, TA
                        self.TA = TA
                        self.TB = TB

                    path = TA.backtrace()[::-1] + TB.backtrace()[1:]
                    path = [start] + path + [goal]

                    if self.config.info:
                        print('last one in path', path[-1], 'length', len(path), 'lvc...', self.config.use_lvc)

                    if self.config.use_lvc:
                        self.original_path = path
                        path = lvc(path, self.collision_checker, self.config.step_size)
                    return path
            TA, TB = TB, TA
        return []