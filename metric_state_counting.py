from dataclasses import dataclass
import numpy as np


def args_for_state_counting(algo_args):
    @dataclass
    class ArgsStateCounter(algo_args):
        new_state_points: int = 1
        """number of points for exploring new state"""

    return ArgsStateCounter


class StateCounter:
    def __init__(self, x_size=10, y_size=7, scale=10):
        self.x_size = x_size
        self.y_size = y_size
        self.scale = scale
        self.shape = (y_size * scale, x_size * scale)
        self.num_states = np.prod(self.shape)

    def get_cell(self, obs):
        x, y, *_ = obs
        shifted_x = int((x.item() + self.x_size / 2) * self.scale)
        shifted_y = int((y.item() + self.y_size / 2) * self.scale)

        return shifted_x + shifted_y * self.shape[1]

    def update_visited_states(self, obs, visited_states):
        rewards = [0] * len(visited_states)
        for i in range(len(visited_states)):
            if self.get_cell(obs[i]) not in visited_states[i]:
                rewards[i] += 1
            visited_states[i].add(self.get_cell(obs[i]))

        return rewards, visited_states
    
    def get_metric_value(self, visited_states):
        return np.mean([len(x) for x in visited_states]) / self.num_states
    
    def get_visitation_maps(self, visited_states):
        visitation = np.zeros(self.shape)  # В эту переменную инициализируем пустой массив размером с карту
        for vis_set in visited_states:
            self.to_2d_visitation_map(visitation, vis_set)
        return visitation
    
    def to_2d_visitation_map(self, visitation, visitation_set):
        for x in visitation_set:
            i, j = divmod(x, self.shape[1])
            visitation[i, j] += 1
