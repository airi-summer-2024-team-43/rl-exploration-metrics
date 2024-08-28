from dataclasses import dataclass


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
        self.num_states = int(x_size * y_size * scale**2)

    def get_cell(self, obs):
        x, y, *_ = obs
        shifted_x = int((x.item() + self.x_size / 2) * self.scale)
        shifted_y = int((y.item() + self.y_size / 2) * self.scale)

        return shifted_x + shifted_y * self.x_size * self.scale

    def update_visited_states(self, obs, visited_states):
        rewards = [0] * len(visited_states)
        for i in range(len(visited_states)):
            if self.get_cell(obs[i]) not in visited_states[i]:
                rewards[i] += 1
            visited_states[i].add(self.get_cell(obs[i]))

        return rewards, visited_states
