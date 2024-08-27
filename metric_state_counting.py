from dataclasses import dataclass

def args_for_state_counting(algo_args):
    @dataclass
    class ArgsStateCounter(algo_args):
        new_state_points: int = 1
        """number of points for exploring new state"""

    return ArgsStateCounter


class StateCounter():
    def __init__(self):
        pass

    def get_cell(self, obs):
        x, y, *_ = obs
        shifted_x = int(x.item() + 5.0)
        shifted_y = int(y.item() + 3.5)

        return shifted_x + shifted_y * 10
    
    def update_visited_states(self, obs, visited_states):
        rewards = [0] * len(visited_states)
        for i in range(len(visited_states)):
            if self.get_cell(obs[i]) not in visited_states[i]:
                rewards[i] += 1
            visited_states[i].add(self.get_cell(obs[i]))

        return rewards, visited_states
    