from dataclasses import dataclass

def args_for_state_counting(algo_args):
    @dataclass
    class ArgsStateCounter(algo_args):
        new_state_points: int = 1
        """number of points for exploring new state"""

    return ArgsStateCounter


class StateCounter:
    def __init__(self):
        pass

    def get_cell(obs):
        x, y, *_ = obs["observation"]
        shifted_x = round(x + 4.5)
        shifted_y = round(y + 3.0)

        return shifted_x + shifted_y * 10

    