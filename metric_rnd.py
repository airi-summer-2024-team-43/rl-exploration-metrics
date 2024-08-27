from dataclasses import dataclass


def args_for_RND(algo_args):
    @dataclass
    class ArgsRND(algo_args):
        # RND
        ext_coef: float = 1.0
        """RND extrinsic reward coef"""
        int_coef: float = 0.5
        """RND intrinsic reward coef"""

    return ArgsRND
