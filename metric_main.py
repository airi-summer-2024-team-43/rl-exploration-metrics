import numpy as np
import torch

from metric_rnd import RNDModel, args_for_rnd
from metric_state_counting import StateCounter, args_for_state_counting


class MetricsUsage:
    def __init__(self) -> None:
        self.args = None
        self.metrics = list()  # rnd, count

    def update_arg_class(self, Old_Args):
        # TODO: как это сделать нормально, ибо вариант с if плохо работает
        New_Args = Old_Args
        New_Args = args_for_rnd(New_Args)
        New_Args = args_for_state_counting(New_Args)
        return New_Args

    def _init_metrics_running_info(self):
        self.running_int_rewards = []
        self.running_state_counts = []
        self.state_counts = [set() for _ in range(self.args.num_envs)]
        self.int_rewards = torch.zeros((self.args.num_steps, self.args.num_envs)).to(
            self.device
        )

    def _update_args(self, args):
        # TODO: не делать отдельный флаг метрики, а их перечислить?
        self.args = args

        args.use_rnd_metric = args.use_rnd_metric or args.use_rnd_intrinsic_reward
        if args.use_rnd_metric:
            self.metrics.append("rnd")
        if args.use_state_counting_metric:
            self.metrics.append("count")

    def init_metrics_info(self, args, obs_space, writer, device):
        self.writer = writer
        self.device = device
        self._update_args(args)
        self._init_metrics_running_info()

        if self.args.use_rnd_metric:
            self.rnd_model = RNDModel(
                input_size=obs_space, output_size=args.output_rnd
            ).to(self.device)
        if self.args.use_state_counting_metric:
            self.state_counter = StateCounter()

    def update_intrinsic_reward(self, obs, rewards):
        args = self.args
        if args.use_rnd_metric:
            intrinsic_rewards = self.rnd_model.get_intrinsic_reward(
                obs[-args.num_steps :]
            )
            self.int_rewards[-args.num_steps :] = intrinsic_rewards
            if args.use_rnd_intrinsic_reward:
                rewards[-args.num_steps :] = (
                    rewards[-args.num_steps :] * args.ext_coef
                    + intrinsic_rewards * args.int_coef
                )
        return rewards

    def update_learnable_metric(self, obs):
        if self.args.use_rnd_metric:
            exploration_loss = self.rnd_model.get_forward_loss(obs)
            self.rnd_model.update(exploration_loss)

    def update_with_state(self, next_obs):
        state_count_rewards = 0
        if self.args.use_state_counting_metric:
            state_count_rewards, self.state_counts = (
                self.state_counter.update_visited_states(next_obs, self.state_counts)
            )
            state_count_rewards = np.mean(state_count_rewards)
        # self.writer.add_scalar(
        #     f"iteration_{iteration}/intrinsic_reward",
        #     self.int_rewards[step].mean(),
        #     global_step,
        # )
        # self.writer.add_scalar(
        #     f"iteration_{iteration}/state_counts",
        #     np.mean([len(x) for x in self.state_counts]),
        #     global_step,
        # )

    def end_episode_update(self, episodic_length):
        self.running_int_rewards.append(
            self.int_rewards[-self.args.num_steps :].cpu().numpy().mean()
        )
        self.running_state_counts.append(np.mean([len(x) for x in self.state_counts]))

    def log_metrics(self, iteration):
        self.writer.add_scalar(
            "metric/intrinsic_reward_mean", self.running_int_rewards[-1], iteration
        )
        self.writer.add_scalar(
            "metric/state_counts", self.running_state_counts[-1], iteration
        )
