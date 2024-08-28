import numpy as np
import torch

from metric_rnd import RNDModel, args_for_rnd
from metric_state_counting import StateCounter, args_for_state_counting
from metric_disagreement import Ensemble, args_for_disagreement


class MetricsUsage:
    def __init__(self) -> None:
        self.args = None
        self.metrics = list()  # rnd, count, model_disagreement

    def update_arg_class(self, args):
        args_class = args
        args_class = args_for_rnd(args_class)
        args_class = args_for_state_counting(args_class)
        args_class = args_for_disagreement(args_class)
        return args_class

    def _init_metrics_running_info(self):
        self.running_int_rewards = []
        self.running_state_counts = []
        self.episode_disagreements = []
        self.running_disagreements = []
        self.state_counts = [set() for _ in range(self.args.num_envs)]
        self.int_rewards = torch.zeros((self.args.num_steps, self.args.num_envs)).to(
            self.device
        )

    def _update_args(self, args):
        self.args = args

        args.use_rnd_metric = args.use_rnd_metric or args.use_rnd_intrinsic_reward
        if args.use_rnd_metric:
            self.metrics.append("rnd")
        if args.use_state_counting_metric:
            self.metrics.append("count")
        if args.use_model_disagreement_metric:
            self.metrics.append("model_disagreement")

    def init_metrics_info(self, args, obs_space, writer, device):
        self.writer = writer
        self.device = device
        self._update_args(args)
        self._init_metrics_running_info()

        if self.args.use_rnd_metric:
            if self.args.model_type == "mlp":
                self.rnd_model = RNDModel(
                    input_size=obs_space, output_size=args.output_rnd
                ).to(self.device)
            elif self.args.model_type == "conv":
                self.rnd_model = RNDModel(
                    input_size=obs_space, output_size=args.output_rnd
                ).to(self.device)
            else:
                raise NotImplementedError("Unknown RND model type")
        if self.args.use_state_counting_metric:
            map_name = args.env_map
            if map_name is None:
                self.state_counter = StateCounter()
            else:
                from maze_maps import maps

                map_size = np.array(maps[map_name]).shape
                self.state_counter = StateCounter(
                    x_size=map_size[1], y_size=map_size[0]
                )

        if self.args.use_model_disagreement_metric:
            self.ensemble = Ensemble(args)

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

    def update_disagreement(self, obs, actions):
        if self.args.use_model_disagreement_metric:
            disagreement = self.ensemble.get_disagreement(obs, actions)
            self.episode_disagreements.append(disagreement)

    def update_learnable_metric(self, obs):
        if self.args.use_rnd_metric:
            exploration_loss = self.rnd_model.get_forward_loss(obs)
            self.rnd_model.update(exploration_loss)

    def update_with_action(self, obs, action, next_obs):
        if self.args.use_model_disagreement_metric:
            disagreement_loss = self.ensemble.get_ensemble_loss(obs, action, next_obs)
            self.ensemble.update(disagreement_loss)

    def update_with_state(self, next_obs):
        if self.args.use_state_counting_metric:
            state_count_rewards, self.state_counts = (
                self.state_counter.update_visited_states(next_obs, self.state_counts)
            )
            state_count_rewards = np.mean(state_count_rewards)

    def end_episode_update(self):
        if self.args.use_rnd_metric:
            self.running_int_rewards.append(
                self.int_rewards[-self.args.num_steps :].cpu().numpy().mean()
            )
        if self.args.use_state_counting_metric:
            metric = self.state_counter.get_metric_value(self.state_counts)
            self.running_state_counts.append(metric)
        if self.args.use_model_disagreement_metric:
            metric = np.mean(self.episode_disagreements)
            self.episode_disagreements = []
            self.running_disagreements.append(metric)

    def log_metrics(self, step):
        if self.args.use_rnd_metric:
            self.writer.add_scalar(
                "metric/novelty_rnd", self.running_int_rewards[-1], step
            )
        if self.args.use_state_counting_metric:
            self.writer.add_scalar(
                "metric/state_counts", self.running_state_counts[-1], step
            )
        if self.args.use_model_disagreement_metric:
            self.writer.add_scalar(
                "metric/model_disagreement", self.running_disagreements[-1], step
            )
