def test_args():
    from ppo_experiments import Args
    from metric_rnd import args_for_rnd

    c = args_for_rnd(Args)()
    print(c.wandb_project_name)


def test_metric_on_random_maze(): ...
