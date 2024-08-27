def test_args():
    from ppo_experiments import Args
    from metric_rnd import args_for_RND

    c = args_for_RND(Args)()
    print(c.wandb_project_name)
