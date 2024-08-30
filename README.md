# Exploration Metrics in RL
This repository presents experiments on measuring exploration in Reinforcement Learning conducted by **Team 43** during **AIRI Summer School 2024** \
For results, see [our report](REPORT.md)


# How to use

- Install [cleanrl](https://docs.cleanrl.dev) along with its dependencies 
- To launch experiment on [Robotics](https://robotics.farama.org) environment (`PointMaze`, `AntMaze`), run: 
```python
python ppo_experiments.py
```
- To launch experiment on [Atari](https://gymnasium.farama.org/environments) environments (`MontezumaRevenge`), run:
```python
python ppo_experiments_atari.py
```
- Metrics for `ppo_experiments.py` are defined in `metric_main.py`. You can track them by passing corresponding arguments to the script
- Available metrics for atari are defined in `ppo_experiments_atari.py`. You can track them by passing corresponding arguments to the script