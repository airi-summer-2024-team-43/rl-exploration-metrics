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

# Reproducing results
Let's define Exploration Boost as one of the following algorithm modes:
- Entriopy: Standard PPO Entropy term is added to policy loss. Other boosts do not exclude entropy from loss, we just don't mention it
- RND: Adding intrinsic reward from RND output. More details [here](https://arxiv.org/pdf/1810.12894)
- Model Disagreement: Adding intrinsic reward from Model Disagreement output. More details [here](https://pathak22.github.io/exploration-by-disagreement/resources/icml19.pdf)
We ran experiments with the following settings:
- For environments **PointMaze** and **AntMaze**:
    - Running every map size (small, medium large)
    - Running with Exploration boosts (Entropy, RND)
    - Running only large maze map with Model Disagreement Exploration boost 
    - Tracking State Counting, RND, and Model Disagreement metrics on every run
- For environment **MontezumaRevenge**:
    - Running with every Exploration boost (Entropy, RND, Model Disagreement)
    - Tracking Episoding Length, RND, Model Disagreement metrics on every run

## Scripts
Below we present shell scripts we used for experiments. 
```bash
# POINT_MAZE
# RND boost, small maze
python ppo_experiments.py --use-entropy-loss --use-rnd-intrinsic-reward --use-rnd-metric --use-state-counting-metric --use-model-disagreement-metric --plot-visitation-map --track --env-map small --exp-name INT_REW_RND_SMALL 

# Entropy boost, small size
python ppo_experiments.py --use-entropy-loss --use-rnd-metric --use-state-counting-metric --use-model-disagreement-metric --plot-visitation-map --track --env-map small --exp-name INT_REW_NONE_SMALL 

# RND boost, medium size
python ppo_experiments.py --use-entropy-loss --use-rnd-intrinsic-reward --use-rnd-metric --use-state-counting-metric --use-model-disagreement-metric --track --plot-visitation-map --env-map medium --max-episode-steps 1200 --exp-name INT_REW_RND_MEDIUM 

# Entropy boost, medium size
python ppo_experiments.py --use-entropy-loss --use-rnd-metric --use-state-counting-metric --use-model-disagreement-metric --plot-visitation-map --track --env-map medium --max-episode-steps 1200 --exp-name INT_REW_NONE_MEDIUM 

# RND boost, large size
python ppo_experiments.py --use-entropy-loss --use-rnd-intrinsic-reward --use-rnd-metric --use-state-counting-metric --use-model-disagreement-metric --track --plot-visitation-map --env-map large --max-episode-steps 2400 --exp-name INT_REW_RND_LARGE 

# Model Disagreement boost, large size
python ppo_experiments.py --use-entropy-loss --use-model-disagreement-intrinsic-reward --use-rnd-metric --use-state-counting-metric --use-model-disagreement-metric --track --plot-visitation-map --env-map large --max-episode-steps 2400 --exp-name INT_REW_MD_LARGE 

# Entropy boost, large size
python ppo_experiments.py --use-entropy-loss --use-rnd-metric --use-state-counting-metric --use-model-disagreement-metric --plot-visitation-map --track --env-map large --max-episode-steps 2400 --exp-name INT_REW_NONE_LARGE 


# ANT_MAZE
# RND boost, small maze
python ppo_experiments.py --wandb-project-name cleanRL-exploration-antmaze --env-id AntMaze_UMaze-v4 --use-entropy-loss --use-rnd-intrinsic-reward --use-rnd-metric --use-state-counting-metric --use-model-disagreement-metric --plot-visitation-map --track --env-map small --exp-name INT_REW_RND_SMALL 

# Entropy boost, small size
python ppo_experiments.py --wandb-project-name cleanRL-exploration-antmaze --env-id AntMaze_UMaze-v4 --use-entropy-loss --use-rnd-metric --use-state-counting-metric --use-model-disagreement-metric --plot-visitation-map --track --env-map small --exp-name INT_REW_NONE_SMALL 

# RND boost, medium size
python ppo_experiments.py --wandb-project-name cleanRL-exploration-antmaze --env-id AntMaze_UMaze-v4 --use-entropy-loss --use-rnd-intrinsic-reward --use-rnd-metric --use-state-counting-metric --use-model-disagreement-metric --track --plot-visitation-map --env-map medium --max-episode-steps 1200 --exp-name INT_REW_RND_MEDIUM 

# Entropy boost, medium size
python ppo_experiments.py --wandb-project-name cleanRL-exploration-antmaze --env-id AntMaze_UMaze-v4 --use-entropy-loss --use-rnd-metric --use-state-counting-metric --use-model-disagreement-metric --plot-visitation-map --track --env-map medium --max-episode-steps 1200 --exp-name INT_REW_NONE_MEDIUM 

# RND boost, large size
python ppo_experiments.py --wandb-project-name cleanRL-exploration-antmaze --env-id AntMaze_UMaze-v4 --use-entropy-loss --use-rnd-intrinsic-reward --use-rnd-metric --use-state-counting-metric --use-model-disagreement-metric --track --plot-visitation-map --env-map large --max-episode-steps 2400 --exp-name INT_REW_RND_LARGE 

# Model Disagreement boost, large size
python ppo_experiments.py --wandb-project-name cleanRL-exploration-antmaze --env-id AntMaze_UMaze-v4 --use-entropy-loss --use-model-disagreement-intrinsic-reward --use-rnd-metric --use-state-counting-metric --use-model-disagreement-metric --track --plot-visitation-map --env-map large --max-episode-steps 2400 --exp-name INT_REW_MD_LARGE 

# Entropy boost, large size
python ppo_experiments.py --wandb-project-name cleanRL-exploration-antmaze --env-id AntMaze_UMaze-v4 --use-entropy-loss --use-rnd-metric --use-state-counting-metric --use-model-disagreement-metric --plot-visitation-map --track --env-map large --max-episode-steps 2400 --exp-name INT_REW_NONE_LARGE 


# MONTEZUMA_REVENGE
# RND boost
python ppo_experiments_montezuma.py --use-entropy-loss --use-rnd-intrinsic-reward --use-rnd-metric --use-model-disagreement-metric --track --capture-video --exp-name INT_REW_RND 

# Model Disagreement boost
python ppo_experiments_montezuma.py --use-entropy-loss --use-model-disagreement-intrinsic-reward --use-rnd-metric --use-model-disagreement-metric --track --capture-video --exp-name INT_REW_MD

# Entropy boost
python ppo_experiments_montezuma.py --use-entropy-loss --use-rnd-metric --use-model-disagreement-metric --track --capture-video --exp-name INT_REW_NONE 
```