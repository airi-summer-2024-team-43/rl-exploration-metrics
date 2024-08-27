import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np

env_id = "PointMaze_Large_Diverse_G-v3"
env = gym.make(env_id)
obs, info = env.reset()
print(obs)

coords = []

while True:
    # Using random policy to make actions
    obs, reward, terminated, truncated, info = env.step(env.action_space.sample())
    print(obs)
    first_two_coords = obs['observation'][:2]
    coords.append(first_two_coords)
    print("x, y координаты:", first_two_coords)
    if (terminated) or (truncated):
        break

coords_array = np.array(coords)
plt.figure(figsize=(10, 8))
plt.hist2d(coords_array[:, 0], coords_array[:, 1], bins=50, cmap='Blues')
plt.title("Тепловая карта координат")
plt.xlabel("Координата X")
plt.ylabel("Координата Y")
plt.colorbar(label='Плотность')
plt.show()
