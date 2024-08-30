import numpy as np
from sklearn.neighbors import NearestNeighbors
import gzip
import tqdm
import json
import matplotlib.pyplot as plt


class DistrMetric:
    def __init__(self) -> None:
        pass

    def k_nearest_neighbors(self, samples, k):
        k_nbrs = NearestNeighbors(n_neighbors=k, algorithm="auto", metric="euclidean")
        k_nbrs.fit(samples)
        return k_nbrs

    def function_term(self, MN, Ns, Ms):
        res = np.zeros_like(Ns, dtype=float)
        mask = Ns > 0
        res[mask] = -np.log(MN * Ns[mask] / (Ms[mask] + 1.0))
        # ans = np.max([ans, np.log(0.0001)])  # TODO - np.log(Cl / Cu)
        return res

    def get_KL(self, in_Ns, M, N, k):
        Ns = np.count_nonzero(in_Ns, axis=-1)
        Ms = k - Ns
        Ds = self.function_term(M / N, Ns, Ms)
        D = np.max([np.mean(Ds), 0.0])
        print(D)
        return D

    def metric(self, sample_X, sample_Y, k):
        sample_Z = np.concatenate((sample_X, sample_Y), axis=0)
        M, N = sample_Y.shape[0], sample_X.shape[0]
        MN = M / N

        # for both X & Y
        k_nbrs = self.k_nearest_neighbors(sample_Z, k)
        neighbors_inds = k_nbrs.kneighbors(sample_Z, return_distance=False)

        # k neighbors (sample_Y) in X: last M indices < N
        D_YX = self.get_KL(neighbors_inds[-M:] < N, M, N, k)
        # k neighbors (sample_X) in Y: first N indices >= N
        D_XY = self.get_KL(neighbors_inds[:N] >= N, N, M, k)
        D = -D_YX - D_XY
        return D


def download_sample(path):
    # with open(path, "rb") as f:
    #     return np.load(f)
    with gzip.GzipFile(path, "rb") as f:
        return np.load(f)


def data_use(
    name="",
    path="history/dataset_PointMaze_UMaze-v3__ppo_experiments__1__1724965430.npy.gz",
    N=20_000,
    k=10,
    env_samples=None,
):
    env_sample = env_samples[:N, :]
    u_sample = np.random.uniform(low=[-11.5, -11.5], high=[11.5, 11.5], size=(N, 2))
    print(f"env = {env_sample.shape}, u = {u_sample.shape}")
    metric = DistrMetric()
    m = metric.metric(env_sample, u_sample, k)
    print(f"metric {name} = {m}")
    return m


def graph(data, global_steps):
    # Prepare the plot
    plt.figure(figsize=(10, 6))

    # Plot each line
    for label, values in data.items():
        plt.plot(global_steps[: len(values)], values, marker="o", label=label)

    # Adding titles and labels
    plt.title("Metric with distr.")
    plt.xlabel("global_steps")
    plt.ylabel("Metric")
    plt.xticks(range(len(values)))  # Set x-ticks to match indices
    plt.legend()  # Show legend
    plt.grid(True, which="both", linestyle="-")  # Add grid for better readability

    # Show the plot
    plt.tight_layout()
    plt.savefig("my_plot_3.png")
    plt.show()


def analize(load=False):
    paths = {
        "none": "history/dataset_PointMaze_UMaze-v3__INT_REW_NONE_LARGE__10__1724965999.npy.gz",
        "RND": "history/dataset_PointMaze_UMaze-v3__INT_REW_RND_LARGE__10__1724965992.npy.gz",
        "MD": "history/dataset_PointMaze_UMaze-v3__INT_REW_MD_LARGE__10__1724965995.npy.gz",
    }

    k = 10

    if load:
        with open(f"m_distr_{k}.json", "r") as openfile:
            res = json.load(openfile)
    else:
        res = {
            "none": [],
            "RND": [],
            "MD": [],
        }

    env_sample = {key: download_sample(elem) for key, elem in paths.items()}

    all_N = [10_000, 50_000, 100_000, 200_000, 500_000, 750_000, 1_000_000, 1_250_000]
    for N in all_N:
        for name in paths:
            m = data_use(name, paths[name], N, k, env_sample[name])
            res[name].append(m)
            print()

        with open(f"m_distr_{k}.json", "w") as outfile:
            json.dump(res, outfile)

        graph(res, all_N)


# Example usage
if __name__ == "__main__":
    # main_random()
    # data_use()
    analize()
