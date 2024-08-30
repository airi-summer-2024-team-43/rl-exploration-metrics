import numpy as np
from sklearn.neighbors import NearestNeighbors
import gzip
import tqdm
import json


class DistrMetric:
    def __init__(self) -> None:
        pass

    def k_nearest_neighbors(self, samples, k):
        k_nbrs = NearestNeighbors(n_neighbors=k, algorithm="auto", metric="euclidean")
        k_nbrs.fit(samples)
        return k_nbrs

    def k_nearest_objects(self, x_i, k_nbrs):
        x_i = x_i.reshape(1, -1)
        distances, indices = k_nbrs.kneighbors(x_i)
        return indices.flatten(), distances.flatten()

    def function_term(self, MN, N_i, M_i):
        ans = MN * N_i / (M_i + 1)
        if N_i > 0:
            ans = -np.log(ans)
        else:
            ans = 0
        # ans = np.max([ans, np.log(0.0001)])  # TODO - np.log(Cl / Cu)
        return ans

    def D(self, sample_X, sample_Y, sample_Z, k):
        M = sample_Y.shape[0]
        N = sample_X.shape[0]
        MN = M / N

        k_nbrs = self.k_nearest_neighbors(sample_Z, k)
        sX, sY = self._to_set(sample_X), self._to_set(sample_Y)

        sum_D = 0
        for y_i in tqdm.tqdm(sample_Y):
            neighbors_ind, _ = self.k_nearest_objects(y_i, k_nbrs)
            R_k = sample_Z[neighbors_ind]

            sRk = self._to_set(R_k)
            N_i = self.n_intersects(sX, sRk)
            M_i = self.n_intersects(sY, sRk)
            sum_D += self.function_term(MN, N_i, M_i)

        sum_D = sum_D / M
        sum_D = np.max([sum_D, 0])  # TODO
        print(sum_D)
        return sum_D

    def metric(self, sample_X, sample_Y, k):
        sample_Z = np.concatenate((sample_X, sample_Y), axis=0)
        ans_metric = -self.D(sample_X, sample_Y, sample_Z, k)
        ans_metric -= self.D(sample_Y, sample_X, sample_Z, k)
        return ans_metric

    @staticmethod
    def n_intersects(a, b):
        return len(a & b)

    @staticmethod
    def _to_set(a):
        return {tuple(x) for x in a}


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
    u_sample = np.random.uniform(low=-10, high=10, size=(N // 4, 2))
    print(f"env = {env_sample.shape}, u = {u_sample.shape}")
    metric = DistrMetric()
    m = metric.metric(env_sample, u_sample, k)
    print(f"metric {name} = {m}")
    return m


def main_random():
    metric = DistrMetric()

    np.random.seed(42)
    samples_X = np.clip(np.random.normal(3, 2.5, size=(10, 3)) + 0.5, a_max=1, a_min=0)
    samples_Y = np.random.rand(50, 3)
    k = 3
    # x_i = np.array([0.5, 0.5, 0.5])

    # k_nbrs = metric.k_nearest_neighbors(samples_X, k)
    print(metric.metric(samples_Y, samples_X, k))

    # neighbors, distances = metric.k_nearest_objects(x_i, k_nbrs)

    # print("Indices of k-nearest neighbors:", neighbors)
    # print("k-nearest neighbors:", samples_X[neighbors])
    # print("Distances to k-nearest neighbors:", distances)


def analize(load=False):
    paths = {
        "none": "history/dataset_PointMaze_UMaze-v3__INT_REW_NONE_LARGE__10__1724965999.npy.gz",
        "rnd": "history/dataset_PointMaze_UMaze-v3__INT_REW_RND_LARGE__10__1724965992.npy.gz",
        "md": "history/dataset_PointMaze_UMaze-v3__INT_REW_MD_LARGE__10__1724965995.npy.gz",
    }

    k = 25

    if load:
        with open(f"m_distr_{k}.json", "r") as openfile:
            res = json.load(openfile)
    else:
        res = {
            "none": [],
            "rnd": [],
            "md": [],
        }

    env_sample = {key: download_sample(elem) for key, elem in paths.items()}

    all_N = [1000, 5000, 10000]
    for N in all_N:
        for name in paths:
            m = data_use(name, paths[name], N, k, env_sample[name])
            res[name].append(m)
            print()

        with open(f"m_distr_{k}.json", "w") as outfile:
            json.dump(res, outfile)


# Example usage
if __name__ == "__main__":
    # main_random()
    # data_use()
    analize()
