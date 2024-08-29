import numpy as np
from sklearn.neighbors import NearestNeighbors
import gzip
import tqdm


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

    def intersection(self, set_A_np, set_B_np):
        set_A = {tuple(row) for row in set_A_np}
        set_B = {tuple(row) for row in set_B_np}
        inter_A_B = set_A.intersection(set_B)
        return inter_A_B

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

        sum_D = 0
        for y_i in tqdm.tqdm(sample_Y):
            neighbors_ind, _ = self.k_nearest_objects(y_i, k_nbrs)
            R_k = sample_Z[neighbors_ind]
            N_i = len(self.intersection(sample_X, R_k))
            M_i = len(self.intersection(sample_Y, R_k))
            sum_D += self.function_term(MN, N_i, M_i)

        sum_D = sum_D / M
        sum_D = np.max([sum_D, 0])  # TODO
        print(sum_D)
        return sum_D

    def cl_cu(self):
        ...
        # distances = <get distance from each point to each cluster: shape=(n_points, n_clusters>

        # cluster_indices = np.argmin(distances, axis=-1)

        # cluster_counts = np.zeros(n_clusters)

        # np.add.at(cluster_counts, cluster_indices, 1)

        # в cluster_counts число точек в каждом кластере

    def metric(self, sample_X, sample_Y, k):
        sample_Z = np.concatenate((sample_X, sample_Y), axis=0)
        ans_metric = -self.D(sample_X, sample_Y, sample_Z, k)
        ans_metric -= self.D(sample_Y, sample_X, sample_Z, k)
        return ans_metric


def download_sample(path):
    # with open(path, "rb") as f:
    #     return np.load(f)
    with gzip.GzipFile(path, "rb") as f:
        return np.load(f)


def data_use():
    path = "history/dataset_PointMaze_UMaze-v3__ppo_experiments__1__1724965430.npy.gz"
    env_sample = download_sample(path)[:20000, :]
    u_sample = np.random.uniform(low=-10, high=10, size=(1000, 2))
    print(f"env = {env_sample.shape}, u = {u_sample.shape}")
    metric = DistrMetric()
    k = 10
    m = metric.metric(env_sample, u_sample, k)
    print(m)


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


# Example usage
if __name__ == "__main__":
    # main_random()
    data_use()
