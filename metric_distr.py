import numpy as np
from sklearn.neighbors import NearestNeighbors


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
            ans = 10e4
        # ans = np.max([ans, np.log(0.0001)])  # TODO - np.log(Cl / Cu)
        return ans

    def D(self, sample_X, sample_Y, k):
        M = sample_Y.shape[0]
        N = sample_X.shape[0]

        sample_Z = np.concatenate((sample_X, sample_Y), axis=0)
        k_nbrs = self.k_nearest_neighbors(sample_Z, k)

        sum_D = 0
        for y_i in sample_Y:
            neighbors_ind, distances = metric.k_nearest_objects(y_i, k_nbrs)
            R_k = sample_Z[neighbors_ind]
            N_i = len(self.intersection(sample_X, R_k))
            M_i = len(self.intersection(sample_Y, R_k))
            MN = M / N
            sum_D += self.function_term(MN, N_i, M_i)

        sum_D = sum_D / M
        sum_D = np.max([sum_D, 0])  # TODO
        print(sum_D)


# Example usage
if __name__ == "__main__":
    metric = DistrMetric()

    np.random.seed(42)
    samples_X = np.clip(np.random.normal(3, 2.5, size=(10, 3)) + 0.5, a_max=1, a_min=0)
    samples_Y = np.random.rand(50, 3)
    k = 3
    # x_i = np.array([0.5, 0.5, 0.5])

    # k_nbrs = metric.k_nearest_neighbors(samples_X, k)
    metric.D(samples_Y, samples_X, k)

    # neighbors, distances = metric.k_nearest_objects(x_i, k_nbrs)

    # print("Indices of k-nearest neighbors:", neighbors)
    # print("k-nearest neighbors:", samples_X[neighbors])
    # print("Distances to k-nearest neighbors:", distances)
