import numpy as np
import pandas as pd
import torch
from numba import jit, prange
from torch.autograd import Function


def smooth_time_series(x, window_size, step):
    smoothed = np.apply_along_axis(
        lambda x: pd.Series(x.squeeze()).rolling(window=window_size, step=step).mean().dropna(),
        axis=1,
        arr=x.squeeze(),
    )
    return smoothed


# ----------------------------------------------------------------------------------------------------------------------
#
# The following is the CPU implementation based on https://github.com/Sleepwalking/pytorch-softdtw
# Credit goes to Kanru Hua.
# I've added support for batching and pruning.
#
# ----------------------------------------------------------------------------------------------------------------------
@jit(nopython=True, parallel=True)
def compute_softdtw(D, gamma, bandwidth):
    B = D.shape[0]
    N = D.shape[1]
    M = D.shape[2]
    R = np.ones((B, N + 2, M + 2)) * np.inf
    R[:, 0, 0] = 0
    for b in prange(B):
        for j in range(1, M + 1):
            for i in range(1, N + 1):

                # Check the pruning condition
                if 0 < bandwidth < np.abs(i - j):
                    continue

                r0 = -R[b, i - 1, j - 1] / gamma
                r1 = -R[b, i - 1, j] / gamma
                r2 = -R[b, i, j - 1] / gamma
                rmax = max(max(r0, r1), r2)
                rsum = np.exp(r0 - rmax) + np.exp(r1 - rmax) + np.exp(r2 - rmax)
                softmin = -gamma * (np.log(rsum) + rmax)
                R[b, i, j] = D[b, i - 1, j - 1] + softmin
    return R


# ----------------------------------------------------------------------------------------------------------------------
@jit(nopython=True, parallel=True)
def compute_softdtw_backward(D_, R, gamma, bandwidth):
    B = D_.shape[0]
    N = D_.shape[1]
    M = D_.shape[2]
    D = np.zeros((B, N + 2, M + 2))
    E = np.zeros((B, N + 2, M + 2))
    D[:, 1 : N + 1, 1 : M + 1] = D_
    E[:, -1, -1] = 1
    R[:, :, -1] = -np.inf
    R[:, -1, :] = -np.inf
    R[:, -1, -1] = R[:, -2, -2]
    for k in prange(B):
        for j in range(M, 0, -1):
            for i in range(N, 0, -1):

                if np.isinf(R[k, i, j]):
                    R[k, i, j] = -np.inf

                # Check the pruning condition
                if 0 < bandwidth < np.abs(i - j):
                    continue

                a0 = (R[k, i + 1, j] - R[k, i, j] - D[k, i + 1, j]) / gamma
                b0 = (R[k, i, j + 1] - R[k, i, j] - D[k, i, j + 1]) / gamma
                c0 = (R[k, i + 1, j + 1] - R[k, i, j] - D[k, i + 1, j + 1]) / gamma
                a = np.exp(a0)
                b = np.exp(b0)
                c = np.exp(c0)
                E[k, i, j] = (
                    E[k, i + 1, j] * a + E[k, i, j + 1] * b + E[k, i + 1, j + 1] * c
                )
    return E[:, 1 : N + 1, 1 : M + 1]


# ----------------------------------------------------------------------------------------------------------------------
class _SoftDTW(Function):
    """
    CPU implementation based on https://github.com/Sleepwalking/pytorch-softdtw
    """

    @staticmethod
    def forward(ctx, D, gamma, bandwidth):
        dev = D.device
        dtype = D.dtype
        gamma = torch.Tensor([gamma]).to(dev).type(dtype)  # dtype fixed
        bandwidth = torch.Tensor([bandwidth]).to(dev).type(dtype)
        D_ = D.detach().cpu().numpy()
        g_ = gamma.item()
        b_ = bandwidth.item()
        R = torch.Tensor(compute_softdtw(D_, g_, b_)).to(dev).type(dtype)
        ctx.save_for_backward(D, R, gamma, bandwidth)
        return R[:, -2, -2]

    @staticmethod
    def backward(ctx, grad_output):
        dev = grad_output.device
        dtype = grad_output.dtype
        D, R, gamma, bandwidth = ctx.saved_tensors
        D_ = D.detach().cpu().numpy()
        R_ = R.detach().cpu().numpy()
        g_ = gamma.item()
        b_ = bandwidth.item()
        E = torch.Tensor(compute_softdtw_backward(D_, R_, g_, b_)).to(dev).type(dtype)
        return grad_output.view(-1, 1, 1).expand_as(E) * E, None, None


# ----------------------------------------------------------------------------------------------------------------------
class SoftDTW(torch.nn.Module):
    """
    The soft DTW implementation that optionally supports CUDA
    """

    def __init__(
        self, use_cuda, gamma=1.0, normalize=False, bandwidth=None, dist_func=None
    ):
        """
        Initializes a new instance using the supplied parameters
        :param use_cuda: Flag indicating whether the CUDA implementation should be used
        :param gamma: sDTW's gamma parameter
        :param normalize: Flag indicating whether to perform normalization
                          (as discussed in https://github.com/mblondel/soft-dtw/issues/10#issuecomment-383564790)
        :param bandwidth: Sakoe-Chiba bandwidth for pruning. Passing 'None' will disable pruning.
        :param dist_func: Optional point-wise distance function to use. If 'None', then a default Euclidean distance function will be used.
        """
        super(SoftDTW, self).__init__()
        self.normalize = normalize
        self.gamma = gamma
        self.bandwidth = 0 if bandwidth is None else float(bandwidth)
        self.use_cuda = use_cuda

        # Set the distance function
        if dist_func is not None:
            self.dist_func = dist_func
        else:
            self.dist_func = SoftDTW._euclidean_dist_func

    def _get_func_dtw(self, x, y):
        """
        Checks the inputs and selects the proper implementation to use.
        """
        bx, lx, dx = x.shape
        by, ly, dy = y.shape
        # Make sure the dimensions match
        # assert bx == by  # Equal batch sizes ### This assertion was removed because looking for pairwise similarities
        assert dx == dy  # Equal feature dimensions

        use_cuda = self.use_cuda

        if use_cuda and (
            lx > 1024 or ly > 1024
        ):  # We should be able to spawn enough threads in CUDA
            print(
                "SoftDTW: Cannot use CUDA because the sequence length > 1024 (the maximum block size supported by CUDA)"
            )
            use_cuda = False

        # Finally, return the correct function
        return None if use_cuda else _SoftDTW.apply

    @staticmethod
    def _euclidean_dist_func(x, y):
        """
        Calculates the Euclidean distance between each element in x and y per timestep
        """
        n = x.size(1)
        m = y.size(1)
        d = x.size(2)
        x = x.unsqueeze(2).expand(-1, n, m, d)
        y = y.unsqueeze(1).expand(-1, n, m, d)
        return torch.pow(x - y, 2).sum(3)

    def forward(self, X, Y):
        """
        Compute the soft-DTW value between X and Y
        :param X: One batch of examples, batch_size x seq_len x dims
        :param Y: The other batch of examples, batch_size x seq_len x dims
        :return: The computed results
        """

        # Check the inputs and get the correct implementation
        func_dtw = self._get_func_dtw(X, Y)

        if self.normalize:
            # Stack everything up and run
            x = torch.cat([X, X, Y])
            y = torch.cat([Y, X, Y])
            D = self.dist_func(x, y)
            out = func_dtw(D, self.gamma, self.bandwidth)
            out_xy, out_xx, out_yy = torch.split(out, X.shape[0])
            return out_xy - 1 / 2 * (out_xx + out_yy)
        else:
            ## repeat X and Y to compute the pairwise distance matrix between the entries of X and Y
            x = torch.cat([X for _ in range(Y.shape[0])])
            y = torch.cat(
                [Y[[i]] for i in range(Y.shape[0]) for _ in range(X.shape[0])]
            )
            D_xy = self.dist_func(x, y)
            return func_dtw(D_xy, self.gamma, self.bandwidth).reshape(
                X.shape[0], Y.shape[0]
            )


SDTW = SoftDTW(gamma=0.005, normalize=False, use_cuda=False)


def compute_CE(x):
    """
    x shape : (n , n_hidden)
    return : output : (n , 1)
    """
    return torch.sqrt(torch.sum(torch.square(x[:, 1:] - x[:, :-1]), dim=1))


def compute_similarity(z, centroids, similarity="EUC"):
    """
    Function that compute distance between a latent vector z and the clusters centroids.

    similarity : can be in [CID,EUC,COR] :  euc for euclidean,  cor for correlation and CID
                 for Complexity Invariant Similarity.
    z shape : (batch_size, n_hidden)
    centroids shape : (n_clusters, n_hidden)
    output : (batch_size , n_clusters)
    """
    n_clusters, n_hidden = centroids.shape[0], centroids.shape[1]
    bs = z.shape[0]

    if similarity == "CID":
        CE_z = compute_CE(z).unsqueeze(1)  # shape (batch_size , 1)
        CE_cen = compute_CE(centroids).unsqueeze(0)  ## shape (1 , n_clusters )
        z = z.unsqueeze(0).expand((n_clusters, bs, n_hidden))
        mse = torch.sqrt(torch.sum((z - centroids.unsqueeze(1)) ** 2, dim=2))
        CE_z = CE_z.expand((bs, n_clusters))  # (bs , n_clusters)
        CE_cen = CE_cen.expand((bs, n_clusters))  # (bs , n_clusters)
        CF = torch.max(CE_z, CE_cen) / torch.min(CE_z, CE_cen)
        return torch.transpose(mse, 0, 1) * CF

    elif similarity == "EUC":
    

        z = z.expand((n_clusters, bs, n_hidden))
        mse = torch.sqrt(torch.sum((z - centroids.unsqueeze(1)) ** 2, dim=2))
        return torch.transpose(mse, 0, 1)

    elif similarity == "COR":
        std_z = (
            torch.std(z, dim=1).unsqueeze(1).expand((bs, n_clusters))
        )  ## (bs,n_clusters)
        mean_z = (
            torch.mean(z, dim=1).unsqueeze(1).expand((bs, n_clusters))
        )  ## (bs,n_clusters)
        std_cen = (
            torch.std(centroids, dim=1).unsqueeze(0).expand((bs, n_clusters))
        )  ## (bs,n_clusters)
        mean_cen = (
            torch.mean(centroids, dim=1).unsqueeze(0).expand((bs, n_clusters))
        )  ## (bs,n_clusters)
        ## covariance
        z_expand = z.unsqueeze(1).expand((bs, n_clusters, n_hidden))
        cen_expand = centroids.unsqueeze(0).expand((bs, n_clusters, n_hidden))
        prod_expec = torch.mean(z_expand * cen_expand, dim=2)  ## (bs , n_clusters)
        pearson_corr = (prod_expec - mean_z * mean_cen) / (std_z * std_cen)
        return torch.sqrt(2 * (1 - pearson_corr))

    elif similarity == "SDTW":
        res = SDTW(z[:, :, None], centroids[:, :, None])
        # print(res.shape)
        # print(res)
        return res
