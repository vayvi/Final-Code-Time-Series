import gc

import torch
import torch.nn as nn
from sklearn.cluster import AgglomerativeClustering

from utils import compute_similarity, smooth_time_series


class TAE_encoder(nn.Module):
    """
    Class for temporal autoencoder encoder.
    filter_1 : filter size of the first convolution layer
    filter_lstm : hidden size of the lstm.
    pooling : pooling number for maxpooling.
    """

    def __init__(self, filter_1, filter_lstm, pooling):
        super().__init__()

        self.hidden_lstm_1 = filter_lstm[0]
        self.hidden_lstm_2 = filter_lstm[1]
        self.pooling = pooling
        self.n_hidden = None
        ## CNN PART
        ### output shape (batch_size, 50 , n_hidden = 64)
        self.conv_layer = nn.Sequential(
            nn.Conv1d(
                in_channels=1,
                out_channels=filter_1,
                kernel_size=10,  # kernel size should probably be not hard coded
                stride=1,
                padding="same",  # padding = 'same' in tf
            ),
            nn.LeakyReLU(),
            nn.MaxPool1d(self.pooling),
        )

        ## LSTM PART
        ### output shape (batch_size , n_hidden = 64 , 50)
        self.lstm_1 = nn.LSTM(
            input_size=50,
            hidden_size=self.hidden_lstm_1,
            batch_first=True,
            bidirectional=True,
        )

        ### output shape (batch_size , n_hidden = 64 , 1)
        self.lstm_2 = nn.LSTM(
            input_size=50,
            hidden_size=self.hidden_lstm_2,
            batch_first=True,
            bidirectional=True,
        )

    def forward(self, x):

        ## encoder
        out_cnn = self.conv_layer(x)
        out_cnn = out_cnn.permute((0, 2, 1))
        out_lstm1, _ = self.lstm_1(out_cnn)
        out_lstm1 = torch.sum(
            out_lstm1.view(
                out_lstm1.shape[0], out_lstm1.shape[1], 2, self.hidden_lstm_1
            ),
            dim=2,
        )
        features, _ = self.lstm_2(out_lstm1)
        features = torch.sum(
            features.view(features.shape[0], features.shape[1], 2, self.hidden_lstm_2),
            dim=2,
        )  ## (batch_size , n_hidden ,1)
        if self.n_hidden == None:
            self.n_hidden = features.shape[1]
        return features


class TAE_decoder(nn.Module):
    """
    Class for temporal autoencoder decoder.
    filter_1 : filter size of the first convolution layer
    filter_lstm : hidden size of the lstm.
    """

    def __init__(self, n_hidden=64, pooling=8):
        super().__init__()

        self.pooling = pooling
        self.n_hidden = n_hidden

        # upsample
        self.up_layer = nn.Upsample(size=pooling)
        self.deconv_layer = nn.ConvTranspose1d(
            in_channels=self.n_hidden,
            out_channels=self.n_hidden,
            kernel_size=10,
            stride=1,
            padding=(10 - 1) // 2,
        )

    def forward(self, features):

        upsampled = self.up_layer(features)  ##(batch_size  , n_hidden , pooling)
        out_deconv = self.deconv_layer(upsampled)[:, :, : self.pooling].contiguous()
        out_deconv = out_deconv.view(out_deconv.shape[0], -1)
        return out_deconv


class TAE(nn.Module):
    """
    Class for temporal autoencoder.
    filter_1 : filter size of the first convolution layer
    filter_lstm : hidden size of the lstm.
    """

    def __init__(self, args, filter_1=50, filter_lstm=[50, 1]):
        super().__init__()

        self.pooling = int(args.pool)
        self.filter_1 = filter_1
        self.filter_lstm = filter_lstm

        self.tae_encoder = TAE_encoder(
            filter_1=self.filter_1,
            filter_lstm=self.filter_lstm,
            pooling=self.pooling,
        )
        n_hidden = self.get_hidden(args.serie_size, args.device)
        args.n_hidden = n_hidden
        self.tae_decoder = TAE_decoder(n_hidden=args.n_hidden, pooling=self.pooling)

    def get_hidden(self, serie_size, device):
        a = torch.randn((1, 1, serie_size)).to(device)
        test_model = TAE_encoder(
            filter_1=self.filter_1,
            filter_lstm=self.filter_lstm,
            pooling=self.pooling,
        ).to(device)
        with torch.no_grad():
            _ = test_model(a)
        n_hid = test_model.n_hidden
        del test_model, a
        gc.collect()
        torch.cuda.empty_cache()
        return n_hid

    def forward(self, x):

        features = self.tae_encoder(x)
        out_deconv = self.tae_decoder(features)
        return features.squeeze(2), out_deconv


class HeatMapNet(nn.Module):
    """
    Create HeatMapNet model
    """

    def __init__(self, args) -> None:
        super().__init__()

        self.n_heatmap_filters = args.n_clusters
        self.input_size = args.n_hidden
        self.total_serie_size = args.serie_size
        self.pooling = int(args.pool)
        self.up_layer = nn.Upsample(scale_factor=self.pooling)
        self.deconv_layer = nn.ConvTranspose1d(
            in_channels=1,
            out_channels=self.n_heatmap_filters,
            kernel_size=10,
            stride=1,
            padding=4,
        )  # (kernel_size - stride) // 2 for padding = same
        self.relu = nn.ReLU()
        self.global_avg_pool = nn.AdaptiveAvgPool1d(output_size=1)

        self.softmax = nn.Softmax(dim=1)

    def forward(self, Z):
        Z = self.up_layer(Z[:, None, :])
        Z = self.deconv_layer(Z)[:, :, : -self.pooling]

        Z = self.relu(Z)
        Q_hmap, _ = nn.Softmax(dim=2)(Z).max(
            axis=2
        )  # will test with global average pooling instead
        # Q_hmap = self.global_avg_pool(Z)

        Q_hmap = self.softmax(Q_hmap)

        return Z, Q_hmap


class ClusterNet(nn.Module):
    """
    class for the defintion of the DTC model
    path_ae : path to load autoencoder
    centr_size : size of the centroids = size of the hidden features
    alpha : parameter alpha for the t-student distribution.
    """

    def __init__(self, args):
        super().__init__()

        ## init with the pretrained autoencoder model
        self.tae = TAE(args)
        self.tae.load_state_dict(
            torch.load(args.path_weights_ae, map_location=args.device)
        )
        ## clustering model
        self.alpha_ = args.alpha
        self.centr_size = args.n_hidden
        self.n_clusters = args.n_clusters
        self.device = args.device
        self.similarity = args.similarity

        if args.heatmap:
            self.heatmap = HeatMapNet(args)
        else:
            self.heatmap = None

    def init_centroids(self, x):
        """
        This function initializes centroids with agglomerative clustering
        + complete linkage.
        """
        z, _ = self.tae(x.squeeze().unsqueeze(1).detach())
        z_np = z.detach().cpu()

        assignements = AgglomerativeClustering(
            n_clusters=self.n_clusters, linkage="complete", affinity="precomputed"
        ).fit_predict(compute_similarity(z_np, z_np, similarity=self.similarity))

        centroids_ = torch.zeros((self.n_clusters, self.centr_size), device=self.device)

        for cluster_ in range(self.n_clusters):
            index_cluster = [
                k for k, index in enumerate(assignements) if index == cluster_
            ]
            centroids_[cluster_] = torch.mean(z.detach()[index_cluster], dim=0)

        self.centroids = nn.Parameter(centroids_)

    def forward(self, x):

        z, x_reconstr = self.tae(x)
        z_np = z.detach().cpu()

        similarity = compute_similarity(z, self.centroids, similarity=self.similarity)

        ## Q (batch_size , n_clusters)
        Q = torch.pow((1 + (similarity / self.alpha_)), -(self.alpha_ + 1) / 2)
        sum_columns_Q = torch.sum(Q, dim=1).view(-1, 1)
        Q = Q / sum_columns_Q

        ## P : ground truth distribution
        P = torch.pow(Q, 2) / torch.sum(Q, dim=0).view(1, -1)
        sum_columns_P = torch.sum(P, dim=1).view(-1, 1)
        P = P / sum_columns_P
        if self.heatmap is not None:
            hmap, hmap_probs = self.heatmap(z)
            return z, x_reconstr, Q, P, hmap, hmap_probs
        return z, x_reconstr, Q, P


class ClusterNetSmooth(nn.Module):
    """
    Deep clustering over smoothed time series
    centr_size : size of the centroids = size of the hidden features
    alpha : parameter alpha for the t-student distribution.
    """

    def __init__(self, args):
        super().__init__()

        ## clustering model
        self.alpha_ = args.alpha
        self.centr_size = args.n_hidden
        self.n_clusters = args.n_clusters
        self.device = args.device
        self.similarity = args.similarity
        self.window_size = args.window_size
        if args.heatmap:
            self.heatmap = HeatMapNet(args)
        else:
            self.heatmap = None

    def init_centroids(self, x):
        """
        This function initializes centroids with agglomerative clustering
        + complete linkage.
        """
        z_np = x.detach().cpu()

        assignements = AgglomerativeClustering(
            n_clusters=self.n_clusters, linkage="complete", affinity="precomputed"
        ).fit_predict(compute_similarity(z_np, z_np, similarity=self.similarity))

        centroids_ = torch.zeros((self.n_clusters, self.centr_size), device=self.device)

        for cluster_ in range(self.n_clusters):
            index_cluster = [
                k for k, index in enumerate(assignements) if index == cluster_
            ]
            centroids_[cluster_] = torch.mean(x.detach()[index_cluster], dim=0)

        self.centroids = nn.Parameter(centroids_)

    def forward(self, x):
        similarity = compute_similarity(
            x.squeeze(), self.centroids, similarity=self.similarity
        )

        ## Q (batch_size , n_clusters)
        Q = torch.pow((1 + (similarity / self.alpha_)), -(self.alpha_ + 1) / 2)
        sum_columns_Q = torch.sum(Q, dim=1).view(-1, 1)
        Q = Q / sum_columns_Q

        ## P : ground truth distribution
        P = torch.pow(Q, 2) / torch.sum(Q, dim=0).view(1, -1)
        sum_columns_P = torch.sum(P, dim=1).view(-1, 1)
        P = P / sum_columns_P
        if self.heatmap is not None:
            hmap, hmap_probs = self.heatmap(x.squeeze())
            return Q, P, hmap, hmap_probs
        return Q, P
