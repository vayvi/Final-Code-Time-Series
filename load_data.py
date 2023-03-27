import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import DataLoader, Dataset
from tslearn.datasets import UCR_UEA_datasets
from tslearn.preprocessing import TimeSeriesScalerMeanVariance

from utils import smooth_time_series


def load_ucr(dataset, scale):
    """
    Load ucr dataset.
    Taken from https://github.com/FlorentF9/DeepTemporalClustering/blob/4f70d6b24722bd9f8331502d9cae00d35686a4d2/datasets.py#L18
    """
    X_train, y_train, X_test, y_test = UCR_UEA_datasets().load_dataset(dataset)
    X = np.concatenate((X_train, X_test))
    y = np.concatenate((y_train, y_test))
    if dataset == "HandMovementDirection":  # this one has special labels
        y = [yy[0] for yy in y]
    y = LabelEncoder().fit_transform(y)  # sometimes labels are strings or start from 1
    assert y.min() == 0  # assert labels are integers and start from 0
    # preprocess data (standardization)
    if scale:
        X = TimeSeriesScalerMeanVariance().fit_transform(X)
    return X, y


def load_spx(dataset, scale, returns=False):
    """
    Load spx dataset. From local csv.
    """
    smoothed_price = pd.read_csv("../data/SPX_Prices.csv", index_col=0)
    if returns:
        returns = smoothed_price.pct_change()
        X = returns.fillna(0).to_numpy().T
    else:
        X = smoothed_price.to_numpy().T
    y = pd.read_csv("../data/SPX_Sectors.csv", index_col=0).to_numpy()
    y = LabelEncoder().fit_transform(y)  # sometimes labels are strings or start from 1
    assert y.min() == 0  # assert labels are integers and start from 0
    # preprocess data (standardization)
    if scale:
        X = TimeSeriesScalerMeanVariance().fit_transform(X)
    return X, y


def load_data(dataset_name, all_ucr_datasets, scale=True):
    """
    args :
        dataset_name : a string representing the dataset name.
        all_ucr_datasets : a list of all ucr datasets present in tslearn UCR_UEA_datasets
        scale : a boolean that represents whether to scale the time series or not.
    return :
        X : time series , scaled or not.
        y : the labels ( binary in our case) . s
    """
    if dataset_name in all_ucr_datasets:
        return load_ucr(dataset_name, scale)
    elif dataset_name == "SPX":
        return load_spx(dataset_name, scale)
    elif dataset_name == "SPX_RET":
        return load_spx(dataset_name, scale, returns=True)
    else:
        print(
            "Dataset {} not available! Available datasets are UCR/UEA univariate and multivariate datasets.".format(
                dataset_name
            )
        )
        exit(0)


class CustomDataset(Dataset):
    def __init__(self, time_series, labels):
        """
        This class creates a torch dataset.

        """
        self.time_series = time_series
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):

        time_serie = torch.tensor(self.time_series[idx]).squeeze().unsqueeze(0)
        label = torch.tensor(self.labels[idx])

        return (time_serie, label)


def get_loader(args):
    """returns dataloader for a ucr dataset

    Args:
        args: argparse output

    Returns:
        trainloader and dataset
    """

    ucr = UCR_UEA_datasets()
    all_ucr_datasets = ucr.list_datasets()
    X_scaled, y = load_data(args.dataset_name, all_ucr_datasets)

    if args.smooth:
        X_scaled = smooth_time_series(X_scaled, args.window_size, args.step_size)
    # create dataset
    trainset = CustomDataset(X_scaled, y)

    ## create dataloader
    trainloader = DataLoader(
        trainset, batch_size=args.batch_size, shuffle=True, num_workers=2
    )
    args.serie_size = X_scaled.shape[1]
    return trainloader, X_scaled
