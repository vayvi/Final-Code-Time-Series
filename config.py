import argparse


def get_arguments():
    parser = argparse.ArgumentParser()
    # data args
    parser.add_argument("--dataset_name", default="MoteStrain", help="dataset name")
    parser.add_argument("--path_data", default="data/{}", help="dataset name")

    # model args
    parser.add_argument(
        "--pool",
        required=True,
        help="pooling hyperparameter. Refer to the paper for each dataset's corresponding value",
    )
    parser.add_argument(
        "--similarity",
        required=True,
        choices=["SDTW", "COR", "EUC", "CID"],
        default="COR",
        help="The similarity type",
    )
    parser.add_argument(
        "--path_weights",
        default="models_weights/{}/",
        help="models weights",
    )
    parser.add_argument(
        "--n_clusters",
        type=int,
        default=2,
        help="Number of clusters , corresponding to the labels number",
    )
    parser.add_argument(
        "--window_size",
        type=int,
        default=7,
        help="Number of clusters , corresponding to the labels number",
    )

    parser.add_argument(
        "--alpha",
        type=int,
        default=1,
        help="alpha hyperparameter for DTC model",
    )
    # training args

    parser.add_argument("--batch_size", default=256, help="batch size")
    parser.add_argument(
        "--epochs_ae",
        type=int,
        default=10,
        help="Epochs number of the autoencoder training",
    )
    parser.add_argument(
        "--max_epochs",
        type=int,
        default=50,
        help="Maximum epochs numer of the full model training",
    )

    parser.add_argument(
        "--max_patience",
        type=int,
        default=5,
        help="The maximum patience for DTC training , above which we stop training.",
    )

    parser.add_argument(
        "--lr_ae",
        type=float,
        default=1e-2,
        help="Learning rate of the autoencoder training",
    )
    parser.add_argument(
        "--step_size",
        type=int,
        default=1,
    )
    parser.add_argument(
        "--smooth",
        action="store_true",
    )
    parser.add_argument(
        "--lr_cluster",
        type=float,
        default=1e-2,
        help="Learning rate of the full model training",
    )
    parser.add_argument(
        "--momentum",
        type=float,
        default=0.9,
        help="SGD momentum for the full model training",
    )
    parser.add_argument("--heatmap", action="store_true")
    parser.add_argument(
        "--initial_heatmap_weight",
        type=float,
        default=0.1,
        help="heatmap alpha weight initialy",
    )

    parser.add_argument(
        "--finetune_heatmap_for_last_epochs",
        type=float,
        default=10,
        help="number of last epochs for finetuning over heatmaps",
    )

    parser.add_argument(
        "--finetuning_heatmap_weight",
        type=float,
        default=0.9,
        help="heatmap alpha weight in finetuning",
    )

    parser.add_argument(
        "--no_autoencoder",
        action="store_true",
        help="Use when you want to perform deep clustering without autoencoder",
    )

    return parser
