import os

import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import adjusted_rand_score, roc_auc_score

from config import get_arguments
from load_data import get_loader
from models import TAE, ClusterNet, ClusterNetSmooth


def pretrain_autoencoder(trainloader, args, verbose=True):
    """
    function for the autoencoder pretraining
    """
    print("Pretraining autoencoder... \n")

    ## define TAE architecture
    tae = TAE(args)
    tae = tae.to(args.device)

    ## MSE loss
    loss_ae = nn.MSELoss()
    ## Optimizer
    optimizer = torch.optim.Adam(tae.parameters(), lr=args.lr_ae)
    tae.train()
    with open(f"data/{args.dataset_name}/log_{args.dataset_name}.txt", "a") as f:

        for epoch in range(args.epochs_ae):
            all_loss = 0
            for batch_idx, (inputs, _) in enumerate(trainloader):
                inputs = inputs.type(torch.FloatTensor).to(args.device)
                optimizer.zero_grad()
                z, x_reconstr = tae(inputs)
                loss_mse = loss_ae(inputs.squeeze(1), x_reconstr)

                loss_mse.backward()
                all_loss += loss_mse.item()
                optimizer.step()
            if verbose:
                loss_str = f"Pretraining autoencoder loss for epoch {epoch} is : {all_loss / (batch_idx + 1)} \n"
                print(loss_str)
                f.write(loss_str)

    print("Ending pretraining autoencoder. \n")
    # save weights
    torch.save(tae.state_dict(), args.path_weights_ae)


def initalize_centroids(X, args, model):
    """
    Function for the initialization of centroids.
    """
    X_tensor = torch.from_numpy(X).type(torch.FloatTensor).to(args.device)
    model.init_centroids(X_tensor)


def kl_loss_function(input, pred):
    out = input * torch.log((input) / (pred))
    return torch.mean(torch.sum(out, dim=1))


def train_ClusterNET(epoch, trainloader, model, optimizer_clu, loss1, args, verbose):
    """
    Function for training one epoch of the DTC
    """
    model.train()
    train_loss = 0
    all_preds, all_gt = [], []
    # use k nn on embedding of auto encoder
    #
    with open(f"data/{args.dataset_name}/log_{args.dataset_name}.txt", "a") as f:
        for batch_idx, (inputs, labels) in enumerate(trainloader):
            inputs = inputs.type(torch.FloatTensor).to(args.device)
            # print(inputs.shape)
            all_gt.append(labels.cpu().detach())
            optimizer_clu.zero_grad()
            if args.heatmap:
                z, x_reconstr, Q, P, _, hmap_probs = model(inputs)
            else:
                z, x_reconstr, Q, P = model(inputs)
            loss_mse = loss1(inputs.squeeze(), x_reconstr)  # TODO check dimensions here
            loss_KL = kl_loss_function(P, Q)

            if args.heatmap:
                loss_hmap = kl_loss_function(P, hmap_probs)
                if epoch >= (args.max_epochs - args.finetune_heatmap_for_last_epochs):
                    hmap_weight = args.finetuning_heatmap_weight
                else:
                    hmap_weight = args.initial_heatmap_weight
                total_loss = (
                    loss_mse
                    + 100 * (1 - hmap_weight) * loss_KL
                    + 100 * hmap_weight * loss_hmap
                )
            else:
                total_loss = loss_mse + loss_KL

            total_loss.backward()
            optimizer_clu.step()

            preds = torch.max(Q, dim=1)[1]
            all_preds.append(preds.cpu().detach())
            train_loss += total_loss.item()
        if verbose:
            loss_str = (
                f"For epoch {epoch} Loss is :  {train_loss / (batch_idx + 1):.3f} \n"
            )
            print(loss_str)
            f.write(loss_str)

    all_gt = torch.cat(all_gt, dim=0).numpy()
    all_preds = torch.cat(all_preds, dim=0).numpy()
    try:
        return (
            preds.detach().cpu().numpy(),
            max(
                roc_auc_score(all_gt, all_preds),
                roc_auc_score(all_gt, 1 - all_preds),
            ),
            train_loss / (batch_idx + 1),
        )
    except:
        return (
            preds.detach().cpu().numpy(),
            adjusted_rand_score(all_gt, all_preds),
            train_loss / (batch_idx + 1),
        )


def train_ClusterNET_smooth(epoch, trainloader, model, optimizer_clu, args, verbose):
    """
    Function for training one epoch of the DTC
    """
    model.train()
    train_loss = 0
    all_preds, all_gt = [], []
    # use k nn on embedding of auto encoder
    #
    with open(f"data/{args.dataset_name}/log_DC_{args.dataset_name}.txt", "a") as f:
        for batch_idx, (inputs, labels) in enumerate(trainloader):
            inputs = inputs.type(torch.FloatTensor).to(args.device)
            # print(inputs.shape)

            all_gt.append(labels.cpu().detach())
            optimizer_clu.zero_grad()
            if args.heatmap:
                Q, P, _, hmap_probs = model(inputs)
            else:
                Q, P = model(inputs)
            loss_KL = kl_loss_function(P, Q)

            if args.heatmap:
                loss_hmap = kl_loss_function(P, hmap_probs)
                if epoch >= (args.max_epochs - args.finetune_heatmap_for_last_epochs):
                    hmap_weight = args.finetuning_heatmap_weight
                else:
                    hmap_weight = args.initial_heatmap_weight
                total_loss = (
                    +100 * (1 - hmap_weight) * loss_KL + 100 * hmap_weight * loss_hmap
                )
            else:
                total_loss = loss_KL

            total_loss.backward()
            optimizer_clu.step()

            preds = torch.max(Q, dim=1)[1]
            all_preds.append(preds.cpu().detach())
            train_loss += total_loss.item()
        if verbose:
            loss_str = (
                f"For epoch {epoch} Loss is :  {train_loss / (batch_idx + 1):.3f} \n"
            )
            print(loss_str)
            f.write(loss_str)

    all_gt = torch.cat(all_gt, dim=0).numpy()
    all_preds = torch.cat(all_preds, dim=0).numpy()
    try:
        return (
            preds.detach().cpu().numpy(),
            max(
                roc_auc_score(all_gt, all_preds),
                roc_auc_score(all_gt, 1 - all_preds),
            ),
            train_loss / (batch_idx + 1),
        )
    except:
        return (
            preds.detach().cpu().numpy(),
            adjusted_rand_score(all_gt, all_preds),
            train_loss / (batch_idx + 1),
        )


def training_function(
    trainloader, X_scaled, model, optimizer_clu, loss1, args, verbose=True
):
    """
    function for training the DTC network.
    """
    ## initialize clusters centroids
    ## train clustering model
    max_roc_score = 0
    print("Training full model ...")
    for epoch in range(args.max_epochs):
        if args.no_autoencoder:
            preds, roc_score, train_loss = train_ClusterNET_smooth(
                epoch, trainloader, model, optimizer_clu, args, verbose=verbose
            )
        else:
            preds, roc_score, train_loss = train_ClusterNET(
                epoch, trainloader, model, optimizer_clu, loss1, args, verbose=verbose
            )
        patience = 0
        if roc_score > max_roc_score:
            max_roc_score = roc_score
            patience = 0
        else:
            patience += 1
            if patience == args.max_patience:
                break

    torch.save(model.state_dict(), args.path_weights_main)
    with open(f"data/{args.dataset_name}/log_{args.dataset_name}.txt", "a") as f:
        f.write(f"Max roc score is: {max_roc_score} \n")

    return max_roc_score


if __name__ == "__main__":

    parser = get_arguments()
    args = parser.parse_args()
    args.path_data = args.path_data.format(args.dataset_name)
    if not os.path.exists(args.path_data):
        os.makedirs(args.path_data)

    path_weights = args.path_weights.format(args.dataset_name)
    if not os.path.exists(path_weights):
        os.makedirs(path_weights)

    args.path_weights_ae = os.path.join(path_weights, "autoencoder_weight.pth")
    args.path_weights_main = os.path.join(path_weights, "full_model_weigths.pth")

    args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    trainloader, X_scaled = get_loader(args)
    for e in trainloader:
        for a in e:
            n_hidden = a.shape[-1]
            break
        break
    args.n_hidden = n_hidden
    if args.no_autoencoder:
        model = ClusterNetSmooth(args)
        loss1 = None

    else:
        pretrain_autoencoder(trainloader, args)
        model = ClusterNet(args)
        loss1 = nn.MSELoss()

    model = model.to(args.device)
    initalize_centroids(X_scaled, args, model)
    optimizer_clu = torch.optim.SGD(
        model.parameters(), lr=args.lr_cluster, momentum=args.momentum
    )
    max_roc_score = training_function(
        trainloader, X_scaled, model, optimizer_clu, loss1, args
    )

    print(
        f"maximum roc score for {args.dataset_name} with {args.similarity} (no autoencoder {args.no_autoencoder}) (smoothing {args.smooth}) is {max_roc_score}"
    )
