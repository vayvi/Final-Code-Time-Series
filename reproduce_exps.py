import os


data = {
    "BeetleFly": 8,
    "BirdChicken": 8,
    "Computers": 10,
    "Earthquakes": 8,
    "MoteStrain": 4,
    "PhalangesOutlinesCorrect": 4,
    "ProximalPhalanxOutlineCorrect": 4,
    "ShapeletSim": 10,
    "ItalyPowerDemand": 4,
    "WormsTwoClass": 10,
#     "SPX" : 10,
#     "SPX_RET" : 10,
}


for dataset_name, pool in data.items():
    command = (
        f"python train.py --similarity SDTW --n_clusters 20 --max_epochs 100 --epochs_ae 100 --max_patience 150 --lr_cluster 0.001 --lr_ae 0.0001 --pool {pool} --dataset_name {dataset_name}"
    )
    os.system(command)