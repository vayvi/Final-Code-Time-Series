# MVA2023 - ML for Time Series - Mini Project

# Paper

Analysis of the Deep-Temporal Clustering model , presented in the paper :
Madiraju, N. S., Sadat, S. M., Fisher, D., & Karimabadi, H. (2018). Deep Temporal Clustering : Fully Unsupervised Learning of Time-Domain Features. <http://arxiv.org/abs/1802.01059>

# Our work

We present in this repository an ablation study of the DTC model.
Our main additions to the code are :

* Added the heatmap network.
* Added Soft-DTW Similarity.
* Changed the execution script to support more evaluation metrics (ROC-AUC, ARI, AMI).
* Added the different analysis notebooks (Exploratory Data Analysis and Ablation study).
* Added the SPX-dataset (not included in the final report).

# Credits

Our code is heavily inspired by the the code in <https://github.com/HamzaG737/Deep-temporal-clustering>.

## Usage

### Installation

Before running the training command, ensure that you have installed all required dependencies listed in the requirements.txt file using the following command:

```shell
pip install -r requirements.txt
```

### Data analysis

We have provided a Jupyter notebook `Data_analysis.ipynb` that performs a comprehensive analysis of the datasets that we worked with. This notebook contains various statistics, visualizations, and exploratory data analysis techniques to help you gain insights into the dataset.

### Training

To train the model, run the following command in your terminal:

```shell
python3 train.py --similarity <similarity_arg> --pool <pool_arg> --dataset_name <dataset_name>
```

Replace <similarity_arg> with the similarity metric you want to use for training, <pool_arg> with the pooling parameter you want to use, and <dataset_name> with the name of the dataset you want to use for training.

For example, to train the model on the MoteStrain dataset with the Euclidean similarity metric and a pooling parameter of 4, run the following command:

```shell
python3 train.py --similarity EUC --pool 4 --dataset_name MoteStrain
```

This train.py file should return the ROC score corresponding to the training parameters.

Note that the similarity and pool arguments are required. You can find a full list of available arguments, including the dataset name, in the config.py file.

### Reproducibility

In order to reproduce the papers' results, you can run the following line of code:

```shell
python3 reproduce_exps.py --similarity <similarity_arg> --pool <pool_arg> --dataset_name <dataset_name>
```

Example notebooks are provided and are structured as follows:

<!-- not organized, maybe we can have bullet points with notebooks and small descriptions?  -->
* A data analysis notebook with various statistics over the UCR datasets: `Data_analysis.ipynb`
* A training example along with heatmap visualization for:
  * The original paper's method, Deep Temporal Clustering (DTC), in `DTC_heatmap.ipynb`
  * Our Deep Clustering without autoencoding method in `Deep_clustering_heatmap.ipynb`
* A comprehensive notebook, `Ablation_studies.ipynb` with all of our ablation studies and comparisons with the following sections:
  * A baseline using K-means clustering over a smoothed time series.
  * Our deep clustering without autoencoding over a smoothed time series.
  * The fully pretrained AutoEncoder only followed by spectral clustering over the initial time series.
  * The joint AutoEncoder and spectral clustering model over the inital time series.

The autoencoder and clustering models weights will be saved in a **models_weights** directory. Also the train.py file returns the ROC score corresponding to the training parameters.
