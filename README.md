# Contrastive Learning for Predicting Cancer Prognosis Using Gene Expression Values

## Overview of the computational pipeline
This codebase on GitHub at [https://github.com/CaixdLab/CL4CaPro] contains all source codes for training and testing the contrastive learning (CL) models for predicting cancer prognosis using gene expression values developed in the paper cited at the end of this document. The models trained with the RNAs-seq and clinical data from The Cancer Genome Atlas (TCGA) are available on [OneDrive](https://miamiedu-my.sharepoint.com/:f:/r/personal/x_cai_miami_edu/Documents/CaixdLab/CL4CaPro/CL4CaPro_Models?csf=1&web=1&e=mT3Z35), and codes for using these models are available in both this codebase and OneDrive. This codebase also includes the codes for validating the models trained with TCGA lung cancer and prostate cancer using two independent datasets: CPTAC-3 lung cancer dataset and DKFZ prostate cancer dataset.  



### Installation

Before running this script, ensure you have Python installed on your system. You may also need to install additional dependencies if not already present. 

```bash
# Clone the repository (if applicable)
git clone https://github.com/CaixdLab/CL4CaPro
cd CL4CaPro

# Install required Python packages based on your OS through anaconda (if any)
conda env create -f environment_W.txt (For Windows OS, we tested on Windows 10)
conda env create -f environment_L.txt (For Linux OS, we tested on Ubuntu 22.04.2 LTS)

# Activate corresponding conda env
conda activate CL4CaPro_L/CL4CaPro_W
```

## Usage of Individual Scripts

###  *main_CLCP.py*
Train Contrastive Learning models for cancer datasets

#### Training Configuration Guide

This guide provides detailed instructions on how to configure and use the training script for a machine learning model. By using command-line arguments, you can customize various aspects of the training process, such as epochs, batch size, learning rate, and much more. Below are the available options and their descriptions.

##### Basic Training Settings

- `--print_freq`: Set the print frequency to monitor the training process. Default is `1`.
- `--save_freq`: Determine how often to save the model. Default is `2000`.
- `--batch_size`: Specify the batch size for training. Default is `128`.
- `--num_workers`: Number of worker processes for loading data. Default is `16`.
- `--epochs`: Total number of training epochs. Default is `2000`.
- `--round`: Number of validation rounds. Default is `1`.
- `--os`: Specify the operating system for experimental purposes. Choose `W` for Windows or `L` for Linux. Default is `W`.
- `--gpu_device`: GPU device ID to use for training. Default is `0`.

##### Optimization

- `--learning_rate`: Initial learning rate. Default is `0.01`.
- `--lr_decay_epochs`: Epochs where learning rate decay should occur, specified as a list (e.g., '500,1000,1500,5000'). Default is `'500,1000,1500,5000'`.
- `--lr_decay_rate`: Learning rate decay rate. Default is `0.1`.
- `--l2_rate`: L2 normalization rate. Default is `0.01`.
- `--weight_decay`: Weight decay (L2 penalty). Default is `1e-4`.
- `--momentum`: Momentum for the optimizer. Default is `0.9`.
- `--lr_early_stop`: Learning rate value at which to stop training early. Default is `0`.
- `--epoch_early_stop`: Epoch number at which to stop training early. Default is `0`.
- `--split_class_num`: Number of classes to split the dataset into. Default is `8`.

##### Model and Dataset

- `--model`: Model to use. Default is `CLCP`.
- `--model_in_dim`: Input dimension for the model. Default is `16008`.
- `--model_n_hidden_1`: First hidden layer dimension. Default is `8196`.
- `--model_n_hidden_2`: Second hidden layer dimension. Default is `4096`.
- `--model_out_dim`: Output dimension of the model. Default is `2048`.
- `--feat_dim`: Feature dimension. Default is `1024`.
- `--dataset`: Dataset to use. Choices are `CancerRNA` or `path`. Default is `path`.
- `--data_folder`: Path to the custom dataset. Default is `None`.
- `--cancer_group`: Specify the cancer group. Default is `SM`.
- `--validation`: Dataset for validation. Default is `TCGA`.

##### Method

- `--method`: Method to use for training. Choices are `SupCon`, `CLCP`, `Test`, `CLCP`. Default is `CLCP`.
- `--task`: Task for training the model. Choices are `WholeTimeSeq` or `Risk`. Default is `WholeTimeSeq`.
- `--train_test`: Train-test split percentage. Default is `100`.
- `--split_seed`: Random seed for train-test splitting. Default is `0`.
- `--pooled`: Whether to pool the cancer groups. Default is `False`.

##### Temperature

- `--temp`: Temperature for the loss function. Default is `0.07`.

##### Other Settings

- `--cosine`: Use cosine annealing. This option does not take a value.
- `--syncBN`: Use synchronized batch normalization. This option does not take a value.
- `--warm`: Warm-up for large batch training. This option does not take a value.

To use these settings, add them as command-line arguments when running the script. For example:

```shell
python main_CLCP.py --dataset CancerRNA --model_in_dim {input_dim} --model_n_hidden_1 \
               {model_n_hidden_1} --model_out_dim {model_out_dim} --feat_dim {feat_dim} \
               --batch_size {batch_size} --train_test 80 --split_seed {seed} \
               --save_freq {train_epoch} --epochs {train_epoch} --learning_rate {lr} \
               --round 1 --lr_decay_epochs 0 --l2_rate {l2_rate} --split_class_num {split_class_num} \
               --os W --task WholeTimeSeq --cancer_group {cancer_group} --gpu_device {device} \
               --validation {validation} --epoch_early_stop 3000 --lr_early_stop 4.0
```

### Usage *GenerateFeatures.py*
Generate Contrastive Learning model features for Cox models

#### Feature Extraction Configuration Guide

This guide outlines how to configure the feature extraction process from trained models using command-line arguments. The script allows you to specify various parameters, including the layer from which to extract features, model dimensions, batch size, learning rates, and more. Below is a description of each option available.

##### Parameters

- `--layer_name`: Specify the name of the layer from which to extract features. Default is `layer1`.
- `--model_in_dim`: Input dimension of the model. Default is `16008`.
- `--dim_1_list`: First dimension test list. Default is `5196`.
- `--dim_2_list`: Second dimension test list. Default is `4096`.
- `--dim_3_list`: Third dimension test list. Default is `4096`.
- `--model_save_path`: Path where the model is saved. Default is `SupCLCP`.
- `--batch_size`: Batch size for feature extraction. Default is `110`.
- `--seed`: Random seed for reproducibility. Default is `1036`.
- `--learning_rate_list`: Test learning rate list. Only a single value of `0.00005` by default.
- `--l2_rate`: L2 normalization rate. Default is `0.01`.
- `--round`: Number of validation rounds. Default is `1`.
- `--split_class_num`: Number of classes into which the dataset is split. Default is `8`.
- `--cancer_group`: Specify the cancer group (e.g., SM, NGT, MSE, CCPRC, HC, GG, DS, LC). Default is `SM`.
- `--task`: Choose the task to train the model on. Options are `WholeTimeSeq` or `Risk`. Default is `WholeTimeSeq`.
- `--gpu_device`: GPU device ID for training the model. Default is `0`.
- `--validation`: Validation dataset to use. Default is `TCGA`.

##### Example Usage

To use these settings, add them as command-line arguments when executing your script. For example:

```shell
python GenerateFeatures.py --layer_name {layer name} --model_in_dim {input_dim} --dim_1_list {model_n_hidden_1} \
                           --dim_2_list {model_out_dim} --dim_3_list {feat_dim} --batch_size {batch_size} \
                           --l2_rate {l2_rate} --validation {validation} --seed {seed} --round 1 --gpu_device {device} \
                           --learning_rate_list {lr} --split_class_num {split_class_num} --task WholeTimeSeq \
                           --cancer_group {cancer_group}
```

###  *Classifier_method.py*
Train different classification models to analysis AUC

#### Analysis Configuration Guide

This guide provides instructions for configuring the analysis of features extracted from trained models. By specifying various command-line arguments, users can tailor the analysis process to their specific needs, focusing on aspects like the model save path, seed for reproducibility, cancer group, and computing resources. Below are the available options along with their descriptions.

##### Parameters

- `--model_save_path`: Path to the directory where the model is saved. This is where the script will look for the extracted features. The default path is `SupCLCP`.
- `--seed`: Random seed used for initializing the analysis to ensure reproducibility. The default value is `1036`.
- `--cancer_group`: Specifies the cancer group to be analyzed. It supports various groups such as SM, NGT, MSE, CCPRC, HC, GG, DS, LC, etc., with the default being `SM`.
- `--core`: The number of CPU cores reserved for running the script. This can be adjusted based on the available system resources to improve performance. The default value is `10`.

##### Example Usage

To perform an analysis with custom settings, you would use the command line to run your script with the desired options. For example:

```shell
python Classifier_method.py --cancer_group BRCA --seed {seed} --core 20 > PlotLog/{}.log &
```

###  *Cox_methods.py*
Train different Cox models to analysis C-index or IBS

#### Analysis Configuration Guide

This guide provides instructions for configuring the analysis of features extracted from trained models. By specifying various command-line arguments, users can tailor the analysis process to their specific needs, focusing on aspects like the model save path, seed for reproducibility, cancer group, and computing resources. Below are the available options along with their descriptions.

##### Parameters

- `--cancer_group`: Specifies the cancer group to be analyzed. It supports various groups such as SM, NGT, MSE, CCPRC, HC, GG, DS, LC, etc., with the default being `SM`.
- `--core`: The number of CPU cores reserved for running the script. This can be adjusted based on the available system resources to improve performance. The default value is `10`.
- `--pooled`: Specifies the cancer group is a pooled group. It supports bool True or False, with the default being `False`.
- `--os_flag`: Specify whether OS time is used or normal PFI time. It supports bool True or False, with the default being `False`.

##### Example Usage

To perform an analysis with custom settings, you would use the command line to run your script with the desired options. For example:

```shell
python Cox_methods.py --cancer_group BRCA --seed {seed} --core 20 > PlotLog/{}.log &
```

### Sample Model Training

[*TrainCL4CaPro.ipynb*](https://github.com/CaixdLab/CL4CaPro/blob/main/TrainCL4CaPro.ipynb) notebook is accessible on Github, offering comprehensive step-by-step instructions on how to effortlessly train a CL4CaPro model from the ground up and validate its performance.

### Public Models

CL-based classifiers and CLCox models are publicly available via: [OneDrive](https://miamiedu-my.sharepoint.com/:f:/r/personal/x_cai_miami_edu/Documents/CaixdLab/CL4CaPro/CL4CaPro_Models?csf=1&web=1&e=mT3Z35)

### Predict Through Public Models

Both the [*PredictThroughClassifierModel.ipynb*](https://github.com/CaixdLab/CL4CaPro/blob/main/PredictThroughClassifierModel.ipynb) and [*PredictThroughCoxModel.ipynb*](https://github.com/CaixdLab/CL4CaPro/blob/main/PredictThroughCoxModel.ipynb) notebooks are accessible on Github and OneDrive. Simply place them alongside our publicly shared models in the same directory. Then, launch the Jupyter notebook, and by adhering to the provided instructions, select the cancer type and set the input file to predict outcomes directly using our public models.

## Citation
If you find our research useful, please cite our paper as follows:

@article{sun2023contrastive,
  title={Contrastive Learning for Predicting Cancer Prognosis Using Gene Expression Values},
  author={Sun, Anchen and Chen, Zhibin and Cai, Xiaodong},
  journal={arXiv preprint arXiv:2306.06276},
  year={2023}
}

## Paper Briefly Intro

### Authors
- Anchen Sun
- Elizabeth J. Franzmann
- Zhibin Chen
- Xiaodong Cai

### Abstract
Predicting the prognosis of cancer using the tumor transcriptome is a significant challenge in the medical field. Although several Artificial Neural Networks (ANNs) have been developed for this purpose, their performance has not substantially surpassed that of the regularized Cox proportional hazards regression model. This paper introduces a novel approach by applying supervised Contrastive Learning (CL) to tumor gene expression and clinical data to generate meaningful low-dimensional feature representations. These representations are then utilized to train a Cox model, named CLCox, for the prognosis prediction of cancer. Our method demonstrated significant improvements over existing models across 19 types of cancer using data from The Cancer Genome Atlas (TCGA). Additionally, we developed CL-based classifiers to categorize tumors into different risk groups, achieving remarkable classification accuracy and area under the receiver operating characteristic curve (AUC) scores. This paper also includes validation of the proposed models with independent lung and prostate cancer cohorts.

### Introduction
This research aims to address the limitations of current Artificial Neural Network (ANN) models in predicting cancer prognosis by leveraging the power of Contrastive Learning (CL). CL has recently shown promising results in image classification tasks by efficiently learning feature representations from a limited dataset. By applying supervised CL to both tumor gene expression and clinical data, we were able to train a Cox model that significantly outperforms traditional methods.

### Methodology
We utilized data from The Cancer Genome Atlas (TCGA) to train our CL-based Cox model (CLCox) and classifiers. The methodology section details the process of applying supervised contrastive learning to generate low-dimensional feature representations and the subsequent training of the Cox model and classifiers using these features.

### Results
Our CLCox model exhibited superior performance in predicting the prognosis of 19 different cancer types, outperforming existing methods significantly. Additionally, our CL-based classifiers demonstrated high accuracy in tumor classification into different risk groups, with AUC scores exceeding 0.8 for 14 cancer types and 0.9 for 2 cancer types. The models' effectiveness was further validated using independent lung and prostate cancer datasets.

### Conclusion
This paper presents a groundbreaking approach to predicting cancer prognosis using supervised contrastive learning. The significant improvements in prognosis prediction and tumor classification underscore the potential of CL in medical applications. Future work will focus on exploring the application of CL to other types of medical data and further refining our models.
