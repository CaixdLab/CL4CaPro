# Contrastive Learning for Predicting Cancer Prognosis Using Gene Expression Values

## Overview of the computational pipeline
This codebase on [GitHub](https://github.com/CaixdLab/CL4CaPro) contains all source codes for training and testing the contrastive learning (CL) models for predicting cancer prognosis using gene expression values developed in the paper cited at the end of this document. The models trained with the RNAs-seq and clinical data from The Cancer Genome Atlas (TCGA) are available on [OneDrive](https://miamiedu-my.sharepoint.com/:f:/r/personal/x_cai_miami_edu/Documents/CaixdLab/CL4CaPro/CL4CaPro_Models?csf=1&web=1&e=mT3Z35), and codes for using these models are available in both this codebase and OneDrive. This codebase also includes the codes for validating the models trained with TCGA lung cancer and prostate cancer using two independent datasets: CPTAC-3 lung cancer dataset and DKFZ prostate cancer dataset. Moreover, this codebase contains the codes for training and tesing prognositic models using the 21 genes of Oncotype DX and all genes in a microarray dataset, as described in the paper. 


### Installation

Install our code and necessary Pyton modules on your system using the following scirpt. You may also need to install additional dependencies if not already present. 

```bash
# Clone the repository (if applicable)
git clone https://github.com/CaixdLab/CL4CaPro
cd CL4CaPro

# Install required Python packages based on your OS through anaconda (if any)
conda env create -f environment_W.yml (For Windows OS, we tested on Windows 10)
conda env create -f environment_L.yml (For Linux OS, we tested on Ubuntu 22.04.2 LTS)

# Activate corresponding conda env
conda activate CL4CaPro_L/CL4CaPro_W
```
Download datasets and models
<ul style="list-style-type:disc">
  <li>TCGA datasetsa </li>
  
  Download TCGA RNA-seq dataset  from the [GDC website](https://gdc.cancer.gov/about-data/publications/pancanatlas) or from our OneDrive folder [TCGAdata](https://miamiedu-my.sharepoint.com/:f:/r/personal/x_cai_miami_edu/Documents/CaixdLab/CL4CaPro/CL4CaPro_Models/TCGA?csf=1&web=1&e=IZYpyp). The RNA-seq data file is EBPlusPlusAdjustPANCAN_IlluminaHiSeq_RNASeqV2.geneExp.tsv and the clinical data file is TCGA-CDR-SupplementalTableS1.xlsx. 
  <li>CPTAC-3 dataset </li>
  
  Download CPTAC-3 RNA-seq and clinical datasets from [GDC data portal](https://portal.gdc.cancer.gov/projects/CPTAC-3). 
  <li>DKFZ prostate dataset </li>
  
  Download DKFZ RNA-seq and cinical datasets from [cBioPortal](https://www.cbioportal.org/).
  <li>Breast cancer microarray and clinical datasets <li>
  <li> CL-based Classifiers and CLCox models <li>
</ul>



   

### Train and test CL-based classifiers or Cox models
As described in the paper, we used two approaches to predict the proganosis of a cancer patient. In the first approach, we trained a XGBoost classifer to categorize a cancer patient into either a low-risk group of recurrence or a high-risk group of recurrence. In the second approach, we trained a Cox proportional hazards model to predict the hazards ratio. In both approaches, we first trained a CL-module that learns feature representations from cancer transcriptomes and then trained a classifier or a Cox model using the learned features. 

[*TrainCL4CaPro.ipynb*](https://github.com/CaixdLab/CL4CaPro/blob/main/TrainCL4CaPro.ipynb) notebook implements the pipeline of training and testing both the CL-based XGBoost (CL_XGBoost) classifier and the CL-based Cox (CLCox) model. The pipleline includes the following main steps
<ul style="list-style-type:disc">
  <li>Load TCGA data</li>
  <li>Train a CL model by calling the script Auto_Train_GPU.py, which in turn calls main_CLCP.py for training the CL model and GenerateFeatures.py for extracting features from the CL model. </li>
  <li>Train and test a XGBoost classifier by calling Classifier_method.py or a Cox model by calling Cox_methods.py</li>
</ul>
As described in the paper, the following three Cox models are trained: Cox-EN, Cox-XGB, and Cox-nnet, when Cox_methods.py is called.  To train the XGBoost classifer and the Cox-XGB model, the Python XGBoost module needs to be installed, which should have been done if the instllation script is run successfully. The Cox-nnet was downloaded from the following website https://github.com/xinformatics/coxnnet_py3/tree/master, and it should have been installed if the installation script is run successfully.

 
### Use the trained models 

CL-based classifiers for 18 types of cancer and CLCox models for 19 types of cancer trained with the TCGA data are available at [OneDrive](https://miamiedu-my.sharepoint.com/:f:/r/personal/x_cai_miami_edu/Documents/CaixdLab/CL4CaPro/CL4CaPro_Models?csf=1&web=1&e=mT3Z35). Given a set of cancer RNA-Seq transcriptomes, one can use the script [*PredictThroughClassifierModel.ipynb*](https://github.com/CaixdLab/CL4CaPro/blob/main/PredictThroughClassifierModel.ipynb) to input the transcriptome of each cancer patient to the CL-based classification model and categorize each patient into a low- or high-risk group of recurrence, or use the script [*PredictThroughCoxModel.ipynb*](https://github.com/CaixdLab/CL4CaPro/blob/main/PredictThroughCoxModel.ipynb) to input the transcriptome of each cancer patient to each of three CLCox models (CLCox-EN, Cox-XGB, and Cox-nnet) to predict the harzards ratio. 

If the information of progression free interval (PFI) of each patient in the RNA-Seq data set is available, the performance metrics of the classifier including the prediction acuracy, receiver operating characteristic (ROC) curve, and the area under the ROC curve (AUC) can be calculated using [*PredictThroughClassifierModel.ipynb*](https://github.com/CaixdLab/CL4CaPro/blob/main/PredictThroughClassifierModel.ipynb), and the performance metrics of Cox models including c-index and integrated Brier score (IBS) can be calculated using [*PredictThroughCoxModel.ipynb*](https://github.com/CaixdLab/CL4CaPro/blob/main/PredictThroughCoxModel.ipynb). 

## Replicate the results in the paper

### Models trained with TCGA data
The results for the CL-based classifiers in Figures 2-3 in the paper and for the CLCox models in Figure 4,  Extended Figures 1 and 4, and in Extended Tables 1 and 2 can be replicated as follows.   

To replicate our validation results on the TCGA dataset, users can specify a single cancer type or a group of cancer types in the *Build Data for given cancer* section of [*TrainCL4CaPro.ipynb*](https://github.com/CaixdLab/CL4CaPro/blob/main/TrainCL4CaPro.ipynb) notebook to train a personalized model using either a specific seed or a series of seeds. Following training, the model along with the contrastive learning features it generates will be automatically saved to your local directory. Subsequently, the performance metrics of the classifier, such as prediction accuracy, receiver operating characteristic (ROC) curve, and area under the ROC curve (AUC), can be determined using [*Classifier_method.py*](https://github.com/CaixdLab/CL4CaPro/blob/main/Classifier_method.py). Additionally, the performance metrics for Cox models, including the concordance index (c-index) and integrated Brier score (IBS), can be assessed using [*Cox_method.py*](https://github.com/CaixdLab/CL4CaPro/blob/main/Cox_methods.py).

### Validation with CPTAC-3 & DKFZ datasets

The dataset is in the GDC data portal: https://portal.gdc.cancer.gov/projects. In the Search-Projects search box at the upper left corner, type in cptac. Then both clinical data and gene expression data can be downloaded.

The mapped CPTAC-3 and DKFZ datasets are available at [CPTAC-3&DKFZ folder](https://github.com/CaixdLab/CL4CaPro/tree/main/CPTAC-3%26DKFZ) based on our supply mapping table [gene_dict_sample.csv](https://github.com/CaixdLab/CL4CaPro/blob/main/CPTAC-3/gene_dict_sample.csv). 
If you're interested in mapping the CPTAC-3 or DKFZ dataset independently, you can proceed with the steps outlined below: 1) The gene mapping dictionary can be created using the [*CPTAC3_mapping_gen.py*](https://github.com/CaixdLab/CL4CaPro/blob/main/CPTAC3_mapping_gen.py) script. 2) It may be necessary to manually verify certain mismatches between official symbol IDs and geneIDs due to updates in versions over recent decades. 3) Once the mapping ID dictionary is finalized, the [*CPTAC3_map.py*](https://github.com/CaixdLab/CL4CaPro/blob/main/CPTAC3_map.py) script can be utilized to produce the CPTAC-3 dataset. 

The mapped dataset is then applied to validate models trained on TCGA data. Subsequently, the [*CPTAC3&DKFZ.ipynb*](https://github.com/CaixdLab/CL4CaPro/blob/main/CPTAC3%26DKFZ.ipynb) notebook can generate CPTAC-3 features using TCGA-trained CL models to assess the performance of CPTAC-3 and DKFZ validation.

### Comparison of Cox models with 16 Oncotype DX genes and all 13,235 genes in a microarray dataset

The expression data, containing 22,268 features and 947 samples from the study by Zhao, Xi, et al., "Systematic assessment of prognostic gene signatures for breast cancer shows distinct influence of time and ER status," published in BMC Cancer (volume 14, article 211, March 19, 2014, DOI:10.1186/1471-2407-14-211), is accessible for download at: https://filetransfer.abbvie.com/w/f-0572ba21-9252-48e3-b8ac-ad36ab1c4feb or in our OneDrive as [Affy947.RDS](https://miamiedu-my.sharepoint.com/:u:/r/personal/x_cai_miami_edu/Documents/CaixdLab/CL4CaPro/CL4CaPro_Models/AffyDataset/Affy947.RDS?csf=1&web=1&e=NKqv7B).

Upon downloading the Affymetrix data, gene identification data can be produced by aligning the Affymetrix data using the GPL96 platform, available at https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GPL96. Subsequently, features relevant to Oncotype DX can be isolated from the gene identification dataset, enabling the performance of validation experiments. Detailed procedures for these validation experiments are documented at [OncotypeDX](https://github.com/CaixdLab/CL4CaPro/blob/main/OncotypeDX.ipynb). And all required files in the notebook are located at [OncotypeDX folder](https://github.com/CaixdLab/CL4CaPro/tree/main/OncotypeDX) and [OneDrive](https://miamiedu-my.sharepoint.com/:f:/r/personal/x_cai_miami_edu/Documents/CaixdLab/CL4CaPro/CL4CaPro_Models?csf=1&web=1&e=mT3Z35).

## Usage of Individual Scripts

###  *main_CLCP.py*
Train supervised CL models

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

###  *GenerateFeatures.py*
Extract features from a CL model.

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
Train XGBoost classifiers using the training data and compute classification accuracy and AUC using the test data

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
Train Cox models using training data and compute C-index and IBS using test data

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



## Citation
If you find our research useful, please cite our paper as follows:

@article{sun2023contrastive,
  title={Contrastive Learning for Predicting Cancer Prognosis Using Gene Expression Values},
  author={Sun, Anchen and Chen, Zhibin and Cai, Xiaodong},
  journal={arXiv preprint arXiv:2306.06276},
  year={2023}
}

## Brief information of the paper
### Title
Contrastive Learning for Predicting Cancer Prognosis Using Gene Expression Values

### Authors
- Anchen Sun
- Elizabeth J. Franzmann
- Zhibin Chen
- Xiaodong Cai

### Abstract
Several artificial neural networks (ANNs) have recently been developed to predict the prognosis of various types of cancer based on the tumor transcriptome. However, they have not shown significantly better performance than the traditional Cox proportional hazards regression model.  Recent advancements in image classification have demonstrated that contrastive learning (CL) can aid in further learning tasks by acquiring good feature representation from a limited number of data samples. In this paper, we applied CL to tumor transcriptomes and clinical data to learn feature representations in a low-dimensional space. We then utilized these learned features to train a  classifier to categorize tumors into a high- or low-risk group of recurrence.  Using data from The Cancer Genome Atlas (TCGA), we demonstrated that CL can significantly improve classification accuracy. Specifically, our CL-based classifiers achieved an area under the receiver operating characteristic curve (AUC) greater than 0.8 for 14 types of cancer, and an AUC greater than 0.9 for 2 types of cancer.  We also developed CL-based Cox (CLCox) models  for predicting cancer prognosis.  Our CLCox models trained with the TCGA data outperformed existing methods significantly in predicting the prognosis of 19 types of cancer under consideration.  The performance of CLCox models and CL-based classifiers trained with TCGA lung and prostate cancer data were validated using the data from two independent cohorts.  We also show that the CLCox model trained with the whole transcriptome  significantly outperforms  the Cox model trained with the 21 genes of  Oncotype DX  that is in clinical use for breast cancer patients.  CL-based classifiers and CLCox models for 19 types of cancer   are publicly available and can be used to  predict  cancer prognosis using the RNA-seq transcriptome of  an individual tumor.  Python codes for model training and testing are also publicly accessible, and can be applied to train new CL-based models using gene expression data of tumors. 
