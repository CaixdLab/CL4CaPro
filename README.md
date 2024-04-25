# Contrastive Learning for Predicting Cancer Prognosis Using Gene Expression Values

## Overview of the computational pipeline
This codebase on [GitHub](https://github.com/CaixdLab/CL4CaPro) contains all source codes for training and testing the contrastive learning (CL) models for predicting cancer prognosis using gene expression values developed in the paper cited at the end of this document. The models trained with the RNAs-seq and clinical data from The Cancer Genome Atlas (TCGA) are available on [Box](https://miami.box.com/s/ylmvqynbtchx5xhof0quaeu9w62mxaca), and codes for using these models are available in this codebase. This codebase also includes the codes for validating the models trained with TCGA lung cancer and prostate cancer using two independent datasets: CPTAC-3 lung cancer dataset and DKFZ prostate cancer dataset. Moreover, this codebase contains the codes for training and tesing prognositic models using the 21 genes of Oncotype DX and all genes in a microarray dataset, as described in the paper. 


### Installation

Install our code and necessary Pyton modules on your system using the following script. You may also need to install additional dependencies if not already present. 
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
The system requirements are as follows:
<ul style="list-style-type:disc">
  <li>Python 3.8 and later</li>
   <li>CUDA 11.3 and later</li>
   <li>PyTorch 2.0 and later for the Linux version and Pytorch 1.11 and later for the Windows version</li>
</ul>
We trained models on two RTX 4090 24GB GPU cards with 8 threads. To optimize the speed, the number of threads in scripts Auto_Train_GPU.py and Auto_Train_AffyOnco_GPU.py may be adjusted according to the GPUs.  

### Download datasets and models
<ul style="list-style-type:disc">
  <li>TCGA dataset<br />
The TCGA dataset can be downloaded from the <a href="https://gdc.cancer.gov/about-data/publications/pancanatlas">GDC website</a> or from our Box folder <a href="https://miami.box.com/s/ylmvqynbtchx5xhof0quaeu9w62mxaca">TCGA data<a>. The RNA-seq data file is EBPlusPlusAdjustPANCAN_IlluminaHiSeq_RNASeqV2.geneExp.tsv,  and the clinical data file is TCGA-CDR-SupplementalTableS1.xlsx. Both files should be placed in the folder CL4CAPro.  TCGA datasets were used to train and test our models. </li>
  
  <li>CPTAC-3 dataset <br />
CPTAC-3 RNA-seq and clinical data  can be downloaded from <a href="https://portal.gdc.cancer.gov/projects/CPTAC-3">GDC data portal</a>, and data files should be placed in the folder CL4CAPro/CPTAC-3&DKFZ.  They were used to validate the lung cancer models trained with TCGA data. </li>

  <li>DKFZ prostate dataset <br /> 
DKFZ RNA-seq and cinical data can be downloaded from <a href="https://www.cbioportal.org">cBioPortal</a>, and data files should be placed in the folder CL4CAPro/CPTAC-3&DKFZ. They were used to validate prostate cancer models trained with TCGA data.  </li>

  <li>Breast cancer dataset <br />
  The breast cancer dataset contains microarray and clinical data of five datasets processed by the authours of the following paper:  <br />
Zhao, X., Rødland, E. A., Sørlie, T., Vollan, H. K. M., Russnes, H. G., Kristensen, V. N., ... & Børresen-Dale, A. L. (2014). Systematic assessment of prognostic gene signatures for breast cancer shows distinct influence of time and ER status. BMC cancer, 14, 1-12.<br />
The dataset can be downloaded from a <a href ="https://www.dropbox.com/scl/fi/f19cymyk5m9li9esnprpb/Affy947.RDS?rlkey=crwv3hpe2a7jwiku2ww3nl81b&dl=0">site</a> provided by the first authour of the paper or our <a href="https://miamiedu-my.sharepoint.com/:f:/r/personal/x_cai_miami_edu/Documents/CaixdLab/CL4CaPro?csf=1&web=1&e=OGT77f">OneDrive</a>. All files should be placed in the folder  CL4CAPro/OncotypeDX.  </li>

  <li> CL-based Classifiers and CLCox models  <br />
  CL-based classifiers for 18 types of cancer and CLCox models for 19 types of cancer trained with the TCGA data are available at <a href="https://miamiedu-my.sharepoint.com/:f:/r/personal/x_cai_miami_edu/Documents/CaixdLab/CL4CaPro/CL4CaPro_Models?csf=1&web=1&e=mT3Z35">OneDrive</a>. The whole folder CL4CaPro_Models including the two subfolders should be placed in the folder   CL4CAPro.   The file for each model is about 1GB.</li>
</ul>



   

### Train and test CL-based classifiers or Cox models
As described in the paper, we used two approaches to predict the proganosis of a cancer patient. In the first approach, we trained a XGBoost classifer to categorize a cancer patient into either a low-risk group of recurrence or a high-risk group of recurrence. In the second approach, we trained a Cox proportional hazards model to predict the hazards ratio. In both approaches, we first trained a CL-module that learns feature representations from cancer transcriptomes and then trained a classifier or a Cox model using the learned features. 

[*TrainCL4CaPro.ipynb*](https://github.com/CaixdLab/CL4CaPro/blob/main/TrainCL4CaPro.ipynb) notebook implements the pipeline of training and testing both the CL-based XGBoost (CL-XGBoost) classifier and the CL-based Cox (CLCox) model. The pipleline includes the following main steps
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
The AUCs and ROCs of the CL-based classifiers in Figures 2-3 in the paper, c-indexes and IBSs of the CLCox models in Figure 4,  Extended Figures 1 and 4, and Extended Tables 1 and 2 can be replicated using the notebook [*TrainCL4CaPro.ipynb*](https://github.com/CaixdLab/CL4CaPro/blob/main/TrainCL4CaPro.ipynb). The notebook uses TCGA data to implement the entire pipeline for training and testing CL-based clasifiers and CLCox models. See the comments in the notebook for the procedure of training and testing models.    


### Validation with CPTAC-3 & DKFZ datasets

#### Preprocessing and generating dataset files
The notebook [*CPTAC3_Preprocessing.ipynb*](https://github.com/CaixdLab/CL4CaPro/blob/main/CPTAC3_Preprocessing.ipynb), [*GenerateCPTAC3_Dataset.ipynb*](https://github.com/CaixdLab/CL4CaPro/blob/main/GenerateCPTAC3_Dataset.ipynb), [*DKFZ_Preprocessing.ipynb*](https://github.com/CaixdLab/CL4CaPro/blob/main/DKFZ_Preprocessing.ipynb), and [*GenerateDKFZ_Dataset.ipynb*](https://github.com/CaixdLab/CL4CaPro/blob/main/GenerateDKFZ_Dataset.ipynb) are available for user to preparing and generating CPTAC-3 and DKFZ datasets from the original downloaded files. They are working on the following tasks:

CPTAC3 and DKFZ Preprocssing:
<ul style="list-style-type:disc">
  <li>Load Gene Expression data and clinical data from the original downloaded files.</li>
  <li>Combine patients' Gene Expression data with their clinical records to create a comprehensive file.</li>
  <li>Standardize the Gene IDs/Symbols used in CPTAC-3 and DKFZ to those used by TCGA, and update this in the comprehensive file.</li>
</ul>

Generate CPTAC3 and DKFZ dataset:
<ul style="list-style-type:disc">
  <li>Select the necessary cases from the preprocessed comprehensive file.</li>
  <li>Normalize the data using housekeeper genes values.</li>
  <li>Create a cancer level data file for validating the CL-based models trained from TCGA datasets.</li>
</ul>

#### Run Validation
Place CPTAC-3 and DKFZ dataset files in the folder CL4CAPro/CPTAC-3&DKFZ. In the Jupyter notebook [CPTAC3&DKFZ.ipynb](https://github.com/CaixdLab/CL4CaPro/blob/main/CPTAC3%26DKFZ.ipynb), follow the comments there to set proper values of two variables Task and Cancer, and then run the code. The notebook uses the TCGA models in the folder CL4CaPro_Models. Since one model is saved for each type of cancer, the validation result for that model will be produced. In Figure 5 of the paper, validation results of 40 models of each type of cancer obtained from 40 random splits of the data are presented. To produce the validation results for 40 models, one needs to train 40 models using TrainCL4CaPro.ipynb with 40 default random seeds already in the code. 



### Comparison of Cox models with 16 Oncotype DX genes and all 13,235 genes in a breast cancer microarray dataset
Download the breast cancer dataset from the [Box](https://miami.box.com/s/ylmvqynbtchx5xhof0quaeu9w62mxaca) and place all files in the folder 
CL4CAPro/OncotypeDX, and then run notebook OncotypeDX.ipynb. This willl produce the box-plots of c-indexes in Figure 6 of the paper. 



## Usage of individual scripts

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
