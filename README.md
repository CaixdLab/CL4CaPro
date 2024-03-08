# Contrastive Learning for Predicting Cancer Prognosis Using Gene Expression Values

## Authors
- Anchen Sun
- Zhibin Chen
- Xiaodong Cai

## Abstract
Predicting the prognosis of cancer using the tumor transcriptome is a significant challenge in the medical field. Although several Artificial Neural Networks (ANNs) have been developed for this purpose, their performance has not substantially surpassed that of the regularized Cox proportional hazards regression model. This paper introduces a novel approach by applying supervised Contrastive Learning (CL) to tumor gene expression and clinical data to generate meaningful low-dimensional feature representations. These representations are then utilized to train a Cox model, named CLCox, for the prognosis prediction of cancer. Our method demonstrated significant improvements over existing models across 19 types of cancer using data from The Cancer Genome Atlas (TCGA). Additionally, we developed CL-based classifiers to categorize tumors into different risk groups, achieving remarkable classification accuracy and area under the receiver operating characteristic curve (AUC) scores. This paper also includes validation of the proposed models with independent lung and prostate cancer cohorts.

## Introduction
This research aims to address the limitations of current Artificial Neural Network (ANN) models in predicting cancer prognosis by leveraging the power of Contrastive Learning (CL). CL has recently shown promising results in image classification tasks by efficiently learning feature representations from a limited dataset. By applying supervised CL to both tumor gene expression and clinical data, we were able to train a Cox model that significantly outperforms traditional methods.

## Methodology
We utilized data from The Cancer Genome Atlas (TCGA) to train our CL-based Cox model (CLCox) and classifiers. The methodology section details the process of applying supervised contrastive learning to generate low-dimensional feature representations and the subsequent training of the Cox model and classifiers using these features.

## Results
Our CLCox model exhibited superior performance in predicting the prognosis of 19 different cancer types, outperforming existing methods significantly. Additionally, our CL-based classifiers demonstrated high accuracy in tumor classification into different risk groups, with AUC scores exceeding 0.8 for 14 cancer types and 0.9 for 2 cancer types. The models' effectiveness was further validated using independent lung and prostate cancer datasets.

## GitHub Repository
We are committed to open-sourcing our research for the benefit of the scientific community. The codebase, along with detailed documentation on how to use our models, is available on GitHub at [https://github.com/CaixdLab/CL4CaPro]. We encourage other researchers to utilize our models and contribute to further advancements in the field.

## Conclusion
This paper presents a groundbreaking approach to predicting cancer prognosis using supervised contrastive learning. The significant improvements in prognosis prediction and tumor classification underscore the potential of CL in medical applications. Future work will focus on exploring the application of CL to other types of medical data and further refining our models.

## Citation
If you find our research useful, please cite our paper as follows: