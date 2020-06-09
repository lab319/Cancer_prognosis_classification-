# Cancer_prognosis_classification
### If you find our method is useful, please cite our paper: Baoshan Ma, Fanyu Meng, Ge Yan, et al. Diagnostic classification of cancers using extreme gradient boosting algorithm and multi-omics data [J]. Computers in Biology and Medicine, vol 121, June 2020, 103761. ###

**This repository contains python and R implementation of the algorithms proposed in "Diagnostic classification of cancers using extreme gradient boosting algorithm and multi-omics data".**

**The version of Python and packages**

Python version=3.6

Xgboost version=0.82

scikit-learn version=0.24.2

numpy version=1.16.3

### The datasets of the program

The data used in this research are collected from The Cancer Genome Atlas (TCGA) project and that are publicly available at <https://portal.gdc.cancer.gov>.

### The describe of the program

The analysis is divided into three sections saved in this repository.

**1)       XGBoost_model:** classification model based on XGBoost was presented to classify the stage (early or late) of KIRC, KIRP, HNSC and LUSC patients.

**2)       Other_machine_learning_models:** several popular machine learning algorithms are compared with XGBoost. The files in the repository are examples, which construct different predictive models to classify the stage (early or late) of KIRP patients.

**3)       Milti_omics_model:** we integrate different types of omics data to further improve the prediction performance of the classification model. The files in the repository are examples, which construct different predictive models to classify the stage (early or late) of KIRP patients. To implement the model, run the file model.pkl, which is a trained DNN model.
