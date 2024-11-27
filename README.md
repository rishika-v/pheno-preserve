# phenopreserve
## FCBB Final Project:

PhenoPreserve is a tool designed to facilitate the prediction of the translational value of preclinical models based on preservation of immune cell phenotypes across different species (e.g., human and mouse). It integrates expression data preprocessing with classification models to predict phenotypic consistency using scRNA-seq data.

## Features
### Data Preparation: 
Processes raw .mtx matrix files, gene names, and metadata to prepare datasets for analysis.
### Classification Analysis: 
Determines the class labels (high or low consistency) based on gene expression data. Identifies most important features for prediction.
### Regression Analysis: 
Predicts quantitative measure of consistency index from gene expression data. Identifies most important features for prediction.

## Dependencies
To run PhenoPreserve, you need the following Python packages:
```
numpy
pandas
scikit-learn
imbalanced-learn
matplotlib
seaborn
scanpy
```

You can install these dependencies via pip:

`pip install numpy pandas scikit-learn imbalanced-learn matplotlib seaborn scanpy`

## Usage
To use PhenoPreserve, you must have the data files in the correct format (.mtx for matrices, .tsv for gene names and metadata). 
The usage function in the script provides instructions on how to run the script with required command-line options. Each option corresponds to a different input file needed for the analysis:

-m: Species 1 Matrix file (.mtx format). <br />
-t: Species 2 Matrix file (.mtx format). <br />
-g: Species 1 Gene Names file (.tsv.gz format).<br />
-n: Species 2 Gene Names file (.tsv.gz format).<br />
-d: Species 1 Metadata file (.tsv format).<br />
-s: Species 2 Metadata file (.tsv format).<br />
-h: Displays how to use phenopreserve.<br />

`chmod +x run_analysis.sh`

`./run_analysis.sh -m species1_matrix.mtx -t species2_matrix.mtx -g species1_genes.tsv.gz -n species2_genes.tsv.gz -d species1_metadata.tsv -s species2_metadata.tsv`

### Review Results: 
After the script completes, check the output files and logs for detailed results and any potential errors.

## Output
The tool generates several outputs:

Classification and regression metrics are logged in model_evaluation.log.
Plots such as confusion matrices and ROC curves are saved as PNG files in the current directory.
