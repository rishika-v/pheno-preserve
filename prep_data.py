import sys
import pandas as pd
import numpy as np
import scanpy as sc
from sklearn.metrics.pairwise import euclidean_distances
import logging

# Setup logging
logging.basicConfig(filename='data_preprocessing.log', level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def load_data(matrix_path, gene_names_path, metadata_path):
    adata = sc.read_mtx(matrix_path)
    gene_names = pd.read_csv(gene_names_path, header=None, sep='\t')
    adata.var_names = gene_names[0]
    metadata = pd.read_csv(metadata_path, sep='\t')
    adata.obs = metadata
    return adata

def intersect_genes(adata_human, adata_mouse):
    common_genes = adata_human.var_names.intersection(adata_mouse.var_names)
    common_genes = sorted(common_genes)
    return adata_human[:, common_genes], adata_mouse[:, common_genes]

def transpose_data(adata):
    return sc.AnnData(X=adata.X.T, obs=adata.var, var=adata.obs)

def preprocess_data(adata):
    dynamic_normalize(adata)
    sc.pp.log1p(adata)
    dynamic_highly_variable_genes(adata)
    if 'highly_variable' in adata.var.columns:
        adata = adata[:, adata.var.highly_variable]
    else:
        logging.warning("Highly variable genes not identified. Proceeding with all genes.")
    sc.pp.scale(adata, max_value=10)
    return adata

def dynamic_normalize(adata):
    median_counts = np.median(adata.X.sum(axis=1))
    sc.pp.normalize_total(adata, target_sum=median_counts)

def dynamic_highly_variable_genes(adata):
    sc.pp.highly_variable_genes(adata, min_mean=np.percentile(adata.X.mean(axis=0), 10),
                                max_mean=np.percentile(adata.X.mean(axis=0), 90),
                                min_disp=np.percentile(adata.X.std(axis=0), 50))

def calculate_ci(adata_human, adata_mouse):
    human_expression_means = np.array(adata_human.X.mean(axis=0)).flatten()
    mouse_expression_means = np.array(adata_mouse.X.mean(axis=0)).flatten()
    human_baseline = pd.DataFrame(human_expression_means, index=adata_human.var_names, columns=['mean_expression'])
    mouse_baseline = pd.DataFrame(mouse_expression_means, index=adata_mouse.var_names, columns=['mean_expression'])
    human_baseline.sort_index(inplace=True)
    mouse_baseline.sort_index(inplace=True)
    ci_scores = euclidean_distances(mouse_baseline.values.reshape(-1, 1), human_baseline.values.reshape(-1, 1))
    adata_mouse.obs['ConsistencyIndex'] = ci_scores.diagonal()
    threshold = np.mean(ci_scores.diagonal())
    adata_mouse.obs['CI_Binary'] = (adata_mouse.obs['ConsistencyIndex'] > threshold).astype(int)

def prepare_data_for_modeling(adata):
    features = pd.DataFrame(adata.X, index=adata.obs_names, columns=adata.var_names)
    X = features.drop(columns=['ConsistencyIndex', 'CI_Binary'], errors='ignore')
    y = adata.obs['ConsistencyIndex'].values
    y_class = adata.obs['CI_Binary'].values
    return X, y, y_class

def save_data(X, y, y_class):
    X.to_csv('X_mouse.csv', index=False)
    pd.DataFrame(y, columns=['ConsistencyIndex']).to_csv('y_mouse.csv', index=False)
    pd.DataFrame(y_class, columns=['CI_Binary']).to_csv('y_class.csv', index=False)

def main():
    if len(sys.argv) < 7:
        logging.error("Incorrect number of arguments provided.")
        sys.exit(1)

    human_matrix, mouse_matrix, human_genes, mouse_genes, human_metadata, mouse_metadata = sys.argv[1:7]
    adata_human = load_data(human_matrix, human_genes, human_metadata)
    adata_mouse = load_data(mouse_matrix, mouse_genes, mouse_metadata)
    adata_human, adata_mouse = intersect_genes(adata_human, adata_mouse)
    adata_human = transpose_data(adata_human)
    adata_mouse = transpose_data(adata_mouse)

    # Preprocess data
    adata_human = preprocess_data(adata_human)
    adata_mouse = preprocess_data(adata_mouse)

    # Calculate Consistency Index
    calculate_ci(adata_human, adata_mouse)

    # Prepare data for modeling
    X_mouse, y_mouse, y