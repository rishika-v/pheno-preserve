# prep_data.py
import sys
import pandas as pd
import scanpy as sc

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


def dynamic_normalize(adata):
    """ Normalizes the data using a dynamically computed target sum based on median counts. """
    median_counts = np.median(adata.X.sum(axis=1))
    sc.pp.normalize_total(adata, target_sum=median_counts)

def dynamic_highly_variable_genes(adata):
    """ Identifies highly variable genes based on dynamic thresholds calculated from data percentiles. """
    sc.pp.highly_variable_genes(adata, min_mean=np.percentile(adata.X.mean(axis=0), 10),
                                max_mean=np.percentile(adata.X.mean(axis=0), 90),
                                min_disp=np.percentile(adata.X.std(axis=0), 50))

def preprocess_data(adata):
    """ Runs preprocessing pipeline on the provided AnnData object. """
    # Dynamic normalization
    dynamic_normalize(adata)
    # Logarithmize the data
    sc.pp.log1p(adata)
    # Identify highly variable genes dynamically
    dynamic_highly_variable_genes(adata)
    
    # Ensure that 'highly_variable' column exists before filtering
    if 'highly_variable' in adata.var.columns:
        adata = adata[:, adata.var.highly_variable]
    else:
        print("Warning: 'highly_variable' genes not identified. Proceeding with all genes.")

    # Scale data
    sc.pp.scale(adata, max_value=10)
    return adata

def main():
    if len(sys.argv) < 5:
        print("Usage: python prep_data.py <human_matrix> <mouse_matrix> <human_genes> <mouse_genes> <human_metadata> <mouse_metadata>")
        sys.exit(1)

    human_matrix = sys.argv[1]
    mouse_matrix = sys.argv[2]
    human_genes = sys.argv[3]
    mouse_genes = sys.argv[4]
    human_metadata = sys.argv[5]
    mouse_metadata = sys.argv[6]

    # Load data
    adata_human = load_data(human_matrix, human_genes, human_metadata)
    adata_mouse = load_data(mouse_matrix, mouse_genes, mouse_metadata)

    # Filter to common genes (default method)
    adata_human, adata_mouse = intersect_genes(adata_human, adata_mouse)

    # Transpose data
    adata_human = transpose_data(adata_human)
    adata_mouse = transpose_data(adata_mouse)

    print(f"Transposed human data shape: {adata_human.shape}")
    print(f"Transposed mouse data shape: {adata_mouse.shape}")

    # Preprocess data
    adata_human = preprocess_data(adata_human)
    adata_mouse = preprocess_data(adata_mouse)

    print(f"Processed human data shape: {adata_human.shape}")
    print(f"Processed mouse data shape: {adata_mouse.shape}")

if __name__ == "__main__":
    main()

