#!/bin/bash
# run_analysis.sh

# Activate environment if needed
# source activate your_conda_environment

# Run Python scripts
echo "Welcome to the scRNA-seq Analysis Tool"

# Function to prompt for file input
prompt_for_file() {
    echo -n "Enter the path to the $1 file: "
    read path
    while [ ! -f "$path" ]; do
        echo "File does not exist. Please try again."
        echo -n "Enter the path to the $1 file: "
        read path
    done
    echo $path
}

# Gather inputs
echo "Please provide the required files."
human_matrix=$(prompt_for_file "Human Matrix (.mtx)")
mouse_matrix=$(prompt_for_file "Mouse Matrix (.mtx)")
human_genes=$(prompt_for_file "Human Gene Names (.tsv.gz)")
mouse_genes=$(prompt_for_file "Mouse Gene Names (.tsv.gz)")
human_metadata=$(prompt_for_file "Human Metadata (.tsv)")
mouse_metadata=$(prompt_for_file "Mouse Metadata (.tsv)")

# Run Python scripts
echo "Running data preparation..."
python prep_data.py $human_matrix $mouse_matrix $human_genes $mouse_genes $human_metadata $



echo "Analysis complete. Check output files for results."