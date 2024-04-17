#!/bin/bash
# run_analysis.sh

# Activate environment if needed
# source activate your_conda_environment

# Run Python scripts
echo "phenopreserve: scRNA-seq Analysis Tool"

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

# Run data preparation Python script
echo "Running data preparation..."
python prep_data.py $human_matrix $mouse_matrix $human_genes $mouse_genes $human_metadata $mouse_metadata

# Check if the preparation was successful and files were created
if [[ -f "X_mouse.csv" && -f "y_mouse.csv" && -f "y_class.csv" ]]; then
    echo "Data prepared successfully. Proceeding with training and evaluation."
    # Run training and evaluation Python script
    python train_evaluate.py X_mouse.csv y_mouse.csv y_class.csv
else
    echo "Data preparation did not complete successfully. Please check the logs for errors."
fi

echo "Analysis complete. Check output files for results."
