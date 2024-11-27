#!/bin/bash
# run_analysis.sh

# Run Python scripts
echo "phenopreserve: scRNA-seq Analysis Tool"

# Function to show script usage
usage() {
    echo "Usage: $0 -m human_matrix -t mouse_matrix -g human_genes -n mouse_genes -d human_metadata -s mouse_metadata"
    exit 1
}

# Check if no options were provided
if [ $# -eq 0 ]; then
    usage
fi

while getopts ":m:t:g:n:d:s:h" opt; do
    case ${opt} in
        m ) human_matrix=$OPTARG ;;
        t ) mouse_matrix=$OPTARG ;;
        g ) human_genes=$OPTARG ;;
        n ) mouse_genes=$OPTARG ;;
        d ) human_metadata=$OPTARG ;;
        s ) mouse_metadata=$OPTARG ;;
        h ) usage ;;
        \? ) echo "Invalid Option: -$OPTARG" >&2; usage ;;
        : ) echo "Option -$OPTARG requires an argument." >&2; exit 1 ;;
    esac
done
# Check that all parameters are set
if [ -z "$human_matrix" ] || [ -z "$mouse_matrix" ] || [ -z "$human_genes" ] || [ -z "$mouse_genes" ] || [ -z "$human_metadata" ] || [ -z "$mouse_metadata" ]; then
    echo "All parameters are required."
    usage
fi

# Run data preparation Python script
echo "Running data preparation..."
python prep_data.py "$human_matrix" "$mouse_matrix" "$human_genes" "$mouse_genes" "$human_metadata" "$mouse_metadata"

# Check if the preparation was successful and files were created
if [[ -f "X_mouse.csv" && -f "y_mouse.csv" && -f "y_class.csv" ]]; then
    echo "Data prepared successfully. Proceeding with training and evaluation."
    # Run training and evaluation Python script
    python train_evaluate.py X_mouse.csv y_mouse.csv y_class.csv
else
    echo "Data preparation did not complete successfully. Check output files and logs for errors."
fi

echo "Analysis complete. Check output files for results."
