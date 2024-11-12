#!/bin/bash

#activate conda environment mpv-sem
source activate mpv-sem

# Path to the directory containing YAML configurations
CONFIG_DIR="configs"
LOG_DIR="logs"

# Create logs directory if it doesn't exist
mkdir -p "$LOG_DIR"

# Iterate over all YAML files in the configuration directory
for CONFIG in "$CONFIG_DIR"/*.yaml; do
    # Extract the base name of the configuration file (e.g., "config_mobilenet_v2")
    CONFIG_BASENAME=$(basename "$CONFIG" .yaml)

    # Define the log file path
    LOG_FILE="$LOG_DIR/${CONFIG_BASENAME}.log"

    # Run the training script and save output to the log file while displaying live output
    echo "Starting training with configuration: $CONFIG"
    echo "Saving logs to: $LOG_FILE"
    python clasification_models.py "$CONFIG" 2>&1 | tee "$LOG_FILE"

    if [ ${PIPESTATUS[0]} -eq 0 ]; then
        echo "Training completed successfully for $CONFIG. Logs saved to $LOG_FILE"
    else
        echo "Training failed for $CONFIG. Check logs in $LOG_FILE for details."
    fi

    echo "--------------------------------------------"
done
