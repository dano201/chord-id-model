# Chord Identification Model

This project provides a sample directory structure and completes all necessary workflows to train and evaluate a CNN for musical pitch identification, implemented using TensorFlow.

The project facilitates the following workflows:

1) Automated pre-processing and labeling of data.
2) Defining and training the model, using hyperparameters specified in the config file.
3) Evaluating the model in terms of accuracy, precision, recall, and F-score.

Disclaimer: Python version 3.9-3.12 is required, as specified in TF documentation: https://www.tensorflow.org/install/pip

## Installation

1. Clone the repository.

2. Create and activate a virtual environment: 
    ```bash
    python -m venv .venv
    .venv\Scripts\activate
    ```

3. Install dependencies:
    `pip install -r requirements.txt`
    
## Using the project

The project was developed to facilitate model training based on guitar chords in WAV format. The `data` directory provided is split into train, test and validation subdirectories; within these, subdirectories exist for raw files, processed files, and labels. The user can use any alternative structure they wish by altering the relevant values in the `config.yaml` file. 

The only requirement is that raw data files be in the format [Chord_Name]_[Fretting].wav, e.g., 'C_X32010.wav'. This format allows the required notes to be extracted for labeling purposes.

Three main executable scripts are included and detailed below.

## Processing Data

Pull the relevant data from the specified raw directory. Generate Constant-Q Transforms of all WAV files within the directory (including subdirectories), as well as corresponding labels. Save both new files to specified processed and raw directories, preserving subdirectory structure.

To execute this stage, run the `process.py` script using your IDE, or the below command from the root directory:
`python src/process.py`

Data locations can be tweaked to suit a different directory structure in `config.yaml`, but all existing fields must be provided.

## Training the model

Train the model using the specified hyperparameters, saving the resulting model using the path specified. Training history is saved in `history.json` within the `model` directory. Checkpoints are used - the intermediary file is stored in `models/model_checkpoint.keras`.

To execute this stage, run the `train_model.py` script using your IDE or the below command (from the root directory):
`python src/train_model.py`

Hyperparameters can be tweaked in `config.yaml`.

## Evaluation

Evaluate the model using a specified test set (earlier pre-processed in `process` stage). The model will be evaluation in terms of Accuracy, Precision, Recall & F1-score.

To execute this stage, run the `evaluate.py` script using your IDE or the below command (from the root directory):
`python src/evaluate.py`

