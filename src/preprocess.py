import librosa
import numpy as np
import os
from helper import find_notes, note_dict

def to_CQT(file):
    y, sr = librosa.load(file, sr=16000)
    CQT = librosa.cqt(y, sr=sr, fmin=librosa.note_to_hz('E2'), n_bins=74, bins_per_octave=24)
    CQT_db = librosa.amplitude_to_db(np.abs(CQT), ref=np.max)

    # Normalise
    CQT_db = (CQT_db - np.min(CQT_db)) / (np.max(CQT_db) - np.min(CQT_db))

    return CQT_db

def create_label(filename):
    frets = filename.split("_")[-1]

    notes = find_notes(frets)
    print("frets: " + frets + "; raw notes: " + notes + "\n")

    label = np.zeros(37)
    
    for note in notes.split(" "):
        if note in note_dict:
            label[note_dict[note]] = 1
    
    return label

def preprocess_dir(input_dir, output_dir, label_dir):
    for root, dirs, files in os.walk(input_dir):
        for f in files:
            if f.endswith(".wav"):
                path = os.path.join(root, f)

                relative_path = os.path.relpath(root, input_dir)
                output_subdir = os.path.join(output_dir, relative_path)
                label_subdir = os.path.join(label_dir, relative_path)
                os.makedirs(output_subdir, exist_ok=True)
                os.makedirs(label_subdir, exist_ok=True)
                
                cqt = to_CQT(path)

                if cqt is not None:
                    output_filepath = os.path.join(output_subdir, f.replace(".wav", ".npy"))
                    np.save(output_filepath, cqt)

                    print(f"Processing label for: {f[:-4]}")
                    label = create_label(f[:-4])

                    label_filepath = os.path.join(label_subdir, f.replace(".wav", "_label.npy"))
                    np.save(label_filepath, label)

if __name__ == "__main__":

    train_input_dir = "data/train/raw/"
    train_output_dir = "data/train/processed/"
    train_label_dir = "data/train/labels/"

    preprocess_dir(train_input_dir, train_output_dir, train_label_dir)

    val_input_dir = "data/validation/raw/"
    val_output_dir = "data/validation/processed/"
    val_label_dir = "data/validation/labels/"

    preprocess_dir(val_input_dir, val_output_dir, val_label_dir)

    test_input_dir = "data/test/raw/"
    test_output_dir = "data/test/processed/"
    test_label_dir = "data/test/labels/"

    preprocess_dir(test_input_dir, test_output_dir, test_label_dir)

