import librosa
import numpy as np
import os
import yaml
from helper import find_notes, make_abs, note_dict

def to_CQT(file):
    y, sr = librosa.load(file, sr=16000)
    cqt = librosa.cqt(y, sr=sr, fmin=librosa.note_to_hz('E2'), n_bins=74, bins_per_octave=24)
    to_db = librosa.amplitude_to_db(np.abs(cqt), ref=np.max)

    cqt = (to_db - np.min(to_db)) / (np.max(to_db) - np.min(to_db))
    return to_db

def create_label(filename):
    frets = filename.split("_")[-1]

    notes = find_notes(frets)
    print("frets: " + frets + "; raw notes: " + notes + "\n")

    label = np.zeros(37)
    
    for note in notes.split(" "):
        if note in note_dict:
            label[note_dict[note]] = 1
    
    return label

def process_dir(input_dir, output_dir, label_dir):
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

    with open(make_abs('./config.yaml'), 'r') as f:
        conf = yaml.safe_load(f)

    process_dir(make_abs(conf['paths']['train_raw']), 
                make_abs(conf['paths']['train_processed']), 
                make_abs(conf['paths']['train_labels']))

    process_dir(make_abs(conf['paths']['val_raw']), 
                make_abs(conf['paths']['val_processed']), 
                make_abs(conf['paths']['val_labels']))

    process_dir(make_abs(conf['paths']['test_raw']), 
                make_abs(conf['paths']['test_processed']), 
                make_abs(conf['paths']['test_labels']))

