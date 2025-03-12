import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from model import build_model
from helper import make_abs
import yaml
import json

def create_dataset(X, y, batch_size):
    def _generator():
        for cqt, label in zip(X, y):
            yield cqt, label

    dataset = tf.data.Dataset.from_generator(_generator, output_signature=(
        tf.TensorSpec(shape=(74, None), dtype=tf.float32),
        tf.TensorSpec(shape=(37,), dtype=tf.float32))) 
    
    dataset = dataset.shuffle(len(X))
    dataset = dataset.padded_batch(batch_size, padded_shapes=([74, None], [37]))

    return dataset

def prepare_data(input_dir, label_dir, batch_size):
    X = []
    y = []
    
    for root, dirs, files in os.walk(input_dir):
        for f in files:
            if f.endswith('.npy'): 
        
                relative_path = os.path.relpath(root, input_dir)
                cqt_filepath = os.path.join(root, f)
                
                label_filename = f.replace(".npy", "_label.npy")
                
                label_filepath = os.path.join(label_dir, relative_path, label_filename)
                
                if os.path.exists(label_filepath):
                    X.append(np.load(cqt_filepath))
                    y.append(np.load(label_filepath))
                    print(f"Processing: {f}")
                else:
                    ValueError(f"Label not found for {f}")

    dataset = create_dataset(X, y, batch_size)

    return dataset

def train(dataset, validation, batch_size, epochs, model_path):
    model = build_model()

    early_stopping = EarlyStopping(
        monitor='val_loss',
        patience = 1, 
        restore_best_weights=True,
        verbose=1)

    checkpoint = ModelCheckpoint(
        filepath='models/model_checkpoint.keras',
        monitor='val_accuracy',
        mode='max',
        save_best_only=True)

    trained = model.fit(dataset,
                        epochs = epochs, 
                        batch_size = batch_size,
                        validation_data = validation,
                        callbacks=[early_stopping, checkpoint])
    
    history = os.path.join(os.path.dirname(model_path), "history.json")
    with open(history, 'w') as f:
        json.dump(trained.history, f)

    model.save(model_path)

    return "Training complete."

if __name__ == "__main__":

    with open(make_abs('./config.yaml'), 'r') as f:
        conf = yaml.safe_load(f)
    
    batch_size = conf['params']['batch_size']
    epochs = conf['params']['epochs']
    model_path = make_abs(conf['paths']['model'])

    train_dir = make_abs(conf['paths']['train_processed'])
    train_label_dir = make_abs(conf['paths']['train_labels'])

    val_dir = make_abs(conf['paths']['val_processed'])
    val_label_dir = make_abs(conf['paths']['val_labels'])

    train_set = prepare_data(train_dir, train_label_dir, batch_size)
    val_set = prepare_data(val_dir, val_label_dir, batch_size)

    trained = train(train_set, val_set, batch_size, epochs, model_path)