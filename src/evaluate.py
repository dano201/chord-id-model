import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
import tensorflow as tf
import os

def load_data(input_dir, label_dir):
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

    return X, y

def create_dataset(X, batch_size=4):
    def _generator():
        for cqt in X:
            yield cqt

    dataset = tf.data.Dataset.from_generator(
        _generator,
        output_signature=tf.TensorSpec(shape=(74, None), dtype=tf.float32)
    )

    dataset = dataset.padded_batch(batch_size, padded_shapes=([74, None]))

    return dataset

def evaluate(input_dir, label_dir):
    X, y_true = load_data(input_dir, label_dir)
    model = tf.keras.models.load_model('models/model.h5')
    
    test_set = create_dataset(X, batch_size=16)
    y_pred_raw = model.predict(test_set)

    threshold = np.max(y_pred_raw, axis=1, keepdims=True) * 0.4

    y_pred = (y_pred_raw >= threshold).astype(int)
    y_true = np.array(y_true)

    accuracy = accuracy_score(y_true.flatten(), y_pred.flatten())
    precision = precision_score(y_true, y_pred, average='samples')
    recall = recall_score(y_true, y_pred, average='samples')
    f1 = f1_score(y_true, y_pred, average='samples')

    print(f"Accuracy: {accuracy:.3f}")
    print(f"Precision: {precision:.3f}")
    print(f"Recall: {recall:.3f}")
    print(f"F1 Score: {f1:.3f}")

if __name__ == "__main__":
    
    cqt_dir = "data/test/processed/"
    label_dir = "data/test/labels/"

    evaluate(cqt_dir, label_dir)

