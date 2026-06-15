import os
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score
import tensorflow as tf

from src.utils.data_utils import load_labels_from_csv, prepare_dual_dataset
from src.embedding.word2vecTraining import global_training_pipeline
from src.model.dh_cnn import build_final_dh_cnn
from src.utils.config  import Config

def train_dh_cnn_end_to_end(dataset_path, csv_path, epochs=Config.EPOCHS, batch_size=Config.BATCH_SIZE):
    """
    Executes the end-to-end training pipeline for the DH-CNN model.
    
    This pipeline trains the Word2Vec model, prepares the dual dataset (Syntaxic 
    and Semantic features), builds the final DH-CNN architecture, trains the neural 
    network with the dual inputs, and evaluates its performance on the test set.

    Args:
        dataset_path (str): The root directory path containing the Java source files.
        csv_path (str): The file path to the CSV containing the defect labels.
        epochs (int, optional): The number of epochs for training. Defaults to Config.EPOCHS.
        batch_size (int, optional): The batch size for training. Defaults to Config.BATCH_SIZE.
        
    Returns:
        tuple: A tuple containing the trained Keras model and the training history.
    """
    w2v_model = global_training_pipeline(dataset_path, source_type="folder", embedding_dim=Config.EMBEDDING_DIM, window=Config.WINDOW_SIZE, min_count=Config.MIN_COUNT)
    
    if w2v_model is None:
        return None, None

    labels_dict = load_labels_from_csv(csv_path)
    X_syntaxic, X_semantic, y = prepare_dual_dataset(dataset_path, labels_dict, w2v_model)
    
    X_syn_train, X_syn_test, X_sem_train, X_sem_test, y_train, y_test = train_test_split(
        X_syntaxic, X_semantic, y, test_size=0.2, random_state=42, stratify=y
    )

    model = build_final_dh_cnn()
    
    history = model.fit(
        x=[X_syn_train, X_sem_train],
        y=y_train,
        epochs=epochs,
        batch_size=batch_size,
        validation_split=0.1,
        verbose=1
    )

    y_proba = model.predict([X_syn_test, X_sem_test]) 
    y_pred  = np.argmax(y_proba, axis=1)
 
    precision = precision_score(y_test, y_pred, zero_division=0)
    recall    = recall_score(y_test, y_pred, zero_division=0)
    f1        = f1_score(y_test, y_pred, zero_division=0)
    auc       = roc_auc_score(y_test, y_proba[:, 1]) if len(np.unique(y_test)) > 1 else float("nan")
 
    print(f"Precision : {precision:.4f}")
    print(f"Recall    : {recall:.4f}")
    print(f"F1        : {f1:.4f}")
    print(f"AUC       : {auc:.4f}")
 
    return model, history


if __name__ == "__main__":
    DATASET_PATH = "src/data/raw/test_java/jakarta-log4j-1.1.3/" # a changer selon chemin/fichier de dataset
    CSV_PATH     = "src/data/raw/test_java/log4j-1.1.csv"
    
    train_dh_cnn_end_to_end(DATASET_PATH, CSV_PATH)