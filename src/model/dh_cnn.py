from src.embedding.word2vecTraining import global_training_pipeline
import pandas as pd
from src.embedding.embedding import prepare_embedding
import os
import tensorflow as tf
from tensorflow import keras
import numpy as np
from sklearn.model_selection import train_test_split
from src.model.cnn_ast import build_syntax_CNN
from src.model.semantic_DH_CNN import build_semantic_level_cnn
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score
def complete_cnn(
    MAX_TOKENS=50,
    EMBEDDING_DIM=100,
    CNN_FILTERS=128,
    KERNEL_SIZE=5,
    DROPOUT_RATE=0.5,
):
    semantic=build_semantic_level_cnn()
    syntaxic=build_syntax_CNN()
    syntax_input   = keras.Input(shape=(MAX_TOKENS, EMBEDDING_DIM), name="syntax_input")
    semantic_input = keras.Input(shape=(MAX_TOKENS, EMBEDDING_DIM), name="semantic_input")
    syntax_out   = syntaxic(syntax_input)
    semantic_out = semantic(semantic_input)
    merged = keras.layers.Concatenate(name="concatenate")([syntax_out, semantic_out])
    
    outputs = keras.layers.Dense(2, activation="softmax", name="output")(merged)
    full_model = keras.Model(
        inputs=[syntax_input, semantic_input],
        outputs=outputs,
        name="DH-CNN_Combined",
    )
    full_model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )
    return full_model

def load_labels_from_csv(csv_path):
    df=pd.read_csv(csv_path)
    labels={}
    for _, row in df.iterrows():
        class_name=row["name"]
        file_name=class_name.split(".")[-1]+".java"
        if int(row["bug"])>0:
            bug_label=1
        else:
            bug_label=0
        labels[file_name]=bug_label
    return labels


def prepare_dataset_with_labels(dataset_root, labels_dict, w2v_model):
    X, y=[],[]
    for root, _, files in os.walk(dataset_root):
        for file in files:
            if not file.endswith(".java"):
                continue
            full_path=os.path.join(root,file)
            if file not in labels_dict:
                continue
            with open(full_path, "r", encoding="utf-8", errors="ignore") as f:
                code = f.read()
            try:
                matrix = prepare_embedding(code, w2v_model)
                X.append(matrix)
                y.append(labels_dict[file])
            except Exception as e:
                print(f"  Skipped {file}: {e}")
    return np.array(X), np.array(y)

def train_and_evaluate(dataset_path, csv_path, source_type="folder", epochs=50, batch_size=32):
    print("Step 1: Word2Vec Training")
    w2v_model=global_training_pipeline(dataset_path, source_type, 100, 5, 1)

    print("Step 2: Loading labels and building embeddings")
    labels_dict=load_labels_from_csv(csv_path)
    X, y=prepare_dataset_with_labels(dataset_path,labels_dict, w2v_model)
    print(f"Dataset: {X.shape} | Buggy: {y.sum()} | Clean: {(y == 0).sum()}")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=128,stratify=y)

    print("Step 3: CNN training")
    model = complete_cnn()
    model.summary()
    history = model.fit(
        [X_train,X_train], y_train,
        epochs=epochs,
        batch_size=batch_size,
        validation_split=0.1,
        verbose=1
    )

    print("Results on test set")
    y_proba   = model.predict([X_test,X_test])
    y_pred    = np.argmax(y_proba, axis=1)
 
    precision = precision_score(y_test, y_pred, zero_division=0)
    recall    = recall_score(y_test, y_pred, zero_division=0)
    f1        = f1_score(y_test, y_pred, zero_division=0)
    auc       = roc_auc_score(y_test, y_proba[:, 1]) if len(np.unique(y_test)) > 1 else float("nan")
 
    print(f"Precision : {precision:.4f}")
    print(f"Recall    : {recall:.4f}")
    print(f"F1        : {f1:.4f}")
    print(f"AUC       : {auc:.4f}")
 
    return model, history


DATASET_PATH = "../py files/test_java/"   # ← à changer selon l'orga de tes fichiers
CSV_PATH     = "../py files/log4j-1.0.csv"               # ← pareil
train_and_evaluate(DATASET_PATH, CSV_PATH)