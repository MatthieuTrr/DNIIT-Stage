import os
import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score
from sklearn.utils.class_weight import compute_class_weight
import tensorflow as tf
from src.utils.data_utils import load_labels_from_csv, prepare_dual_dataset
from src.embedding.word2vecTraining import global_training_pipeline
from src.model.dh_cnn import build_final_dh_cnn
from src.utils.config import Config


def train_single_run(X_syn_train, X_sem_train, y_train,
                     X_syn_test, X_sem_test, y_test,
                     epochs=Config.EPOCHS, batch_size=Config.BATCH_SIZE):
    """
    Trains one fresh model instance and returns its metrics.
    Called repeatedly by the main loop for averaging over 30 runs.
    """
    classes = np.unique(y_train)
    weights = compute_class_weight(class_weight="balanced", classes=classes, y=y_train)
    class_weight_dict = dict(zip(classes.tolist(), weights.tolist()))

    model = build_final_dh_cnn()

    model.fit(
        x=[X_syn_train, X_sem_train],
        y=y_train,
        epochs=epochs,
        batch_size=batch_size,
        validation_split=0.1,
        class_weight=class_weight_dict,
        verbose=0
    )

    y_proba = model.predict([X_syn_test, X_sem_test], verbose=0)
    y_pred  = np.argmax(y_proba, axis=1)

    precision = precision_score(y_test, y_pred, zero_division=0)
    recall    = recall_score(y_test, y_pred, zero_division=0)
    f1        = f1_score(y_test, y_pred, zero_division=0)
    auc       = roc_auc_score(y_test, y_proba[:, 1]) if len(np.unique(y_test)) > 1 else float("nan")

    print(f"  P={precision:.4f}  R={recall:.4f}  F1={f1:.4f}  AUC={auc:.4f}")
    return np.array([precision, recall, f1, auc])


N_RUNS = 30

if __name__ == "__main__":
    TRAIN_DATASET_PATH = "src/data/raw/test_java/jedit40source/jEdit/"
    TRAIN_CSV_PATH     = "src/data/raw/test_java/jedit40source/jedit-4.0.csv"
    TEST_DATASET_PATH  = "src/data/raw/test_java/jedit41source/jEdit"
    TEST_CSV_PATH      = "src/data/raw/test_java/jedit41source/jedit-4.1.csv"

    # Feature extraction — done ONCE (W2V is seeded, Node2Vec re-runs each iteration)
    w2v_model = global_training_pipeline(
        TRAIN_DATASET_PATH, source_type="folder",
        embedding_dim=Config.AST_EMBEDDING,
        window=Config.WINDOW_SIZE,
        min_count=Config.MIN_COUNT
    )

    if w2v_model is None:
        print("ERROR: Word2Vec training failed. Aborting.")
        exit()

    train_labels = load_labels_from_csv(TRAIN_CSV_PATH)
    test_labels  = load_labels_from_csv(TEST_CSV_PATH)

    # 30 independent runs — Node2Vec and model weights are re-randomised each time
    results = np.zeros(4)
    for i in range(N_RUNS):
        print(f"Run {i+1}/{N_RUNS}")

        X_syn_train, X_sem_train, y_train = prepare_dual_dataset(
            TRAIN_DATASET_PATH, train_labels, w2v_model
        )
        X_syn_test, X_sem_test, y_test = prepare_dual_dataset(
            TEST_DATASET_PATH, test_labels, w2v_model
        )

        metrics = train_single_run(
            X_syn_train, X_sem_train, y_train,
            X_syn_test,  X_sem_test,  y_test
        )
        results += metrics

    avg = results / N_RUNS
    print(f"\nAverage over {N_RUNS} runs:")
    print(f"  Precision : {avg[0]:.4f}")
    print(f"  Recall    : {avg[1]:.4f}")
    print(f"  F1        : {avg[2]:.4f}")
    print(f"  AUC       : {avg[3]:.4f}")