import os
import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score
from sklearn.utils.class_weight import compute_class_weight
import tensorflow as tf

from src.utils.data_utils import load_labels_from_csv, prepare_dual_dataset
from src.embedding.syntaxic_training import global_training_pipeline
from src.embedding.embedding import load_codebert_tokenizer
from src.model.dh_cnn import build_final_dh_cnn
from src.utils.config import Config

def train_single_run(X_syn_train_list, X_sem_train, y_train,
                     X_syn_test_list, X_sem_test, y_test,
                     epochs=Config.EPOCHS, batch_size=Config.BATCH_SIZE):
    """
    Trains one fresh model instance and returns its metrics.
    Dynamic inputs are passed as lists to accommodate both modes.
    Called repeatedly by the main loop for averaging results over N_RUNS.
    """
    classes = np.unique(y_train)
    weights = compute_class_weight(class_weight="balanced", classes=classes, y=y_train)
    class_weight_dict = dict(zip(classes.tolist(), weights.tolist()))

    model = build_final_dh_cnn()

    # Combine dynamic syntax inputs with fixed semantic inputs
    train_inputs = X_syn_train_list + [X_sem_train]
    test_inputs = X_syn_test_list + [X_sem_test]

    model.fit(
        x=train_inputs,
        y=y_train,
        epochs=epochs,
        batch_size=batch_size,
        validation_split=0.1,
        class_weight=class_weight_dict,
        verbose=0
    )

    y_proba = model.predict(test_inputs, verbose=0)
    y_pred  = np.argmax(y_proba, axis=1)

    precision = precision_score(y_test, y_pred, zero_division=0)
    recall    = recall_score(y_test, y_pred, zero_division=0)
    f1        = f1_score(y_test, y_pred, zero_division=0)
    auc       = roc_auc_score(y_test, y_proba[:, 1]) if len(np.unique(y_test)) > 1 else float("nan")

    print(f"  P={precision:.4f}  R={recall:.4f}  F1={f1:.4f}  AUC={auc:.4f}")
    return np.array([precision, recall, f1, auc])

N_RUNS = 30

if __name__ == "__main__":
    # TRAIN_DATASET_PATH  = "src/data/raw/test_java/log4j/log4j-v_1_1/log4j-v_1_1/"
    # TRAIN_CSV_PATH      = "src/data/raw/test_java/log4j/log4j-v_1_1/log4j-1.1.csv"
    # TEST_DATASET_PATH   = "src/data/raw/test_java/lucene20/org/"
    # TEST_CSV_PATH       = "src/data/raw/test_java/lucene20/lucene-2.0.csv"

    # TRAIN_DATASET_PATH = "src/data/raw/test_java/jedit40source/jEdit/"
    # TRAIN_CSV_PATH     = "src/data/raw/test_java/jedit40source/jedit-4.0.csv"
    # TEST_DATASET_PATH  = "src/data/raw/test_java/jedit41source/jEdit"
    # TEST_CSV_PATH      = "src/data/raw/test_java/jedit41source/jedit-4.1.csv"
    
    
    TRAIN_DATASET_PATH = "src/data/raw/test_java/lucene20/org/"
    TRAIN_CSV_PATH     = "src/data/raw/test_java/lucene20/lucene-2.0.csv"
    TEST_DATASET_PATH  = "src/data/raw/test_java/lucene22/org/"
    TEST_CSV_PATH      = "src/data/raw/test_java/lucene22/lucene-2.2.csv"


    
    # TRAIN_DATASET_PATH = "src/data/raw/test_java/synapse/synapse11/"
    # TRAIN_CSV_PATH     = "src/data/raw/test_java/synapse/synapse11/synapse-1.1.csv"

    # TRAIN_DATASET_PATH = "src/data/raw/test_java/ANT/ant-1.6/apache-ant-1.6.0/"
    # TRAIN_CSV_PATH     = "src/data/raw/test_java/ANT/ant-1.6/ant-1.6-Unified.csv"

    # TEST_DATASET_PATH  = "src/data/raw/test_java/POI/POI313/org/"
    # TEST_CSV_PATH      = "src/data/raw/test_java/POI/POI313/poi-3.0.csv"

    # TEST_DATASET_PATH  = "src/data/raw/test_java/synapse/synapse12/"
    # TEST_CSV_PATH      = "src/data/raw/test_java/synapse/synapse12/synapse-1.2.csv"




    # TRAIN_DATASET_PATH  = "src/data/raw/test_java/lucene22/org/"
    # TRAIN_CSV_PATH      = "src/data/raw/test_java/lucene22/lucene-2.2.csv"

    
    # TEST_DATASET_PATH = "src/data/raw/test_java/ANT/ant-1.6/apache-ant-1.6.0/"
    # TEST_CSV_PATH     = "src/data/raw/test_java/ANT/ant-1.6/ant-1.6-Unified.csv"


    
    # TRAIN_DATASET_PATH  = "src/data/raw/test_java/log4j/log4j-v_1_1/log4j-v_1_1/"
    # TRAIN_CSV_PATH      = "src/data/raw/test_java/log4j/log4j-v_1_1/log4j-1.1.csv"

    
    # TEST_DATASET_PATH = "src/data/raw/test_java/lucene20/org/"
    # TEST_CSV_PATH     = "src/data/raw/test_java/lucene20/lucene-2.0.csv"


    print(f"Starting Training Pipeline in Mode: {Config.SYNTAX_MODE.upper()}")

    # Determine processor (Tokenizer vs W2V Model)
    if Config.SYNTAX_MODE == "codebert":
        processor = load_codebert_tokenizer()
    else:
        processor = global_training_pipeline(
            TRAIN_DATASET_PATH, source_type="folder",
            embedding_dim=Config.W2V_EMBEDDING_DIM,
            window=Config.W2V_WINDOW,
            min_count=Config.W2V_MIN_COUNT
        )
        if processor is None:
            print("ERROR: Word2Vec training failed. Aborting.")
            exit()

    train_labels = load_labels_from_csv(TRAIN_CSV_PATH)
    test_labels  = load_labels_from_csv(TEST_CSV_PATH)

    # Data Generation
    X_syn_1_train, X_syn_2_train, X_sem_train, y_train = prepare_dual_dataset(
        TRAIN_DATASET_PATH, train_labels, processor
    )
    X_syn_1_test, X_syn_2_test, X_sem_test, y_test = prepare_dual_dataset(
        TEST_DATASET_PATH, test_labels, processor
    )

     # Structure inputs dynamically
    x_syn_train_list = [X_syn_1_train, X_syn_2_train] if Config.SYNTAX_MODE == "codebert" else [X_syn_1_train]
    x_syn_test_list = [X_syn_1_test, X_syn_2_test] if Config.SYNTAX_MODE == "codebert" else [X_syn_1_test]

    
    # N_RUNS - Node2Vec and model weights are re-randomized for each run to ensure robustness of results.
    results = np.zeros(4)

    for i in range(N_RUNS):
        print(f"Run {i+1}/{N_RUNS}")

        metrics = train_single_run(
            x_syn_train_list, X_sem_train, y_train,
            x_syn_test_list,  X_sem_test,  y_test
        )
        results += metrics

    avg = results / N_RUNS
    print(f"\nAverage over {N_RUNS} runs ({Config.SYNTAX_MODE.upper()} mode):")
    print(f"  Precision : {avg[0]:.4f}")
    print(f"  Recall    : {avg[1]:.4f}")
    print(f"  F1        : {avg[2]:.4f}")
    print(f"  AUC       : {avg[3]:.4f}")