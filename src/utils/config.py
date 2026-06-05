# src/config.py

class Config:
    # --- Dimensions ---
    MAX_TOKENS = 50
    EMBEDDING_DIM = 50

    # --- CNN Parameters ---
    CNN_FILTERS = 128
    KERNEL_SIZE = 5
    DROPOUT_RATE = 0.5
    FC_UNITS = 10
    
    # --- Word2Vec/Node2Vec param ---
    W2V_WINDOW = 5
    W2V_MIN_COUNT = 1
    N2V_WALK_LENGTH = 80
    N2V_NUM_WALKS = 10
    N2V_batch_words = 4
    N2V_P = 1.0
    N2V_Q = 1.0

    # --- Training Parameters ---
    EPOCHS = 50
    BATCH_SIZE = 32
    WINDOW_SIZE = 5
    MIN_COUNT = 5