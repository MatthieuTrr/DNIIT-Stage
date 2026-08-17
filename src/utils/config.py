class Config:
    SYNTAX_MODE = "codebert"  # Options: "word2vec" or "codebert"

    # --- Node2Vec / Semantic Parameters ---
    MAX_TOKENS = 50
    EMBEDDING_DIM = 50
    N2V_WALK_LENGTH = 10
    N2V_NUM_WALKS = 10
    N2V_batch_words = 4
    N2V_P = 1.0
    N2V_Q = 1.0

    # --- Word2Vec Parameters (Baseline) ---
    W2V_MAX_TOKENS = 50
    W2V_EMBEDDING_DIM = 100
    W2V_WINDOW = 5
    W2V_MIN_COUNT = 5

    # --- CodeBERT Parameters (Transformer) ---
    CODEBERT_MAX_TOKENS = 512
    CODEBERT_MODEL_NAME = "microsoft/codebert-base"
    CODEBERT_HIDDEN_DIM = 768

    # --- CNN Parameters ---
    CNN_FILTERS = 128
    KERNEL_SIZE = 5
    DROPOUT_RATE = 0.5
    FC_UNITS = 10

    # --- Training Parameters ---
    EPOCHS = 100
    BATCH_SIZE = 32
    WINDOW_SIZE = 5
    MIN_COUNT = 5