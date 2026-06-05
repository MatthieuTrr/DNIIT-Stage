import tensorflow as tf
from tensorflow import keras
from src.utils.config import Config

def build_syntax_CNN(
        MAX_TOKENS = Config.MAX_TOKENS,    
        EMBEDDING_DIM = Config.EMBEDDING_DIM,   
        CNN_FILTERS   = Config.CNN_FILTERS,   
        KERNEL_SIZE   = Config.KERNEL_SIZE,    
        DROPOUT_RATE  = Config.DROPOUT_RATE,
        FC_UNITS      = Config.FC_UNITS
        ):
    """
        Constructs the Syntax-level DH-CNN branch (AST)
    """
    inputs=keras.Input(shape=(MAX_TOKENS,EMBEDDING_DIM), name="ast_input")
    
    cnn = keras.layers.Conv1D(filters=CNN_FILTERS, kernel_size=KERNEL_SIZE, activation="relu", name="syntaxic_conv1d")(inputs)
    cnn = keras.layers.GlobalMaxPooling1D(name="syntaxic_max_pool")(cnn)
    cnn = keras.layers.Dropout(DROPOUT_RATE, name="syntaxic_dropout")(cnn)
    outputs = keras.layers.Dense(FC_UNITS, activation="sigmoid", name="syntaxic_fully_connected")(cnn)

    model = keras.Model(inputs=inputs, outputs=outputs, name="syntax_level_DH_CNN")
    return model