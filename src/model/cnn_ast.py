import os
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score
import tensorflow as tf
from tensorflow import keras
from src.embedding.word2vecTraining import global_training_pipeline
from src.data_parsing.parse_to_AST import parse_java_to_ast_vectors
import pandas as pd
from src.embedding.embedding import prepare_embedding

def build_syntax_CNN(
        MAX_TOKENS = 50,    
        EMBEDDING_DIM = 100,   
        CNN_FILTERS   = 128,   
        KERNEL_SIZE   = 5,    
        DROPOUT_RATE  = 0.5,
        FC_UNITS      = 10,   
        EPOCHS        = 50,
        BATCH_SIZE    = 32
        ):
    inputs=keras.Input(shape=(MAX_TOKENS,EMBEDDING_DIM), name="ast_input")
    cnn=keras.layers.Conv1D(filters=CNN_FILTERS,kernel_size=KERNEL_SIZE,activation="relu",name="Conv1D")(inputs)
    cnn = keras.layers.GlobalMaxPooling1D(name="global_max_pool")(cnn)
    cnn = keras.layers.Dropout(DROPOUT_RATE, name="dropout")(cnn)
    cnn = keras.layers.Dense(FC_UNITS, activation="sigmoid", name="fc")(cnn)

    # TEMPORAIRE to test AST before adding the merging part with the semantic CNN

    model = keras.Model(inputs=inputs, outputs=cnn, name="DH-CNN_Syntax")
    return model






