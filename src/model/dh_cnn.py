import tensorflow as tf
from tensorflow.keras.layers import Concatenate, Dense
from tensorflow.keras.models import Model

from src.model.cnn_ast import build_syntax_CNN
from src.model.semantic_DH_CNN import build_semantic_level_cnn 

def build_final_dh_cnn():
    
    ast_model = build_syntax_CNN()
    semantic_model = build_semantic_level_cnn()

    ast_input = ast_model.input 
    semantic_input = semantic_model.input

    Cs = ast_model.output
    Cg = semantic_model.output

    syntax_gate=Dense(10,activation="sigmoid", use_bias=True, name="syntax_gate")(Cs)
    semantic_gate =Dense(10, activation="sigmoid", use_bias=True, name="semantic_gate")(Cg)
    Gm=Concatenate(name="merging_layer")([syntax_gate,semantic_gate])
    final_output=Dense(2, activation="softmax", name="final_classification")(Gm)
    final_model = Model(
        inputs=[ast_input, semantic_input], 
        outputs=final_output, 
        name="Full_DH_CNN_Model"
    )

    final_model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )

    return final_model