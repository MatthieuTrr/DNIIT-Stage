import tensorflow as tf
from tensorflow.keras.layers import Concatenate, Dense
from tensorflow.keras.models import Model

from src.model.syntaxic_model import build_syntax_model
from src.model.semantic_DH_CNN import build_semantic_level_cnn 
from src.utils.config import Config

def build_final_dh_cnn():
    """Assembles the final Dual-Branch architecture dynamically."""
    ast_model = build_syntax_model()
    semantic_model = build_semantic_level_cnn()

    # Dynamic input resolution based on mode
    if Config.SYNTAX_MODE == "codebert":
        ast_input_ids, ast_attention_mask = ast_model.inputs 
        model_inputs = [ast_input_ids, ast_attention_mask, semantic_model.input]
    else:
        ast_input = ast_model.input 
        model_inputs = [ast_input, semantic_model.input]

    Cs = ast_model.output
    Cg = semantic_model.output

    syntax_gate = Dense(10, activation="sigmoid", use_bias=True, name="syntax_gate")(Cs)
    semantic_gate = Dense(10, activation="sigmoid", use_bias=True, name="semantic_gate")(Cg)
    
    Gm = Concatenate(name="merging_layer")([syntax_gate, semantic_gate])
    final_output = Dense(2, activation="softmax", name="final_classification")(Gm)
    
    final_model = Model(
        inputs=model_inputs, 
        outputs=final_output, 
        name=f"Full_DH_CNN_Model_{Config.SYNTAX_MODE}"
    )

    final_model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )

    return final_model