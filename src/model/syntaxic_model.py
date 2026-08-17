import tensorflow as tf
from tensorflow import keras
from transformers import TFAutoModel, RobertaConfig
from src.utils.config import Config

def _load_codebert_backbone(model_name=Config.CODEBERT_MODEL_NAME):
    """Loads the CodeBERT model or initializes a blank architecture if unavailable."""
    try:
        return TFAutoModel.from_pretrained(model_name)
    except Exception:
        config = RobertaConfig(
            vocab_size=50265,
            hidden_size=Config.CODEBERT_HIDDEN_DIM,
            intermediate_size=3072,
            num_hidden_layers=12,
            num_attention_heads=12,
            max_position_embeddings=514,
            type_vocab_size=1,
        )
        return TFAutoModel.from_config(config)

def build_w2v_cnn(max_tokens=Config.W2V_MAX_TOKENS, 
                  embedding_dim=Config.W2V_EMBEDDING_DIM, 
                  filters=Config.CNN_FILTERS, 
                  kernel_size=Config.KERNEL_SIZE, 
                  dropout=Config.DROPOUT_RATE):
    """Constructs the baseline CNN branch for Word2Vec embeddings."""
    inputs = keras.Input(shape=(max_tokens, embedding_dim), name="ast_input")
    cnn = keras.layers.Conv1D(filters=filters, kernel_size=kernel_size, activation="relu", name="syntaxic_conv1d")(inputs)
    cnn = keras.layers.GlobalMaxPooling1D(name="syntaxic_max_pool")(cnn)
    output = keras.layers.Dropout(dropout, name="syntaxic_dropout")(cnn)
    return keras.Model(inputs=inputs, outputs=output, name="syntax_level_w2v_cnn")

class CodeBERTWrapper(keras.layers.Layer):
    """
        Keras custom layer to encapsulate the Hugging Face model.
        Protects the HF model from strict KerasTensor checks.
    """
    def __init__(self, transformer_model, **kwargs):
        super().__init__(**kwargs)
        self.transformer = transformer_model

    def call(self, inputs):
        input_ids = inputs[0]
        attention_mask = inputs[1]

        outputs = self.transformer(input_ids=input_ids, attention_mask=attention_mask)
        return outputs[0]

def build_codebert_transformer(max_tokens=Config.CODEBERT_MAX_TOKENS, filters=Config.CNN_FILTERS, dropout=Config.DROPOUT_RATE):
    """
        Constructs the Syntax level branch using CodeBERT.
    """
    input_ids = keras.Input(shape=(max_tokens,), dtype=tf.int32, name="ast_input_ids")
    attention_mask = keras.Input(shape=(max_tokens,), dtype=tf.int32, name="ast_attention_mask")

    transformer = _load_codebert_backbone()
    
    inputs_dict = {
        "input_ids": input_ids, 
        "attention_mask": attention_mask
    }
    
    sequence_output = CodeBERTWrapper(transformer, name="codebert_layer")([input_ids, attention_mask])
    cls_token = sequence_output[:, 0, :]

    projection = keras.layers.Dense(filters, activation="relu", name="syntax_projection")(cls_token)
    output = keras.layers.Dropout(dropout, name="syntax_dropout")(projection)

    model = keras.Model(inputs=[input_ids, attention_mask], outputs=output, name="syntax_level_codebert")
    return model

def build_syntax_model():
    """Returns the correct syntax model based on Config.SYNTAX_MODE."""
    if Config.SYNTAX_MODE == "codebert":
        return build_codebert_transformer()
    elif Config.SYNTAX_MODE == "word2vec":
        return build_w2v_cnn()
    else:
        raise ValueError("Invalid SYNTAX_MODE in Config.")