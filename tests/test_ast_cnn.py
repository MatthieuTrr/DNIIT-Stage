import pytest
import numpy as np
import tensorflow as tf
from src.utils.config import Config
from src.model.syntaxic_model import build_w2v_cnn, build_codebert_transformer

def test_syntaxic_w2v_cnn_architecture():
    """
        Test: Validates the architecture and input/output shapes of the Word2Vec CNN branch.
    """
    sequence_length = Config.W2V_MAX_TOKENS
    embedding_dim = Config.W2V_EMBEDDING_DIM
    batch_size = 2

    model = build_w2v_cnn(
        max_tokens=sequence_length,
        embedding_dim=embedding_dim
    )

    rand_input = np.random.rand(batch_size, sequence_length, embedding_dim)
    output = model(rand_input)

    assert isinstance(output, tf.Tensor), "Output should be a TensorFlow Tensor."
    assert output.shape == (batch_size, Config.CNN_FILTERS), \
        f"Expected shape ({batch_size}, {Config.CNN_FILTERS}), but got {output.shape}"

    print("\n--- Syntax-level DH-CNN Architecture ---")
    model.summary()

def test_syntaxic_codebert_architecture():
    """
        Test: Validates the architecture of the CodeBERT Transformer branch.
    """
    sequence_length = Config.CODEBERT_MAX_TOKENS
    batch_size = 2

    model = build_codebert_transformer(max_tokens=sequence_length)
    input_ids = np.random.randint(0, 100, size=(batch_size, sequence_length))
    attention_mask = np.ones((batch_size, sequence_length), dtype=np.int32)
    output = model([input_ids, attention_mask])

    assert isinstance(output, tf.Tensor), "Output should be a TensorFlow Tensor."
    assert output.shape == (batch_size, Config.CNN_FILTERS), \
        f"Expected shape ({batch_size}, {Config.CNN_FILTERS}), but got {output.shape}"
    print("\n--- Syntax-level CodeBERT Architecture ---")
    model.summary()