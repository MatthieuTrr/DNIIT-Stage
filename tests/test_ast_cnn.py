import pytest
import numpy as np
import tensorflow as tf
from src.utils.config import Config
from src.model.cnn_ast import build_syntax_CNN

def test_syntaxic_cnn_architecture():
    """
        Test: Validates the architecture and input/output shapes of the Syntax-level DH-CNN
    """
    sequence_length = Config.MAX_TOKENS
    embedding_dim = Config.AST_EMBEDDING  # ← était Config.EMBEDDING_DIM
    batch_size = 2

    model = build_syntax_CNN(
        MAX_TOKENS=sequence_length,
        EMBEDDING_DIM=embedding_dim
    )

    rand_input = np.random.rand(batch_size, sequence_length, embedding_dim)
    output = model(rand_input)

    assert isinstance(output, tf.Tensor), "Output should be a TensorFlow Tensor."
    assert output.shape == (batch_size, Config.CNN_FILTERS), \
        f"Expected shape ({batch_size}, {Config.CNN_FILTERS}), but got {output.shape}"

    print("\n--- Syntax-level DH-CNN Architecture ---")
    model.summary()