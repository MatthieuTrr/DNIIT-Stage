import pytest
import numpy as np
import tensorflow as tf
from src.utils.config import Config
from src.model.semantic_DH_CNN import build_semantic_level_cnn

def test_semantic_cnn_architecture():
    """
        Test: Validates the architecture and input/output shapes of the Semantic-level DH-CNN
    """
    sequence_length = Config.MAX_TOKENS
    embedding_dim = Config.EMBEDDING_DIM
    batch_size = 2

    model = build_semantic_level_cnn(
        sequence_length=sequence_length,
        embedding_dim=embedding_dim
    )

    rand_input = np.random.rand(batch_size, sequence_length, embedding_dim)
    output = model(rand_input)

    assert isinstance(output, tf.Tensor), "Output should be a TensorFlow Tensor."
    assert output.shape == (batch_size, Config.CNN_FILTERS), \
        f"Expected shape ({batch_size}, {Config.CNN_FILTERS}), but got {output.shape}"

    print("\n--- Semantic-level DH-CNN Architecture ---")
    model.summary()