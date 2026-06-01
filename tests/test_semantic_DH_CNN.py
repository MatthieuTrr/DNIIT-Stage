import pytest
import numpy as np
import tensorflow as tf
from src.model.semantic_DH_CNN import build_semantic_level_cnn

def test_semantic_cnn_architecture():
    """
        Test: Validates the architecture and input/output shapes of the Semantic-level DH-CNN
    """

    sequence_length = 50
    embedding_dim = 50
    batch_size = 2 
    
    model = build_semantic_level_cnn(
        sequence_length=sequence_length, 
        embedding_dim=embedding_dim
    )
    
    rand_input = np.random.rand(batch_size, sequence_length, embedding_dim)

    output = model(rand_input)

    assert isinstance(output, tf.Tensor), "Output should be a TensorFlow Tensor."
    assert output.shape == (batch_size, 10), f"Expected shape ({batch_size}, 10), but got {output.shape}"
    final_layer = model.layers[-1]
    assert final_layer.activation.__name__ == 'sigmoid', "The final Fully Connected layer MUST use Sigmoid activation."

    print("\n--- Semantic-level DH-CNN Architecture ---")
    model.summary()