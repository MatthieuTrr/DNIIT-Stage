import pytest
import numpy as np
import tensorflow as tf
from src.model.dh_cnn import build_final_dh_cnn
from src.utils.config import Config

def test_final_dh_cnn_architecture():
    """
        Test: Validates that the global DH-CNN model correctly assembles both branches,
        takes 2 inputs (Syntaxic and Semantic), and outputs a binary prediction.
    """
    model = build_final_dh_cnn()
    
    assert len(model.inputs) == 2, "The final model must take exactly 2 inputs (Syntaxic and Semantic)."
    
    batch_size = 2
    sequence_length = Config.MAX_TOKENS
    embedding_dim = Config.EMBEDDING_DIM 
    
    dummy_syntaxic = np.random.rand(batch_size, sequence_length, embedding_dim)
    dummy_semantic = np.random.rand(batch_size, sequence_length, embedding_dim)
    
    output = model([dummy_syntaxic, dummy_semantic])
    
    assert isinstance(output, tf.Tensor), "Output should be a TensorFlow Tensor."
    assert output.shape == (batch_size, 2), f"Expected shape ({batch_size}, 2), but got {output.shape}"
    
    final_layer = model.layers[-1]
    assert final_layer.activation.__name__ == 'softmax', "The final layer MUST use Softmax activation."
    
    print("\n--- Full DH-CNN Architecture ---")
    model.summary()