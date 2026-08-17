import pytest
import numpy as np
import tensorflow as tf
from src.model.dh_cnn import build_final_dh_cnn
from src.utils.config import Config

def test_final_dh_cnn_w2v_architecture(monkeypatch):
    """
        Test: Validates that the global DH-CNN model dynamically assembled for Word2Vec (2 Inputs).
    """
    monkeypatch.setattr(Config, "SYNTAX_MODE", "word2vec")
    model = build_final_dh_cnn()

    assert len(model.inputs) == 2, "The final model must take exactly 2 inputs."

    batch_size = 2

    dummy_syntaxic = np.random.rand(batch_size, Config.W2V_MAX_TOKENS, Config.W2V_EMBEDDING_DIM) 
    dummy_semantic = np.random.rand(batch_size, Config.MAX_TOKENS, Config.EMBEDDING_DIM) 

    output = model([dummy_syntaxic, dummy_semantic])
    assert output.shape == (batch_size, 2)
    assert model.layers[-1].activation.__name__ == 'softmax'

def test_final_dh_cnn_codebert_architecture(monkeypatch):
    """
        Test: Validates that the global DH-CNN model dynamically assembled for CodeBERT (3 Inputs).
    """
    monkeypatch.setattr(Config, "SYNTAX_MODE", "codebert")
    model = build_final_dh_cnn()

    assert len(model.inputs) == 3, "CodeBERT model must take exactly 3 inputs."
    batch_size = 2

    dummy_input_ids = np.random.randint(0, 100, size=(batch_size, Config.CODEBERT_MAX_TOKENS))
    dummy_attention_mask = np.ones((batch_size, Config.CODEBERT_MAX_TOKENS), dtype=np.int32)
    dummy_semantic = np.random.rand(batch_size, Config.MAX_TOKENS, Config.EMBEDDING_DIM)

    output = model([dummy_input_ids, dummy_attention_mask, dummy_semantic])
    assert output.shape == (batch_size, 2)
    assert model.layers[-1].activation.__name__ == 'softmax'