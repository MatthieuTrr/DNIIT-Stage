from src.embedding.word2vecTraining import global_training_pipeline, load_dataset_from_file
from src.data_parsing.parse_to_AST import  parse_java_to_ast_vectors
import numpy as np
from src.utils.config import Config

def prepare_embedding(code,w2v_model, sequence_length=Config.MAX_TOKENS, embedding_dim=Config.EMBEDDING_DIM):
    tokens = parse_java_to_ast_vectors(code)
    vecs = [w2v_model.wv[token[0]] for token in tokens if token and token[0] in w2v_model.wv]
    vectors = vecs[:sequence_length]
    matrix = np.zeros((sequence_length, embedding_dim))
    for i, v in enumerate(vectors):
        matrix[i] = v
    return matrix
