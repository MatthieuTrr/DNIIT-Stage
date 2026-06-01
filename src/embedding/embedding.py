from word2vecTraining import global_training_pipeline, load_dataset_from_file
from src.data_parsing.parse_to_AST import  parse_java_to_ast_vectors
import numpy as np

def prepare_embedding(code,w2v_model):
    tokens = parse_java_to_ast_vectors(code)
    vecs = [w2v_model.wv[token[0]] for token in tokens if token and token[0] in w2v_model.wv]
    vectors = vecs[:50]
    matrix = np.zeros((50, 100))
    for i, v in enumerate(vectors):
        matrix[i] = v
    return matrix
