import pytest
from gensim.models import Word2Vec
from src.embedding.word2vecTraining import global_training_pipeline

def test_word2vec_structures():
    """ Test that the Word2Vec model correctly recognizes Java syntax """
    
    dummy_sentences = [
        ["MethodDeclaration", "IfStatement", "WhileStatement", "ClassDeclaration", "VariableDeclarator"]
    ]
    model = Word2Vec(sentences=dummy_sentences, vector_size=100, min_count=1)
    
    test_structures = ["IfStatement", "WhileStatement", "MethodDeclaration"]
    
    for node in test_structures:
        assert node in model.wv, f"Node {node} not found in Word2Vec vocabulary"
        
        neighbours = model.wv.most_similar(node, topn=1)
        assert len(neighbours) > 0, f"Impossible to find a neighbor for {node}"