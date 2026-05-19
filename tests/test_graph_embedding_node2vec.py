import pytest
import networkx as nx
import numpy as np
from src.data_parsing.parse_to_CFG import parse_java_to_cfg
from src.data_parsing.parse_to_DDG import parse_cfg_to_ddg
from src.embedding.graph_embedding_node2vec import GraphEmbedder, generate_semantic_embeddings

def test_node2vec_dimensions():
    """
        Test: respect the dimension constraint
    """
    code = """
    public class Test {
        public void method() {
            int x = 10;
            if (x > 5) {
                x = 0;
            }
        }
    }
    """
    cfg = parse_java_to_cfg(code)
    ddg = parse_cfg_to_ddg(cfg)

    embedder = GraphEmbedder(dimensions=100)
    cfg_embeddings = embedder.embed_graph(cfg, walk_length=5, num_walks=10)

    ddg_embeddings = embedder.embed_graph(ddg, walk_length=5, num_walks=10)

    assert len(cfg_embeddings) > 0, "Embeddings dictionary should not be empty."
    assert len(ddg_embeddings) > 0, "Embeddings dictionary should not be empty."
    
    for node_id, vector in cfg_embeddings.items():
        assert isinstance(vector, np.ndarray), f"Vector for {node_id} is not a numpy array."
        assert vector.shape == (100,), f"Expected dimension 100, got {vector.shape[0]} for node {node_id}."
    
    for node_id, vector in ddg_embeddings.items():
        assert isinstance(vector, np.ndarray), f"Vector for {node_id} is not a numpy array."
        assert vector.shape == (100,), f"Expected dimension 100, got {vector.shape[0]} for node {node_id}."

def test_unified_embedding_pipeline():
    """
        Test: Ensures CFG and DDG can be embedded and merged into a single (n, 100) matrix
    """
    code = """
    public class Test {
        public void method(int a) {
            int x = a + 1;
        }
    }
    """
    cfg = parse_java_to_cfg(code)
    ddg = parse_cfg_to_ddg(cfg)
    
    cfg_vecs, ddg_vecs, final_matrix = generate_semantic_embeddings(cfg, ddg)
    
    assert isinstance(cfg_vecs, dict)
    assert isinstance(ddg_vecs, dict)
    assert len(cfg_vecs) == len(cfg.nodes())
    assert len(ddg_vecs) == len(ddg.nodes())
    
    n_nodes = len(cfg.nodes())
    assert isinstance(final_matrix, np.ndarray), "The final fused structure must be a numpy array."
    assert final_matrix.shape == (n_nodes, 100), f"Expected shape ({n_nodes}, 100), got {final_matrix.shape}"