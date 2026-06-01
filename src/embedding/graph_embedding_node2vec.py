import networkx as nx
from node2vec import Node2Vec
import numpy as np

class GraphEmbedder:
    """
        To handle conversion of NetworkX graphs (CFG + DDG) into numerical vectors using Node2Vec algo
    """
    def __init__(self, dimensions=50, window=5, min_count=1, batch_words=4, p=1.0, q=1.0):
        """        
            Args:
                dimensions: Vector size
                window: Context window size for Word2Vec training
                min_count: Minimum word frequency
                batch_words: Batch words parameter for training
                p: Return parameter for Node2Vec
                q: In-out parameter for Node2Vec
        """
        self.dimensions = dimensions
        self.window = window
        self.min_count = min_count
        self.batch_words = batch_words
        self.p = p
        self.q = q

    def embed_graph(self, graph: nx.DiGraph, walk_length=80, num_walks=10):
        """
            Generates random walks and trains embedding model
        
            Args:
                graph: CFG or DDG
                walk_length: The length of each random walk
                num_walks: Number of random walks/node
                
            Returns:
                dict: A dictionary mapping node IDs (str) to their embedding vectors (numpy arrays)
        """
        if not graph or len(graph.nodes()) == 0:
            return {}
        n2v_generator = Node2Vec(
            graph,
            dimensions=self.dimensions,
            walk_length=walk_length,
            num_walks=num_walks,
            p=self.p,
            q=self.q,
            workers=1,
            quiet=True
        )

        model = n2v_generator.fit(
            window=self.window,
            min_count=self.min_count,
            batch_words=self.batch_words
        )

        embeddings = {}
        for node in graph.nodes():
            node_id_str = str(node)
            if node_id_str in model.wv:
                embeddings[node] = model.wv[node_id_str]
            else:
                embeddings[node] = np.zeros(self.dimensions)

        return embeddings

def align_and_fuse_embeddings(cfg_vectors: dict, ddg_vectors: dict, ordered_nodes: list, dimensions: int = 50):
    """
        Merging CFG and DDG embeddings by aligning nodes in a specific order
        Returns a Numpy matrix of size (n, dimensions).
    """
    n = len(ordered_nodes)
    final_matrix = np.zeros((n, dimensions))
    
    for i, node in enumerate(ordered_nodes):
        vec_c = cfg_vectors.get(node, np.zeros(dimensions))
        vec_d = ddg_vectors.get(node, np.zeros(dimensions))
        
        final_matrix[i] = (vec_c + vec_d) / 2.0
        
    return final_matrix

def generate_semantic_embeddings(cfg: nx.DiGraph, ddg: nx.DiGraph):
    embedder = GraphEmbedder(dimensions=50)
    cfg_vectors = embedder.embed_graph(cfg)
    ddg_vectors = embedder.embed_graph(ddg)

    ordered_nodes = list(cfg.nodes())
    final_graph_matrix = align_and_fuse_embeddings(cfg_vectors, ddg_vectors, ordered_nodes, embedder.dimensions)

    
    return cfg_vectors, ddg_vectors, final_graph_matrix