from word2vecTraining import global_training_pipeline, load_dataset_from_file
from parse_to_AST import  parse_java_to_ast_vectors
import numpy as np

# creats matrixes for CNN input
def prepare_embedding(dataset):
    model=global_training_pipeline(dataset)
    embeddings=[]
    source_files=load_dataset_from_file(dataset)
    for code in source_files:
        tokens=parse_java_to_ast_vectors(code)
        vec=[ model.wv[token] for token in tokens if token in model.wv ]
        vectors=vec[:50] 
        matrix=np.zeros((50,100))
        for i, v in enumerate(vectors):
            matrix[i]=v
        embeddings.append(matrix)
    X=np.array(embeddings)
    return X