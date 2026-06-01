import os
from gensim.models import Word2Vec
from src.data_parsing.parse_to_AST import parse_java_to_ast_vectors

def load_dataset_from_file(file_path):
    """ Load the dataset from given path """
    loaded_codes=[]
    for root, _,files in os.walk(file_path):
        for file in files:
            if file.endswith('.java'):
                full_path=os.path.join(root, file)
                with open(full_path,'r', encoding='utf-8', errors='ignore') as f:
                    code=f.read()
                    if code.strip():
                        loaded_codes.append(code)
    print(f"{len(loaded_codes)} Java source files have been found and loaded.\n")
    return loaded_codes

def global_training_pipeline(source_data,source_type="folder",embedding_dim=100,window=5,min_count=5):
    """ Complete pipeline: loads Java code, parses it into AST vectors, trains and validates the model."""
    if source_type=="folder":
        java_programs=load_dataset_from_file(source_data)
    else:
        raise ValueError("source_type must be 'folder' or 'jsonl'.")
    
    if not java_programs:
        print("Error: no Java files loaded. Stopping pipeline...")
        return None
        
    sentences=[]
    for idx, code in enumerate(java_programs):
        tokens=parse_java_to_ast_vectors(code)
        tokens=[t[0] for t in tokens if t]
        if tokens:
            sentences.append(tokens)

    print(f"Done extracting. {len(sentences)} sequences ready for Word2Vec training")
    model=Word2Vec(
        sentences=sentences,
        vector_size=embedding_dim,
        window=window,
        min_count=min_count,
        workers=4,
        epochs=100,
        sg=0, # CBOW
        seed=42,
        batch_words=50,
        negative=10
    )

    print(f"Model Trained. Vocabulary size : {len(model.wv)}")
    return model