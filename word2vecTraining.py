import os
import json
from gensim.models import Word2Vec
from parse_to_AST import parse_java_to_ast_vectors


# The goal of this function is to load the dataset from given path
def load_dataset_from_file(file_path):
    loaded_codes=[]
    for root, _,files in os.walk(file_path):
        for file in files:
            if file.endswith('.java'):
                full_path=os.path.join(root, file)
                with open(full_path,'r', encoding='utf-8', errors='ignore') as f:
                    code=f.read()
                    if code.strip():
                        loaded_codes.append(code)
    print(f"{len(loaded_codes)} Java source files have been found and loaded.")
    print(f"\n")
    return loaded_codes


# The goal of this function is to load the dataset from given jsonl files
def load_dataset_from_jsonl(jsonl_file_path,code_key="whole_func_string"):
    loaded_codes=[]
    with open(jsonl_file_path, 'r', encoding='utf-8') as f:
        for line in f:
            data=json.loads(line)
            if code_key in data:
                loaded_codes.append(data[code_key])
    print(f"{len(loaded_codes)} Java source files have been found and loaded.")
    print(f"\n")
    return loaded_codes

# This is the complete pipeline: loads, parses, trains and validates the model.
def global_training_pipeline(source_data,source_type="folder",embedding_dim=100,window=5,min_count=5):
    if source_type=="folder":
        java_programs=load_dataset_from_file(source_data)
    elif source_type=="jsonl":
        java_programs=load_dataset_from_jsonl(source_data)
    else:
        raise ValueError("source_type must be 'folder' or 'jsonl'.")
    if not java_programs:
        print("Error: no Java files loaded. Stopping pipeline...")
        return None
    sentences=[]
    for idx, code in enumerate(java_programs):
        tokens=parse_java_to_ast_vectors(code)
        if tokens:
            sentences.append(tokens)

    print(f"Done extracting. {len(sentences)} sequences ready for Word2Vec training")
    model=Word2Vec(
        sentences=sentences,
        vector_size=embedding_dim,
        window=window,
        min_count=min_count,
        workers=4, #
        epochs=100,
        sg=0, # CBOW
        seed=42,
        batch_words=50,
        negative=10
    )

    print(f"Model Trained. Vocabulary size : {len(model.wv)}")

    print(f"test...")
    test_structures=["IfStatement","WhileStatement","MethodDeclaration"]
    for node in test_structures:
        if node in model.wv:
            neighbours=model.wv.most_similar(node,topn=3)
            print(f"\n Most similar Syntaxic nodes to [{node}] :")
            for neighbour, score in neighbours:
                print(f"   -> {neighbour} (Score: {score})")
    return model
