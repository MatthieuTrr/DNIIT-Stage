import os
import json
import javalang
from gensim.models import Word2Vec

# Code from a precedent issue
def parse_java_to_ast_sequence(java_source_code):
    sequence = []
    tree = javalang.parse.parse(java_source_code)
    for path, node in tree:
        if node:
            sequence.append(type(node).__name__)
    return sequence

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
        for ligne in f:
            data=json.loads(ligne)
            if code_key in data:
                loaded_codes.append(data[code_key])
    print(f"{len(loaded_codes)} Java source files have been found and loaded.")
    print(f"\n")
    return loaded_codes

# This is the complete pipeline: charges, parses, trains and validates the model.
def global_training_pipeline(source_data, source_type="repertory", embedding_dim=50, window=25, max_files=5000):
    if source_type=="repertory":
        java_programs=load_dataset_from_file(source_data)
    elif source_type=="jsonl":
        java_programs=load_dataset_from_jsonl(source_data)
    else:
        raise ValueError("source_type must be 'repertory' or 'jsonl'.")
    if not java_programs:
        print("Error: no Java files loaded. Stopping pipeline...")
        return None
    sentences=[]
    for idx, code in enumerate(java_programs):
        tokens=parse_java_to_ast_sequence(code)
        if tokens:
            sentences.append(tokens)
            if (idx+1)%max_files==0: # we stop at 5000 source codes by default
                print(f" ->{idx+1} analyzed programs...")

    print(f"Done extracting. {len(sentences)} sequences ready for Word2Vec training")
    model=Word2Vec(
        sentences=sentences,
        vector_size=embedding_dim,
        window=window,
        min_count=5, # ignoring rare structures
        workers=4, # Might need adjustment
        epochs=15,
        sg=0, # CBOW
        seed=42
    )

    print(f"Model Trained. Vocabulary size : {len(model.wv)}")

    # We run a quick quality check:
    print("Running quick validation test...")
    test_structures=["IfStatement","WhileStatement","MethodDeclaration"]
    for node in test_structures:
        if node in model.wv:
            neighbours=model.wv.most_similar(node,topn=3)
            print(f"\n Most similar Syntaxic nodes to [{node}] :")
            for neighbour, score in neighbours:
                print(f"   -> {neighbour:25s} (Score: {score:.4f})")
    return model