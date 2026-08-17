from src.data_parsing.parse_to_AST import parse_java_to_ast_vectors
import numpy as np
from transformers import AutoTokenizer
from src.utils.config import Config

def ast_tokens_to_text(code):
    """Convert the AST tokens list into a string for CodeBERT."""
    tokens = parse_java_to_ast_vectors(code)
    if tokens and isinstance(tokens[0], tuple):
        return " ".join([t[0] for t in tokens])
    return " ".join(tokens)

def load_codebert_tokenizer(model_name=Config.CODEBERT_MODEL_NAME):
    """Load the HuggingFace tokenizer for CodeBERT."""
    return AutoTokenizer.from_pretrained(model_name)

def prepare_codebert_embedding(code, tokenizer, max_length=Config.CODEBERT_MAX_TOKENS):
    """Prepare the input IDs and attention mask for the CodeBERT model."""
    ast_text = ast_tokens_to_text(code)
    encoded = tokenizer(
        ast_text,
        padding="max_length",
        truncation=True,
        max_length=max_length,
        return_tensors="np"
    )
    return encoded["input_ids"][0], encoded["attention_mask"][0]

def prepare_word2vec_embedding(code, w2v_model, sequence_length=Config.W2V_MAX_TOKENS, embedding_dim=Config.W2V_EMBEDDING_DIM):
    """Prepare the semantic matrix for the CNN using Word2Vec."""
    tokens = parse_java_to_ast_vectors(code)
    
    vecs = []
    for token in tokens:
        t_val = token[0] if isinstance(token, tuple) else token
        if t_val in w2v_model.wv:
            vecs.append(w2v_model.wv[t_val])
            
    vectors = vecs[:sequence_length]
    matrix = np.zeros((sequence_length, embedding_dim))
    for i, v in enumerate(vectors):
        matrix[i] = v
    return matrix, None

def prepare_syntax_embedding(code, model_or_tokenizer):
    """Route to the correct embedding method based on Config.SYNTAX_MODE."""

    if Config.SYNTAX_MODE == "codebert":
        return prepare_codebert_embedding(code, model_or_tokenizer)
    elif Config.SYNTAX_MODE == "word2vec":
        return prepare_word2vec_embedding(code, model_or_tokenizer)
    else:
        raise ValueError("Invalid SYNTAX_MODE. Choose either 'codebert' or 'word2vec'.")