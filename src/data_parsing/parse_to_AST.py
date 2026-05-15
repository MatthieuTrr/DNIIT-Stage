import javalang

def parse_java_to_ast_vectors(java_source_code):
    vectors = []
    tree = javalang.parse.parse(java_source_code)
    for path, node in tree:
        vi = []
        if node:
            node_type = type(node).__name__
            vi.append(node_type)
        vectors.append(vi)
    return vectors