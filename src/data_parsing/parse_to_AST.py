import javalang

def parse_java_to_ast_vectors(java_source_code):
    vectors = []
    try:
        tree = javalang.parse.parse(java_source_code)
        for path, node in tree:
            vi = []
            if node:
                node_type = type(node).__name__
                vi.append(node_type)
            vectors.append(vi)
        return vectors
    except javalang.parser.JavaSyntaxError:
        # Silently skip files that the parser cannot understand
        return []
    except Exception as e:
        # Catch unexpected errors (like memory issues or encoding)
        print(f"Skipping file due to error: {e}")
        return []