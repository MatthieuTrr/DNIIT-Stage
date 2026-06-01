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

def test_ast():
    java_program_class_basic="""
    public class Point2D {
    int x;
    int y;
    }
    """
    java_program_class_if_statement="""
    public class Loop{
    int a;
    public static void if_statement(boolean B){
        if (B==true){
            a=2;
            }
        else{
            a=1;
            }
        }
    }
    """
    vectors_basic=parse_java_to_ast_vectors(java_program_class_basic)
    vectors_if=parse_java_to_ast_vectors(java_program_class_if_statement)
    print(f"Vecteurs de vectors_basic : ")
    for v in vectors_basic:
        print(v)
    print(f"Vecteurs de vectors_if : ")
    print(f"(On devrait voir la boucle if)")
    for v in vectors_if:
        print(v)

test_ast()
