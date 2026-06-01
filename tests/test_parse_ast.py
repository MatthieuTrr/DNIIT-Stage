import pytest
from src.data_parsing.parse_to_AST import parse_java_to_ast_vectors

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
    
    assert len(vectors_basic) > 0, "Basic parsing failed"
    types_in_if_program = [v[0] for v in vectors_if if v]
    assert "IfStatement" in types_in_if_program, "AST did not detect the 'IfStatement'"

    print(f"Vecteurs de vectors_basic : ")
    for v in vectors_basic:
        print(v)
    print(f"Vecteurs de vectors_if : ")
    print(f"(On devrait voir la boucle if)")
    for v in vectors_if:
        print(v)