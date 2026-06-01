import pytest
import networkx as nx
from src.data_parsing.parse_to_CFG import parse_java_to_cfg
from src.data_parsing.parse_to_DDG import parse_cfg_to_ddg

def test_data_dependency_flow():
    """
        Test: A variable defined in one node and used in another must create a data dependency edge in the DDG.
    """
    code = """
    public class Test {
        public void method() {
            int x = 10;
            if (x > 5) {
                int y = x + 1;
            }
        }
    }
    """
    cfg = parse_java_to_cfg(code)
    ddg = parse_cfg_to_ddg(cfg) 
    
    dependencies = [ (u, v) for u, v, d in ddg.edges(data=True) if d.get('variable') == 'x' ]
    
    assert len(dependencies) >= 1, "The dependency for variable 'x' was not found in the DDG!"

def test_no_dependency_if_no_path():
    """
    Test: No data dependency if the nodes are in independent branches.
    """
    code = """
    public class Test {
        public void method(int a) {
            int x = 0;
            if (a > 0) {
                x = 1;
            } else {
                int z = x; 
            }
        }
    }
    """
    cfg = parse_java_to_cfg(code)
    ddg = parse_cfg_to_ddg(cfg)

    node_x_0 = get_node_id_by_ast_string(ddg, "x", 0) # Finds 'x = 0'
    node_x_1 = get_node_id_by_ast_string(ddg, "x", 1) # Finds 'x = 1'
    node_z_x = get_node_id_by_ast_string(ddg, "z", "x") # Finds 'z = x'

    assert node_x_0 is not None, "Could not find node for x = 0"
    assert node_x_1 is not None, "Could not find node for x = 1"
    assert node_z_x is not None, "Could not find node for z = x"

    assert ddg.has_edge(node_x_0, node_z_x), "Path from x=0 to z=x is missing!"
    assert not ddg.has_edge(node_x_1, node_z_x), "Path from x=1 to z=x should NOT exist!"

# Helper function for tests
def get_node_id_by_ast_string(graph, var_name, value=None):
    for node_id, data in graph.nodes(data=True):
        for statement in data.get("ast_nodes", []):

            if hasattr(statement, "declarators"):
                for declarators in statement.declarators:
                    if declarators.name != var_name:
                        continue

                    initializer = declarators.initializer

                    if value is None:
                        return node_id

                    if hasattr(initializer, "value") and str(initializer.value) == str(value):
                        return node_id

                    if hasattr(initializer, "member") and initializer.member == value:
                        return node_id

            if hasattr(statement, "expression"):
                assign_node = statement.expression
                
                if hasattr(assign_node, "expressionl") and hasattr(assign_node, "value"):
                    left = assign_node.expressionl
                    right = assign_node.value

                    if hasattr(left, "member") and left.member == var_name:
                        if value is None:
                            return node_id
                        if hasattr(right, "value") and str(right.value) == str(value):
                            return node_id
                        if hasattr(right, "member") and right.member == value:
                            return node_id

    return None