import pytest
import networkx as nx
from src.data_parsing.parse_to_CFG import parse_java_to_cfg

def test_basic_block_grouping():
    """
    Test: Sequential statements should be grouped into a single 'Basic Block' node.
    """
    code_linear = """
    public class TestClass {
        public void method() {
            int a = 1;
            int b = 2;
            int c = a + b;
        }
    }
    """
    graphe = parse_java_to_cfg(code_linear)
    basic_blocks = [n for n, data in graphe.nodes(data=True) if data.get('type') == 'BasicBlock']
    
    assert len(basic_blocks) == 1, "Sequential statements should be grouped into a single 'Basic Block' node."

def test_if_control_branching():
    """
    Test: An IF statement should create a 'Control' node and split the graph into multiple edges.
    """
    code_if = """
    public class TestClass {
        public void method(int a) {
            if (a > 0) {
                a = 1;
            } else {
                a = -1;
            }
        }
    }
    """
    graphe = parse_java_to_cfg(code_if)
    
    control_nodes = [n for n, data in graphe.nodes(data=True) if data.get('type') == 'Control']
    assert len(control_nodes) == 1, "The IF condition node was not correctly identified."
    
    if_node = control_nodes[0]
    
    out_edges = list(graphe.successors(if_node))
    assert len(out_edges) == 2, f"The IF node should have 2 outgoing paths, but has {len(out_edges)}."

def test_while_loop_cycle():
    """
    Test: A WHILE statement must create a cycle (a back-edge) in the graph.
    """
    code_while = """
    public class TestClass {
        public void method() {
            int i = 0;
            while(i < 10) {
                i++;
            }
        }
    }
    """
    graphe = parse_java_to_cfg(code_while)
    try:
        cycles = list(nx.find_cycle(graphe, orientation='original'))
        has_cycle = True
    except nx.NetworkXNoCycle:
        has_cycle = False
        
    assert has_cycle, "The graph does not contain a cycle. The back-edge from the WHILE loop is not handled!"