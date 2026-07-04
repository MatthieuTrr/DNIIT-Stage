import javalang
import networkx as nx

class DFGBuilder:
    def __init__(self, cfg):
        self.cfg = cfg
        self.dfg = nx.DiGraph()
        
        for node, data in cfg.nodes(data=True):
            self.dfg.add_node(node, **data)

    def _extract_variables_from_ast(self, ast_nodes):
        """
            Analyze AST nodes and extract all variables used or defined.
            In DFG, we care about the flow, so we collect both definitions and usages.
        """
        variables = set()

        for statement in ast_nodes:
            if not isinstance(statement, javalang.tree.Node):
                continue
                
            for path, node in statement:
                # Retrieve variables declarations
                if isinstance(node, javalang.tree.VariableDeclarator):
                    variables.add(node.name)
                # Retrieve variables uses
                elif isinstance(node, javalang.tree.MemberReference):
                    variables.add(node.member)

        return variables

    def build(self):
        """
        Builds the DFG edges based on data flow across CFG nodes.
        Links node A to node B if they share a variable and there is a path.
        """
        node_vars = {}
        
        # Extract variables for each node in the CFG
        for node in self.cfg.nodes():
            ast_nodes = self.cfg.nodes[node].get('ast_nodes', [])
            node_vars[node] = self._extract_variables_from_ast(ast_nodes)

        # Creation data flow 
        for node_a in self.cfg.nodes():
            vars_a = node_vars[node_a]
            
            for var in vars_a:
                for node_b in self.cfg.nodes():
                    if node_a == node_b:
                        continue
                        
                    vars_b = node_vars[node_b]
                    # if same variable is used in both nodes and there is a logical execution path from A to B
                    if var in vars_b:
                        if nx.has_path(self.cfg, node_a, node_b):
                            self.dfg.add_edge(node_a, node_b, variable=var, type="DataFlow")
        
        return self.dfg

def parse_cfg_to_dfg(cfg):
    """Entry point to convert a CFG into a DFG."""
    builder = DFGBuilder(cfg)
    return builder.build()