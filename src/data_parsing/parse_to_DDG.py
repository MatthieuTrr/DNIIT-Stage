import javalang
import networkx as nx

class DDGBuilder:
    def __init__(self, cfg):
        self.cfg = cfg
        self.ddg = nx.DiGraph()
        for node, data in cfg.nodes(data=True):
            self.ddg.add_node(node, **data)

    def _extract_def_use_from_ast(self, ast_nodes):
        """
            Analyze AST nodes and extract:
            - Defined variables (left side of assignments, declarations)
            - Used variables (right side of assignments, conditions, method arguments)
        """
        defined = set()
        used = set()

        for statement in ast_nodes:
            if not isinstance(statement, javalang.tree.Node):
                continue
                
            for path, node in statement:
                if isinstance(node, javalang.tree.VariableDeclarator):
                    defined.add(node.name)
                    if node.initializer:
                        for _, child in node.initializer:
                            if isinstance(child, javalang.tree.MemberReference):
                                used.add(child.member)

                elif isinstance(node, javalang.tree.Assignment):
                    if isinstance(node.expressionl, javalang.tree.MemberReference):
                        defined.add(node.expressionl.member)
                    elif hasattr(node.expressionl, 'name'):
                        defined.add(node.expressionl.name)

                    for _, child in node.value:
                        if isinstance(child, javalang.tree.MemberReference):
                            used.add(child.member)

                elif isinstance(node, javalang.tree.MemberReference):
                    used.add(node.member)

        return defined, used

    def build(self):
        """
        Builds the DDG edges based on strict AST data flow analysis.
        """
        node_vars = {}
        
        for node in self.cfg.nodes():
            ast_nodes = self.cfg.nodes[node].get('ast_nodes', [])
            node_vars[node] = self._extract_def_use_from_ast(ast_nodes)

        for node_a in self.cfg.nodes():
            defs_a, _ = node_vars[node_a]
            
            for var in defs_a:
                for node_b in self.cfg.nodes():
                    if node_a == node_b:
                        continue
                        
                    _, uses_b = node_vars[node_b]
                    if var in uses_b:
                        if nx.has_path(self.cfg, node_a, node_b):
                            self.ddg.add_edge(node_a, node_b, variable=var, type="DataDependency")
        
        return self.ddg

def parse_cfg_to_ddg(cfg):
    """Entry point to convert a CFG into a DDG."""
    builder = DDGBuilder(cfg)
    return builder.build()