import javalang
import networkx as nx

class CFGBuilder:
    def __init__(self):
        self.cfg = nx.DiGraph()
        self.node_counter = 0

    def _create_node(self, label, node_type="BasicBlock", statements=None):
        """Creates new node in the graph and returns its ID."""
        node_id = f"{node_type}_{self.node_counter}"
        self.cfg.add_node(node_id, label=label, type=node_type, ast_nodes=statements or [])
        self.node_counter += 1
        return node_id

    def build_from_method(self, method_node):
        """Constructs the CFG for a specific method."""
        entry_node = self._create_node(f"Entry: {method_node.name}", "MethodEntry")
        
        if method_node.body:
            final_exits = self._process_statements(method_node.body, [entry_node])

            exit_node = self._create_node("MethodExit", "MethodExit")
            for ext in final_exits:
                self.cfg.add_edge(ext, exit_node)

        return self.cfg

    def _process_statements(self, statements, incoming_edges):
        """
            Processes a list of statements. 
            incoming_edges: list of parent nodes pointing to this block.
            Returns: list of terminal nodes of this block (to link to the next block).
        """
        current_exits = incoming_edges
        current_basic_block = []

        def flush_basic_block():
            """Flushes accumulated statements into a new BasicBlock node."""
            nonlocal current_exits, current_basic_block
            if current_basic_block:
                label = "Block:\n" + "\n".join([type(s).__name__ for s in current_basic_block])
                block_node = self._create_node(label, "BasicBlock", statements=current_basic_block)
                
                for ext in current_exits:
                    self.cfg.add_edge(ext, block_node)
                
                current_exits = [block_node]
                current_basic_block = []

        for stmt in statements:
            if isinstance(stmt, (javalang.tree.IfStatement, javalang.tree.WhileStatement)):
                flush_basic_block()
                
                if isinstance(stmt, javalang.tree.IfStatement):
                    cond_node = self._create_node("If Condition", "Control")
                    for ext in current_exits:
                        self.cfg.add_edge(ext, cond_node)
                    
                    true_stmts = stmt.then_statement.statements if isinstance(stmt.then_statement, javalang.tree.BlockStatement) else [stmt.then_statement]
                    true_exits = self._process_statements(true_stmts, [cond_node])
                    
                    false_exits = []
                    if getattr(stmt, 'else_statement', None):
                        false_stmts = stmt.else_statement.statements if isinstance(stmt.else_statement, javalang.tree.BlockStatement) else [stmt.else_statement]
                        false_exits = self._process_statements(false_stmts, [cond_node])
                    else:
                        false_exits = [cond_node]
                    
                    current_exits = true_exits + false_exits

                elif isinstance(stmt, javalang.tree.WhileStatement):
                    cond_node = self._create_node("While Condition", "Control")
                    for ext in current_exits:
                        self.cfg.add_edge(ext, cond_node)

                    loop_stmts = stmt.body.statements if isinstance(stmt.body, javalang.tree.BlockStatement) else [stmt.body]
                    loop_exits = self._process_statements(loop_stmts, [cond_node])

                    for ext in loop_exits:
                        self.cfg.add_edge(ext, cond_node)
                        
                    current_exits = [cond_node]

            elif isinstance(stmt, javalang.tree.ReturnStatement):
                flush_basic_block()
                ret_node = self._create_node("Return", "Exit")
                for ext in current_exits:
                    self.cfg.add_edge(ext, ret_node)

            else:
                current_basic_block.append(stmt)

        flush_basic_block()
        return current_exits


def parse_java_to_cfg(java_source_code):
    """Main function to generate the complete CFG."""
    tree = javalang.parse.parse(java_source_code)
    builder = CFGBuilder()
    
    for path, node in tree.filter(javalang.tree.MethodDeclaration):
        return builder.build_from_method(node)
