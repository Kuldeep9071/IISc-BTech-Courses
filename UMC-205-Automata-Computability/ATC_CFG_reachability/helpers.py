import re

class CFG:
    def __init__(self, productions):
        self.productions = self.parse_productions(productions)

    def parse_productions(self, productions):
        production_dict = {}
        # Split production rules using semicolon (;)
        for production in productions.split(';'):
            head, body = production.split('->')
            head = head.strip()
            # Handle multiple right-hand side cases (separated by '|')
            bodies1 = body.split('|')
            bodies = []
            for b in bodies1:
                bodies.append(b.strip().split())
            if head not in production_dict:
                production_dict[head] = []
            production_dict[head].extend(bodies)  # Add the bodies to the list for the current head
        return production_dict


class Graph:
    def __init__(self):
        self.adjacency_list = {}

    def add_edge(self, source, target, label):
        if source not in self.adjacency_list:
            self.adjacency_list[source] = []
        self.adjacency_list[source].append((target, label))

    def get_edges(self):
        return self.adjacency_list
