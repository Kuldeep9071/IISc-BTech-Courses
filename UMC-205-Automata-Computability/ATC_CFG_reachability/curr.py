from helpers import CFG
from helpers import Graph

# DFS to enumerate all paths from start_vertex to end_vertex
def DFS(graph, start_vertex, end_vertex, visited, string, paths):
    if start_vertex == end_vertex:
        paths.append(string)
        return
    
    visited[start_vertex] = True  # Mark as visited

    for vertex in graph.adjacency_list.get(start_vertex, []):
        if not visited.get(vertex[0], False):
            DFS(graph, vertex[0], end_vertex, visited, string + vertex[1], paths)

    visited[start_vertex] = False  # Backtrack


# Cocke–Younger–Kasami Algorithm

def CKY(cfg, path):
    n = len(path)
    table = [[set() for _ in range(n)] for _ in range(n)]
    for i in range(n):
        end = path[i]
        for key in cfg.productions.keys():
            if [end] in cfg.productions[key]:
                table[i][i].add(key)

    # print("Initialised table: \n")
    # for row in table:
    #     print(row)

    
    for l in range(1,n):
        for r in range(0,n-l):
            i = l + r 
            j = r
            for k in range(i):
                for left in table[k][j]:
                    for right in table[i][k+1]:
                        for key in cfg.productions.keys():
                            if [left+right] in cfg.productions[key]:
                                table[i][j].add(key)

    # print("Final table: \n")
    # for row in table:
    #     print(row)

    return 'S' in table[n-1][0]
                    

        
    
def check_reachability(cfg, graph, start_vertex, end_vertex):
    # Epsilon Case
    if start_vertex == end_vertex:
        return False
    paths = []
    visited = {}
    DFS(graph, start_vertex, end_vertex, visited, '', paths)
    
    for path in paths:
        if CKY(cfg, path):
            return True
    return False

def read_input(file_path):
    with open(file_path, 'r') as file:
        num_inputs = int(file.readline().strip())
        inputs = []
        for _ in range(num_inputs):
            cfg_productions = file.readline().strip()
            graph_data = file.readline().strip()
            start_vertex = file.readline().strip()
            end_vertex = file.readline().strip()
            inputs.append((cfg_productions, graph_data, start_vertex, end_vertex))
        return inputs

def write_output(file_path, results):
    with open(file_path, 'w') as file:
        for result in results:
            file.write(result + '\n')

def main(input_file, output_file):
    inputs = read_input(input_file)
    results = []
    for cfg_productions, graph_data, start_vertex, end_vertex in inputs:
        cfg = CFG(cfg_productions)
        graph = Graph()
        edge_data = graph_data.split(' ')
        for edge in edge_data:
            src = edge[0]
            dst = edge[1]
            label = edge[3]
            graph.add_edge(src, dst, label)
        reachable = check_reachability(cfg, graph, start_vertex, end_vertex)
        results.append('Yes' if reachable else 'No')
    
    write_output(output_file, results)

if __name__ == "__main__":
    input_file = 'test.txt'
    output_file = 'res.txt'
    main(input_file, output_file)
