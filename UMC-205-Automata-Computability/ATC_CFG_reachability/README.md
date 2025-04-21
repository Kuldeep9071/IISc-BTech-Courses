# ATC_CFG_Reachability

## 🔹Problem Description
**Given**
- Directed Weighted Graph (each edge is labeled is a charactor 'a' to 'z')
- Start Vertex
- End Vertex
- Context Free Grammer in Chomsky Normal Form (CNF)

**Target**

We have to determine  whether  there  is  a  path  from  a  given source node to a target node such that the sequence of edge labels along the path forms  a  string  derivable  from  the  CFG

## 🔹 Input Format

- Number of inputs (n)
- Next 4n lines contains each input
  - CFG in CNF form
  - Graph
  - start_vertex
  - end_vertex


## 🔹 Output Format
- **`Yes`** if there exist a path as described above
- **`No`** otherwise.

## 🔹Example
### **Input**
```
2
S -> AB;A -> a;B -> BC| b;C -> c
PQ:a QR:b RT:c 
P
T
S -> AB;A -> a;B -> BC| b;C -> c
PQ:a QR:c RT:b
P
T
```

### **Output**
```
Yes
No
```
