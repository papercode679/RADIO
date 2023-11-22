import random
import numpy as np
import scipy.stats as stats
import torch
import networkx as nx
import operator,collections,math

from deepsnap.graph import Graph as DSGraph, Graph
from deepsnap.batch import Batch
from torch_geometric.data import Data
import time

def get_device(device=None):
    if device:
        return torch.device(device)
    return torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


def find_weights(edge_list, other_variable):
    weights = []
    for edge in edge_list:
        for item in other_variable:
            if (edge[0] == item[0] and edge[1] == item[1]) or (edge[1] == item[0] and edge[0] == item[1]):
                weights.append(item[2])
                break
    return weights



def get_weights_from_edges(node_list, edges):
    node_set = set(node_list)
    weights = []
    edge_info = []

    for edge in edges:
        source_node, target_node, weight = edge[:3]

        if source_node in node_set and target_node in node_set:
            weights.append(weight)
            edge_info.append([source_node, target_node, weight])

    return weights, edge_info


def get_edges_from_nodes(neigh, edges):
    node_edges = {}
    for edge in edges:
        source_node, target_node, weight = edge[:3]
        node_pair = (source_node, target_node)
        if node_pair not in node_edges:
            node_edges[node_pair] = []
        node_edges[node_pair].append(weight)

    edges_info = []
    for node_pair in neigh:
        if node_pair in node_edges:
            edges_info.append(node_edges[node_pair])

    return edges_info



def has_edges_with_vertex(edge_info, vertex):
    for edge in edge_info:
        if vertex in edge[:2]:
            return True
    return False



def process_edges(edge_info, num, subgraph_vertices):
    flag = num
    i=0

    while flag > 0:
        i += 1
        if i > 700:
            break;

        w = []
        for edge in edge_info:
            w.append(edge[2])

        selected_edge = random.choices(edge_info, weights=w)[0]



        temp_edge_info = edge_info[:]
        updated_subgraph_vertices = subgraph_vertices[:]

        node1, node2, weight = selected_edge

        if weight >= 2:
            selected_edge[2] -= 1
            weight -= 1
            index = temp_edge_info.index(selected_edge)
            temp_edge_info[index][2] = weight
            flag -= 1

        else:
            if (node1 in updated_subgraph_vertices) and (node2 in updated_subgraph_vertices) and node1 != node2:
                temp_edge_info.remove(selected_edge)
                if not has_edges_with_vertex(temp_edge_info,node1):
                    updated_subgraph_vertices.remove(node1)
                if not has_edges_with_vertex(temp_edge_info,node2):
                    updated_subgraph_vertices.remove(node2)

                graph = nx.Graph()
                graph.add_nodes_from(updated_subgraph_vertices)
                graph.add_weighted_edges_from(temp_edge_info)
                if graph.number_of_nodes() > 0:
                    is_connected = nx.is_connected(graph)
                else:
                    is_connected = False


                if not is_connected:
                    temp_edge_info.append(selected_edge)
                    if node1 not in updated_subgraph_vertices:
                        updated_subgraph_vertices.append(node1)
                    if node2 not in updated_subgraph_vertices:
                        updated_subgraph_vertices.append(node2)
                else:
                    flag -= 1

    return temp_edge_info, updated_subgraph_vertices


def sample_neigh(graphs, size, edges, graph_weight):
    """Sampling function during training"""
    ps = np.array([len(g) for g in graphs], dtype=np.float64)
    ps /= np.sum(ps)
    dist = stats.rv_discrete(values=(np.arange(len(graphs)), ps))

    while True:
        idx = dist.rvs()
        graph = graphs[idx]
        edge_info = []
        we = graph_weight
        neigh = []
        for edge in graph.edges(data=True):
            edge_info.append([edge[0], edge[1], edge[2]['weight']])
        total = list(set(graph.nodes))
        for node in total:
            neigh.append(node)

        num = abs(size - graph_weight)
        if num != 0 and num < we:
            updated_edge_info, neigh = process_edges(edge_info, num, neigh)
        else:
            updated_edge_info = edge_info
        return graph, neigh, updated_edge_info



def generate_ego_net(graph, start_node, benford_G,k=1, max_size=15, choice="subgraph"):
    """Generate **k** ego-net"""
    q = [start_node]
    visited = [start_node]

    iteration = 0
    while True:
        if iteration >= k:
            break
        length = len(q)
        if length == 0 or len(visited) >= max_size:
            break

        for i in range(length):
            # Queue pop
            u = q[0]
            q = q[1:]

            for v in list(graph.neighbors(u)):
                if v not in visited:
                    q.append(v)
                    visited.append(v)
                if len(visited) >= max_size:
                    break
            if len(visited) >= max_size:
                break
        iteration += 1
    visited = sorted(visited)
    peel_visited = naivePeeling(visited,graph,start_node)
    if choice =="neighbors":
        return peel_visited
    elif choice =="multi":
        multi = [start_node]
        benford = []
        for i in range(9):
            benford.append(math.log10(1 + 1 / (i + 1)))
        sub = benford_G.subgraph(peel_visited)
        t = nx.get_edge_attributes(sub, 'weight')
        res = []
        for j in t:
            res += t[j]
        xx = collections.Counter(res)
        obs = []
        for i in range(1, 10):
            if i not in xx:
                obs.append(0)
            else:
                obs.append(xx[i])
        times = sum(obs)
        chi = 0
        denominators = np.array(benford) * times
        non_zero_indices = np.nonzero(denominators)

        for i in range(9):
                denominator = benford[i] * times
                if denominator != 0:
                    chi += (obs[i] - denominator) ** 2 / denominator
                else:

                    chi += 0
        multi.append(chi)
        multi = multi + peel_visited
        return multi
    else:
        return graph.subgraph(peel_visited)

def generate_outer_boundary_3(graph, com_nodes, max_size=20):
    outer_nodes = []

    for node in com_nodes:
        outer_nodes += list(graph.neighbors(node))
    outer_nodes = list(set(outer_nodes) - set(com_nodes))
    outer_nodes = sorted(outer_nodes)

    outer_nodes_2 = {}
    for node in com_nodes:
        for neighbor in graph.neighbors(node):
            if neighbor not in com_nodes:
                weight = graph[node][neighbor]['weight']
                outer_nodes_2[neighbor] = weight
    sorted_keys = sorted(outer_nodes_2.keys(), key=lambda x: outer_nodes_2[x], reverse=True)

    if len(outer_nodes)<=max_size:
        new_array = create_new_array(outer_nodes,sorted_keys,len(outer_nodes))
    else:
        new_array = create_new_array(outer_nodes,sorted_keys,max_size)

    return new_array



def create_new_array(sorted_keys, outer_nodes, max_size):
    if len(sorted_keys) >= max_size and len(outer_nodes) >= max_size:
        selected_sorted_keys = sorted_keys[:10]
        selected_outer_nodes = outer_nodes[:10]
    else:
        selected_sorted_keys = sorted_keys[:len(sorted_keys) // 2]
        selected_outer_nodes = outer_nodes[:len(outer_nodes) // 2]

    new_array = selected_sorted_keys + selected_outer_nodes
    unique_elements = list(set(new_array))
    return unique_elements[:max_size]



def batch2graphs(graphs, device=None):
    """Transform `List[nx.Graph]` into `DeepSnap.Batch` object"""
    graph_data = [DSGraph(g) for g in graphs]
    for graph, graph_edge in zip(graph_data, graphs):
        weights = []
        for u, v, data in graph_edge.edges(data=True):
            weight = data.get('weight')
            weights.append(weight)
        graph.weight = weights
    batch = Batch.from_data_list(graph_data)
    batch = batch.to(get_device(device=device))

    return batch



def generate_embedding(batch, model, device=None):
    batches = batch2graphs(batch, device=device)
    pred = model.encoder(batches)
    pred = pred.cpu().detach().numpy()
    return pred


def split_abnormalsubgraphs(subgraphs, n_train, n_val=0):
    print(f"Split abnormalsubgraphs, # Train {n_train}, # Val {n_val}, # Test {len(subgraphs) - n_train - n_val}")
    random.shuffle(subgraphs)
    # train, val, test
    return subgraphs[:n_train], subgraphs[n_train:n_train + n_val], subgraphs[n_train + n_val:]



def minHeapify(pos, heap, heapSize, heapPos):
    leftPos = 2 * pos+1
    rightPos = leftPos + 1
    least = pos

    if leftPos < heapSize and heap[leftPos][0] < heap[pos][0]:
        least = leftPos

    if rightPos < heapSize and heap[rightPos][0] < heap[least][0]:
        least = rightPos

    if least != pos:
        currentID = heap[pos][1]
        leastID = heap[least][1]

        temp = heap[least]
        heap[least] = heap[pos]
        heap[pos] = temp

        tempPos = heapPos[least]
        heapPos[least] = heapPos[pos]
        heapPos[pos] = tempPos

        minHeapify(least, heap, heapSize, heapPos)


def heapDecreaseKey(heap, heapPos, id, value, adj):
    # pos = heapPos[id]
    if id in heapPos:
        x=heapPos.index(id)
        pos=heapPos[x]

        new_value = heap[x][0] - value
        new_pair = (new_value, id)
        heap[x] = new_pair

    while x > 0 and new_value < heap[(x -1) // 2][0]:
        currentID = heap[x][1]
        parentID = heap[(x -1) // 2][1]

        temp = heap[(x - 1) // 2]
        heap[(x-1) // 2] = heap[x]
        heap[x] = temp

        if parentID in heapPos:
            par_x = heapPos.index(parentID)
        if currentID in heapPos:
            cur_x = heapPos.index(currentID)

        tempPos = heapPos[x]
        heapPos[x] = heapPos[(x-1) // 2]
        heapPos[(x-1) // 2] = tempPos

        x = (x - 1) // 2


def heapExtractMin(heap, heapSize, heapPos, adj, indicateGraph,origin):
    minStrength = heap[0][0]
    minID = heap[0][1]
    new = (10000, -1)
    heap[0] = new
    heapSize -= 1
    # heapPos[minID] = -1
    if minID in heapPos:
        x=heapPos.index(minID)
        heapPos[x] = -1

    indicateGraph[minID] = 0
    for i in range(origin // 2 - 1, -1, -1):
        minHeapify(i, heap, origin, heapPos)

    adjacent = adj[minID]
    for adjPair in adjacent:
        if indicateGraph[adjPair[0]] > 0:
            heapDecreaseKey(heap, heapPos, adjPair[0], adjPair[1],adj)
    return minStrength,heapSize,indicateGraph


def naivePeeling(visited, graph, subgraphs_node):
    start = time.time()
    visited_graph = graph.subgraph(visited)
    V = len(visited_graph.nodes())
    adj = [[] for _ in range(V)]
    strength = [0.0] * V
    totalW = 0.0
    node_dict = {}

    for n1, n2, w in visited_graph.edges(data=True):
        n1 = int(n1)
        n2 = int(n2)
        weight = w.get('weight')
        w = int(weight)

        if n1 not in node_dict:
            node_dict[n1] = len(node_dict)

        if n2 not in node_dict:
            node_dict[n2] = len(node_dict)

        n1_idx = node_dict[n1]
        n2_idx = node_dict[n2]

        adj[n1_idx].append((n2_idx, w))
        adj[n2_idx].append((n1_idx, w))
        strength[n1_idx] += w
        strength[n2_idx] += w
        totalW += w

    indicateGraph = [1] * V

    heapSize = V
    heapPos = [0] * V
    dmax = 0.0
    densityMax = totalW / V
    origin = heapSize

    heap = [(strength[i], i) for i in range(V) if indicateGraph[i] > 0]

    cp = 0
    for i in range(0, V):
        if indicateGraph[i] > 0:
            heapPos[i] = cp
            cp += 1

    for i in range(heapSize // 2 - 1, -1, -1):
        minHeapify(i, heap, heapSize, heapPos)

    max_gr = []

    while heapSize > 2:
        tempS ,heapSize,indicateGraph = heapExtractMin(heap, heapSize, heapPos, adj, indicateGraph,origin)

        totalW -= tempS
        if totalW<0:
            break

        if tempS > dmax:
            dmax = tempS

        gr=[]

        for ig, nd in zip(indicateGraph, node_dict):
            if ig == 1:
                gr.append(nd)
        if densityMax < totalW / heapSize and subgraphs_node in gr:
            densityMax = totalW / heapSize
            max_gr = gr
    if max_gr == []:
        max_gr = visited


    duration = time.time() - start
    return max_gr