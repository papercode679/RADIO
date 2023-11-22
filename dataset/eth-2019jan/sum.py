# check statistical significance
from scipy import stats
import networkx as nx
import random, collections

import math
from peel_by_motif import *
import numpy as np
def kl_divergence(p, q):
    return sum(p[i] * math.log2(p[i]/q[i]) for i in range(len(p)) if p[i]>0)
def node_chisquare(edgelist, node_num, dist, adjust_idx=1, directed='both', diff_func='chi'):
    '''
    edgelist: list of edges. Each item is a tuple contains (node_from, node_to, edge_weight) representing a weighted edge.
    node_num: number of nodes.
    dist: list of float sum to 1, describe the distribution used to calculate the chi square statistic.
    '''
    node_induced_dist = [[] for i in range(node_num)]
    node_chis = []
    for edge in edgelist:
        if directed=='both' or directed=='out':
            node_induced_dist[edge[0]].append(edge[2])
        if directed=='both' or directed=='in':
            node_induced_dist[edge[1]].append(edge[2])
#             G[node][neighbor]['weight']
    for node_dist in node_induced_dist:
        count_dist = count_occ(node_dist, len(dist), adjust_idx)
#        get the chi square statistic, the higher it is, more abnormal the node is
        if diff_func=='chi':
            node_chis.append(chisquare(count_dist, sum(count_dist)*np.array(dist))[0])
        elif diff_func=='kl':
            node_chis.append(kl_divergence(np.array(count_dist)/sum(count_dist), np.array(dist)))
    return node_chis

def get_start_digit(v):
    if v==0:
        return 0
    if v<0:
        v = -v
    while v<1:
        v = v*10
    return int(str(v)[:1])


dsd_output_prefix = 'eth2018/output_file_prefix'

G = nx.DiGraph()
f = open('eth-2019jan.csv', 'r')
edgelist = []
node_map = {}
line = f.readline()
line = f.readline()
e_count = 0
n_idx = 0
while line:
    tmp = line.split(',')
    line = f.readline()
    date = tmp[-2].strip()
    money = tmp[-1].strip()
    if len(money) <= 18:
        continue
    if tmp[1] == tmp[2]:
        continue
    if tmp[1] not in node_map:
        node_map[tmp[1]] = n_idx
        n_idx += 1
    if tmp[2] not in node_map:
        node_map[tmp[2]] = n_idx
        n_idx += 1
    money = int(money[:-18])
    edgelist.append((node_map[tmp[1]], node_map[tmp[2]], get_start_digit(money), money, date))

    if node_map[tmp[1]] in G and node_map[tmp[2]] in G[node_map[tmp[1]]]:
        if 'weight' not in G[node_map[tmp[1]]][node_map[tmp[2]]]:
            G[node_map[tmp[1]]][node_map[tmp[2]]]['weight'] = []
        G[node_map[tmp[1]]][node_map[tmp[2]]]['weight'].append(money)
    else:
        G.add_edge(node_map[tmp[1]], node_map[tmp[2]], weight=[money])
    e_count += 1

#     break
print('amount of transactions with >1 volume', e_count, 'by', date)
G = G.to_undirected()
G.remove_edges_from(nx.selfloop_edges(G))
G.number_of_nodes(), G.number_of_edges()

import time
start = time.time()
x = [e[2] for e in edgelist]
x = collections.Counter(x)
num_occ = np.array([x[k] for k in range(1,10)])/sum([x[k] for k in range(1,10)])

node_chisquares = node_chisquare(edgelist, G.number_of_nodes(), num_occ, adjust_idx=1, diff_func='kl')
print(time.time() - start)



f_out = open('eth-2019jan-1.90.ungraph.txt','w')
s = 0
f_out.write(str(G.number_of_nodes())+' '+str(G.number_of_edges())+'\n')
for edge in G.edges():
    f_out.write(str(edge[0])+' '+str(edge[1])+' '+str(int(100*math.sqrt(node_chisquares[edge[0]]*node_chisquares[edge[1]])))+'\n')
f_out.close()

print(G.number_of_nodes(), G.number_of_edges())
