import numpy as np
from tqdm import tqdm
import networkx as nx
import scipy.stats as stats
from .model import AbnormalSubgraphOrderEmbedding
import random
from utils import sample_neigh, batch2graphs, generate_embedding, generate_ego_net
import torch
import torch.optim as optim
from tensorboardX import SummaryWriter
from scipy.special import kl_div
import math
import collections
import time
import multiprocessing

class AbnormalMatching:
    def __init__(self, args, graph, train_abnormalsubgraphs, val_abnormalsubgraphs, edges, edgelist, benford_G, node_chisquares):
        self.args = args

        self.graph = graph
        self.edges = edges
        self.seen_nodes = {node for com in train_abnormalsubgraphs + val_abnormalsubgraphs for node in com}
        self.train_abnormalsubgraphs, self.val_abnormalsubgraphs = self.init_subgraphs(train_abnormalsubgraphs), self.init_subgraphs(val_abnormalsubgraphs)
        self.edgelist= edgelist
        self.benford_G = benford_G
        self.node_chisquares = node_chisquares
        self.model = self.load_model()
        self.opt = optim.Adam(filter(lambda p: p.requires_grad, self.model.parameters()), lr=args.lr)
        self.writer = SummaryWriter(args.writer_dir)

    def load_model(self, load=False):
        """Load AbnormalSubgraphOrderEmbedding"""
        model = AbnormalSubgraphOrderEmbedding(self.args)
        model.to(self.args.device)
        if self.args.subm_path and load:
            model.load_state_dict(torch.load(self.args.subm_path, map_location=self.args.device))
        return model

    def init_subgraphs(self, subs):
        if len(subs) > 0:
            return [self.graph.subgraph(com) for com in subs if len(list(self.graph.subgraph(com).edges())) > 0]
        return []

    def find_weights(self, edge_list, other_variable):
        edge_index = {}

        # 构建边索引
        for item in other_variable:
            edge_index[(item[0], item[1])] = item[2]
            edge_index[(item[1], item[0])] = item[2]

        weights = []
        edge_info = []
        for edge in edge_list:
            if edge in edge_index:
                weight = edge_index[edge]
                weights.append(weight)
                edge_info.append((edge[0], edge[1], weight))

        return weights, edge_info

    def generate_batch(self, batch_size, benford_G,valid=False, min_size=2, max_size=100):
        graphs = self.train_abnormalsubgraphs if not valid or len(self.val_abnormalsubgraphs) == 0 else self.val_abnormalsubgraphs

        pos_a, pos_b = [], []

        ratio = self.args.fine_ratio

        pos_edge_a, pos_edge_b = [], []

        pos_benford_a,pos_benford_b = [], []

        # Generate positive pairs
        for i in range(batch_size // 2):
            prob = random.random()
            if prob <= ratio:
                # Fine-grained sampling
                size = random.randint(min_size + 1, max_size)
                graph, a, _ = sample_neigh(graphs, size, self.edges, graph_weight)
                if len(a) - 1 <= min_size:
                    b = a
                else:
                    b = a[:random.randint(max(len(a) - 2, min_size), len(a))]
            else:
                graph = None
                while graph is None or len(graph) < min_size:
                    graph = random.choice(graphs)
                a = graph.nodes

                graph_weight = 0
                edge_info = []
                chi_info = []

                for edge in graph.edges(data=True):
                    weight = edge[2]['weight']
                    graph_weight += weight
                    chi = int(100*math.sqrt(self.node_chisquares[edge[0]]*self.node_chisquares[edge[1]]))
                    edge_info.append([edge[0], edge[1], edge[2]['weight']])
                    chi_info.append([edge[0],edge[1],chi])


                if graph_weight<=2:
                    continue
                _, b, updated_edge_info = sample_neigh([graph],
                                                       random.randint(max(graph_weight - 2, min_size), graph_weight),
                                                       self.edges, graph_weight)
                # print(f"[Pos pair] Choose graph {a}, subgraph {b}")
            neigh_a, neigh_b = graph.subgraph(a), graph.subgraph(b)


            neighaedge = []
            neigh_a_new = nx.Graph(neigh_a)
            neigh_a_new.add_weighted_edges_from(edge_info, weight='weight')
            for u,v,data in neigh_a_new.edges(data = True):
                neighaedge.append(data['weight'])

            neighbedge = []
            neigh_b_new = nx.Graph(neigh_b)
            neigh_b_new.add_weighted_edges_from(updated_edge_info, weight='weight')
            for u,v,data in neigh_b_new.edges(data = True):
                neighbedge.append(data['weight'])

            neighaedgebenford = []
            agraph = benford_G.subgraph(a)
            for u,v,data in agraph.edges(data = True):
                for num in data['weight']:
                    neighaedgebenford.append(num)

            neighbedgebenford = []
            bgraph = benford_G.subgraph(b)
            for u, v, data in bgraph.edges(data=True):
                for num in data['weight']:
                    neighbedgebenford.append(num)

            if len(neigh_a_new.edges()) > 0 and len(neigh_b_new.edges()) > 0:
                pos_a.append(neigh_a_new)
                pos_b.append(neigh_b_new)
                pos_edge_a.append(neighaedge)
                pos_edge_b.append(neighbedge)
                pos_benford_a.append(neighaedgebenford)
                pos_benford_b.append(neighbedgebenford)

        # Generate negative pairs
        neg_a, neg_b = [], []
        neg_edge_a, neg_edge_b = [], []
        neg_benford_a,neg_benford_b = [], []
        for i in range(batch_size // 2):
            prob = random.random()
            if prob <= ratio:
                size = random.randint(min_size + 1, max_size)
                graph_a, a, _ = sample_neigh(graphs, random.randint(min_size, size), self.edges, graph_weight)
                graph_b, b, _ = sample_neigh(graphs, size, self.edges, graph_weight)
            else:
                graph_b = None
                while graph_b is None or len(graph_b) < min_size:
                    graph_b = random.choice(graphs)
                b = graph_b.nodes

                graph_weight = 0
                edge_info_neg = []
                for edge in graph_b.edges(data=True):
                    weight = edge[2]['weight']
                    graph_weight += weight
                    edge_info_neg.append([edge[0], edge[1], edge[2]['weight']])

                if graph_weight<=2:
                    continue

                graph_a, a, updated_edge_info_neg = sample_neigh(graphs, random.randint(min_size, graph_weight),
                                                                 self.edges, graph_weight)
                # print(f"[Neg pair] Choose graph a{a}, graph b{b}")
            neigh_a, neigh_b = graph_a.subgraph(a), graph_b.subgraph(b)

            neighaedge = []
            neigh_a_new = nx.Graph(neigh_a)
            neigh_a_new.add_weighted_edges_from(updated_edge_info_neg, weight='weight')
            for u,v,data in neigh_a_new.edges(data = True):
                neighaedge.append(data['weight'])

            neighbedge = []
            neigh_b_new = nx.Graph(neigh_b)
            neigh_b_new.add_weighted_edges_from(edge_info_neg, weight='weight')
            for u,v,data in neigh_b_new.edges(data = True):
                neighbedge.append(data['weight'])

            neighaedgebenford = []
            agraph = benford_G.subgraph(a)
            for u,v,data in agraph.edges(data = True):
                for num in data['weight']:
                    neighaedgebenford.append(num)

            neighbedgebenford = []
            bgraph = benford_G.subgraph(b)
            for u, v, data in bgraph.edges(data=True):
                for num in data['weight']:
                    neighbedgebenford.append(num)

            if len(neigh_a_new.edges()) > 0 and len(neigh_b_new.edges()) > 0:
                neg_a.append(neigh_a_new)
                neg_b.append(neigh_b_new)
                neg_edge_a.append(neighaedge)
                neg_edge_b.append(neighbedge)
                neg_benford_a.append(neighaedgebenford)
                neg_benford_b.append(neighbedgebenford)

        pos_a = batch2graphs(pos_a, device=self.args.device)
        pos_b = batch2graphs(pos_b, device=self.args.device)
        neg_a = batch2graphs(neg_a, device=self.args.device)
        neg_b = batch2graphs(neg_b, device=self.args.device)
        return pos_a, pos_b, neg_a, neg_b,pos_edge_a,pos_edge_b,neg_edge_a,neg_edge_b,pos_benford_a,pos_benford_b,neg_benford_a,neg_benford_b

    def train_epoch(self, epochs):
        self.model.share_memory()

        batch_size = self.args.batch_size
        pairs_size = self.args.pairs_size
        device = self.args.device

        valid_set = []
        for _ in range(batch_size):
            pos_a, pos_b, neg_a, neg_b, pos_edge_a, pos_edge_b, neg_edge_a, neg_edge_b,pos_benford_a,pos_benford_b,neg_benford_a,neg_benford_b = self.generate_batch(pairs_size, self.benford_G,valid=True)
            valid_set.append((pos_a, pos_b, neg_a, neg_b, pos_edge_a, pos_edge_b, neg_edge_a, neg_edge_b,pos_benford_a,pos_benford_b,neg_benford_a,neg_benford_b))

        for epoch in range(epochs):
            for batch in range(batch_size):
                self.model.train()

                pos_a, pos_b, neg_a, neg_b, pos_edge_a, pos_edge_b, neg_edge_a, neg_edge_b,pos_benford_a,pos_benford_b,neg_benford_a,neg_benford_b = self.generate_batch(pairs_size,self.benford_G)
                emb_pos_a, emb_pos_b = self.model.encoder(pos_a), self.model.encoder(pos_b)
                emb_neg_a, emb_neg_b = self.model.encoder(neg_a), self.model.encoder(neg_b)

                emb_as = torch.cat((emb_pos_a, emb_neg_a), dim=0)
                emb_bs = torch.cat((emb_pos_b, emb_neg_b), dim=0)

                labels = torch.tensor([1] * pos_a.num_graphs + [0] * neg_a.num_graphs).to(device)
                pred = self.model(emb_as, emb_bs)

                self.model.zero_grad()
                loss = self.model.criterion(pred, labels,pos_edge_a, pos_edge_b, neg_edge_a, neg_edge_b,pos_benford_a,pos_benford_b,neg_benford_a,neg_benford_b)
                loss = torch.sum(loss)
                loss.backward()
                self.opt.step()

                if (batch + 1) % 5 == 0:
                    self.writer.add_scalar(f"subgraphM Loss/Train", loss.item(), batch + epoch * batch_size)
                    print(f"Epoch {epoch + 1}, Batch{batch + 1}, Loss {loss.item():.4f}")
                if (batch + 1) % 10 == 0:
                    self.valid_model(valid_set, batch + epoch * batch_size)
        torch.save(self.model.state_dict(), self.args.writer_dir + "/subgraphrm.pt")

    def valid_model(self, valid_set, batch_num):
        """Test model on `valid_set`"""
        self.model.eval()
        device = self.args.device

        total_loss = 0
        for pos_a, pos_b, neg_a, neg_b, pos_edge_a, pos_edge_b, neg_edge_a, neg_edge_b,pos_benford_a,pos_benford_b,neg_benford_a,neg_benford_b in valid_set:
            labels = torch.tensor([1] * pos_a.num_graphs + [0] * neg_a.num_graphs).to(device)

            with torch.no_grad():
                emb_pos_a, emb_pos_b = self.model.encoder(pos_a), self.model.encoder(pos_b)
                emb_neg_a, emb_neg_b = self.model.encoder(neg_a), self.model.encoder(neg_b)

                emb_as = torch.cat((emb_pos_a, emb_neg_a), dim=0)
                emb_bs = torch.cat((emb_pos_b, emb_neg_b), dim=0)

                pred = self.model(emb_as, emb_bs)
                loss = self.model.criterion(pred, labels,pos_edge_a, pos_edge_b, neg_edge_a, neg_edge_b,pos_benford_a,pos_benford_b,neg_benford_a,neg_benford_b)
                total_loss += loss.item()
        total_loss /= len(valid_set)
        self.writer.add_scalar(f"subgraphM Loss/Val", loss.item(), batch_num)
        print("[Eval-Test] Validation Loss{:.4f}".format(total_loss))

        # TODO: Save model
        # torch.save(self.model.state_dict(), self.args.writer_dir + "/subgraphm.pt")

    def get_start_digit(self,v):
        if v == 0:
            return 0
        if v < 0:
            v = -v
        while v < 1:
            v = v * 10
        return int(str(v)[:1])

    def kl_divergence(self,p, q):
        return sum(p[i] * math.log2(p[i] / q[i]) for i in range(len(p)) if p[i] > 0)

    def count_occ(self,l, dist_len, adjust_idx=0):
        result = [0 for i in range(dist_len)]
        for e in l:
            result[e - adjust_idx] += 1
        return result

    def node_chisquare(self,edgelistl, node_num, dist, adjust_idx=1, directed='both', diff_func='chi'):
        '''
        edgelist: list of edges. Each item is a tuple contains (node_from, node_to, edge_weight) representing a weighted edge.
        node_num: number of nodes.
        dist: list of float sum to 1, describe the distribution used to calculate the chi square statistic.
        '''
        node_induced_dist = [[] for i in range(node_num)]
        node_chis = []
        for edge in edgelistl:
            if directed == 'both' or directed == 'out':
                node_induced_dist[edge[0]].append(edge[2])
            if directed == 'both' or directed == 'in':
                node_induced_dist[edge[1]].append(edge[2])
        #             G[node][neighbor]['weight']
        for node_dist in node_induced_dist:
            count_dist = self.count_occ(node_dist, len(dist), adjust_idx)
            #        get the chi square statistic, the higher it is, more abnormal the node is
            if diff_func == 'chi':
                node_chis.append(stats.chisquare(count_dist, sum(count_dist) * np.array(dist))[0])
            elif diff_func == 'kl':
                node_chis.append(self.kl_divergence(np.array(count_dist) / sum(count_dist), np.array(dist)))
        return node_chis

    def generate_ego_net_wrapper(self,graph, start, end, result, benford_G,lock):
        graphs = [generate_ego_net(graph, g, benford_G,k=self.args.kego, max_size=self.args.ab_size, choice="multi") for g in
                  range(start, end)]

        with lock:
            result.extend(graphs)


    def load_embedding(self):
        query_emb = generate_embedding(self.train_abnormalsubgraphs + self.val_abnormalsubgraphs, self.model, device=self.args.device)
        k = 50
        n_node = len(list(self.graph.nodes()))
        batch_size = 10000
        batch_len = int((n_node / batch_size) + 1)
        benford = []
        for i in range(9):
            benford.append(math.log10(1 + 1 / (i + 1)))

        all_emb = np.zeros((n_node, self.args.output_dim))
        for batch_num in range(batch_len):
            start_time_ego = time.time()
            start, end = batch_num * batch_size, min((batch_num + 1) * batch_size, n_node)


            num_iterations = end-start
            num_processes = 35
            processes = []
            file_lock = multiprocessing.Lock()
            with multiprocessing.Manager() as manager:
                results = manager.list()

                for i in range(num_processes):
                    startmulti = i * (num_iterations) // num_processes+start
                    endmulti = (i + 1) * (num_iterations) // num_processes+start
                    process = multiprocessing.Process(target=self.generate_ego_net_wrapper,
                                                      args=(self.graph, startmulti, endmulti, results, self.benford_G,file_lock))
                    processes.append(process)
                    process.start()

                for process in processes:
                    process.join()

                sorted_list = sorted(results, key=lambda x: x[0])
                sorted_list_without_first_element = [item[1:] for item in sorted_list]
            graphs = sorted_list_without_first_element
            first_numbers = [item.pop(0) for item in graphs]
            end_time_ego = time.time()
            execution_time_ego = end_time_ego - start_time_ego
            print("k-ego net and chisquares: ", execution_time_ego, "秒")
            node_dict = {}
            start_time = time.time()
            subchi = first_numbers
            a=0
            for chi in subchi:
                node_dict[start + a] = chi
                a += 1

            graph_dict = {}

            for sublist, number in zip(graphs, subchi):
                graph_dict[tuple(sublist)] = number

            node_sorted_keys = sorted(node_dict, key=node_dict.get, reverse=True)
            top_1000_node = node_sorted_keys[:k]
            sorted_dict = sorted(graph_dict,key=graph_dict.get , reverse=True)

            top_1000_graph = [list(key) for key in sorted_dict[:k]]
            graphemb = []
            for graph_id in top_1000_graph:
                graphemb.append(self.graph.subgraph(graph_id))
            tmb_emb = generate_embedding(graphemb,self.model, device=self.args.device)
            for i in range(k):
                all_emb[top_1000_node[i],:] = tmb_emb[i]

            print(
                "No.{}-{} candidate AbnormalSubgraphs embedding finish".format(start, end))
            end_time = time.time()
            execution_time = end_time - start_time
            print("embedding time: ", execution_time, "秒")
        np.save(self.args.writer_dir + "/emb", all_emb)
        np.save(self.args.writer_dir + "/query", query_emb)
        return all_emb, query_emb


    def make_prediction(self,graph, seen_nodes):
        all_emb, query_emb = self.load_embedding()
        print(f"[Load Embedding], All shape {all_emb.shape}, Query shape {query_emb.shape}")

        pred_abnormalsubgraphs = []

        pred_size = 50
        single_pred_size = int(pred_size / query_emb.shape[0])
        print("single_pred_sized")
        print(single_pred_size)

        seeds = []

        for i in tqdm(range(query_emb.shape[0]), desc="Matching AbnormalSubgraphs"):
            q_emb = query_emb[i, :]

            distance = np.sqrt(np.sum(np.asarray(q_emb - all_emb) ** 2, axis=1))
            sort_dic = list(np.argsort(distance))

            if len(pred_abnormalsubgraphs) >= pred_size:
                break

            length = 0
            for node in sort_dic:
                if length >= single_pred_size:
                    break
                neighs = generate_ego_net(graph, node, self.benford_G,k=self.args.kego, max_size=200,
                                          choice="neighbors")

                if neighs not in pred_abnormalsubgraphs and len(pred_abnormalsubgraphs) < pred_size and node not in seen_nodes and node \
                        not in seeds:
                    seeds.append(node)
                    pred_abnormalsubgraphs.append(neighs)
                    length += 1
        lengths = np.array([len(pred_sub) for pred_sub in pred_abnormalsubgraphs])
        print(f"[Generate] Pred size {len(pred_abnormalsubgraphs)}, Avg Length {np.mean(lengths):.04f}")
        return pred_abnormalsubgraphs, all_emb
