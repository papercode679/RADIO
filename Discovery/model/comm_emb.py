from .gnn import GNNEncoder
import torch
import torch.nn as nn
import torch.nn.functional as F

class AbnormalSubgraphOrderEmbedding(nn.Module):
    def __init__(self, args):
        super(AbnormalSubgraphOrderEmbedding, self).__init__()

        self.encoder = GNNEncoder(args)
        self.margin = args.margin
        self.device = args.device



    def align_nested_lists(self,list1, list2):
        max_length = max(len(max(list1, key=len)), len(max(list2, key=len)))
        aligned_list1 = [sublist + [0] * (max_length - len(sublist)) for sublist in list1]
        aligned_list2 = [sublist + [0] * (max_length - len(sublist)) for sublist in list2]
        return aligned_list1, aligned_list2

    def calculate_kl_divergence_loss(self,list1, list2):
        list1_aligned, list2_aligned = self.align_nested_lists(list1, list2)
        kl_div_losses = []
        for sublist1, sublist2 in zip(list1_aligned, list2_aligned):
            tensor1 = torch.tensor(sublist1, dtype=torch.float32)
            tensor2 = torch.tensor(sublist2, dtype=torch.float32)
            kl_div_loss = F.kl_div(F.log_softmax(tensor1, dim=-1),F.softmax(tensor2, dim=-1),reduction='batchmean')
            kl_div_losses.append(kl_div_loss.item())
        return kl_div_losses

    def align_lists(self,list1, list2, fill_value=0):
        max_length = max(len(lst) for lst in list1 + list2)

        for lst in list1:
            while len(lst) < max_length:
                lst.append(fill_value)

        for lst in list2:
            while len(lst) < max_length:
                lst.append(fill_value)

    def forward(self, emb_as, emb_bs):
        return emb_as, emb_bs

    def predict(self, pred):
        emb_as, emb_bs = pred

        e = torch.sum(torch.max(torch.zeros_like(emb_as, device=self.device), emb_bs - emb_as) ** 2, dim=1)
        return e

    def criterion(self, pred, labels, pos_edge_a, pos_edge_b, neg_edge_a, neg_edge_b,pos_benford_a,pos_benford_b,neg_benford_a,neg_benford_b):
        emb_as, emb_bs = pred
        device = self.device  # 获取设备
        e = torch.sum(torch.max(torch.zeros_like(emb_as, device=device), emb_bs - emb_as) ** 2, dim=1)
        margin = self.margin
        e[labels == 0] = torch.max(torch.tensor(0.0, device=device), margin - e)[labels == 0]
        kl_div_loss = F.kl_div(F.log_softmax(emb_as, dim=1), F.softmax(emb_bs, dim=1), reduction='batchmean')


        self.align_lists(pos_edge_a, pos_edge_b)
        posa_tensor = torch.tensor(pos_edge_a, dtype=torch.float).to(device)
        posb_tensor = torch.tensor(pos_edge_b, dtype=torch.float).to(device)
        kl_div_loss_pos_1 = F.kl_div(F.log_softmax(posa_tensor, dim=1), F.softmax(posb_tensor, dim=1), reduction='batchmean')
        self.align_lists(neg_edge_a, neg_edge_b)
        nega_tensor = torch.tensor(neg_edge_a, dtype=torch.float).to(device)
        negb_tensor = torch.tensor(neg_edge_b, dtype=torch.float).to(device)
        kl_div_loss_neg_1 = F.kl_div(F.log_softmax(nega_tensor, dim=1), F.softmax(negb_tensor, dim=1), reduction='batchmean')

        self.align_lists(pos_benford_a,pos_benford_b)
        posa_tensor = torch.tensor(pos_benford_a, dtype=torch.float).to(device)
        posb_tensor = torch.tensor(pos_benford_a, dtype=torch.float).to(device)
        kl_div_loss_pos = F.kl_div(F.log_softmax(posa_tensor, dim=1), F.softmax(posb_tensor, dim=1), reduction='batchmean')
        self.align_lists(neg_benford_a,neg_benford_b)
        nega_tensor = torch.tensor(neg_benford_a, dtype=torch.float).to(device)
        negb_tensor = torch.tensor(neg_benford_b, dtype=torch.float).to(device)
        kl_div_loss_neg = F.kl_div(F.log_softmax(nega_tensor, dim=1), F.softmax(negb_tensor, dim=1), reduction='batchmean')


        return torch.sum(e) + kl_div_loss_pos +kl_div_loss_neg
