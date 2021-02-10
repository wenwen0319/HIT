import logging
import time
import numpy as np
import torch
import multiprocessing as mp
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from transformer import TransformerEncoderLayer, TransformerDecoderLayer
from utils import *
from position import *
from numba import jit
PRECISION = 5
POS_DIM_ALTER = 100


class MergeLayer(torch.nn.Module):
    def __init__(self, dim1, dim2, dim3, dim4, non_linear=True):
        super().__init__()
        #self.layer_norm = torch.nn.LayerNorm(dim1 + dim2)
        self.fc1 = torch.nn.Linear(dim1 + dim2, dim3)
        self.fc2 = torch.nn.Linear(dim3, dim4)
        self.act = torch.nn.ReLU()

        torch.nn.init.xavier_normal_(self.fc1.weight)
        torch.nn.init.xavier_normal_(self.fc2.weight)

        # special linear layer for motif explainability
        self.non_linear = non_linear
        if not non_linear:
            assert(dim1 == dim2)
            self.fc = nn.Linear(dim1, 1)
            torch.nn.init.xavier_normal_(self.fc1.weight)

    def forward(self, x1, x2):
        z_walk = None
        if self.non_linear:
            x = torch.cat([x1, x2], dim=-1)
            #x = self.layer_norm(x)
            h = self.act(self.fc1(x))
            z = self.fc2(h)
        else: # for explainability
            # x1, x2 shape: [B, M, F]
            x = torch.cat([x1, x2], dim=-2)  # x shape: [B, 2M, F]
            z_walk = self.fc(x).squeeze(-1)  # z_walk shape: [B, 2M]
            z = z_walk.sum(dim=-1, keepdim=True)  # z shape [B, 1]
        return z, z_walk

class finalClassifier_inter(torch.nn.Module):
    def __init__(self, dim1, dim2, dim3, dim4, dim5):
        super().__init__()
        # self.layer_norm = torch.nn.LayerNorm(dim1 + dim2 + dim3)
        self.fc1 = torch.nn.Linear(dim1, dim4)
        self.fc2 = torch.nn.Linear(dim4, dim5)
        self.act = torch.nn.ReLU()


        torch.nn.init.xavier_normal_(self.fc1.weight)
        torch.nn.init.xavier_normal_(self.fc2.weight)

        self.MLP_two_nodes = torch.nn.Linear(dim1, dim4)
        self.MLP_last = torch.nn.Linear(dim4, dim4)

        # self.graph_dim = dim5
        # self.graph = nn.Sequential(nn.Linear(dim5 * dim5, dim5 * dim5), nn.ReLU(), nn.Linear(dim5* dim5, dim5))

    def forward(self, x1, x2, x3):
        # # MLP(u;v;w)
        # x = torch.cat([x1, x2, x3], dim=-1)
        # x = self.layer_norm(x)
        # h = self.act(self.fc1(x))
        # return self.fc2(h)

        # MLP(MLP(v;w) + MLP(u;w))
        if x2 is None:
            h = self.fc2(self.act(self.MLP_last(x1)))
        else:
            h = self.fc2(self.act(self.MLP_last(x1+x2+x3)))

        # p = (h.view(-1, self.graph_dim, 1) * h.view(-1, 1, self.graph_dim)).view(-1, self.graph_dim * self.graph_dim)
        # final_graph = self.graph(p)
        return h

class finalClassifier(torch.nn.Module):
    def __init__(self, dim1, dim2, dim3, dim4, dim5):
        super().__init__()
        self.layer_norm = torch.nn.LayerNorm(dim1 + dim2 + dim3)
        self.fc1 = torch.nn.Linear(dim1 + dim2 + dim3, dim4)
        self.fc2 = torch.nn.Linear(dim4, dim5)
        self.act = torch.nn.ReLU()

        torch.nn.init.xavier_normal_(self.fc1.weight)
        torch.nn.init.xavier_normal_(self.fc2.weight)

        # original
        # """
        self.MLP_two_nodes = torch.nn.Linear(dim1 + dim2 + dim3, dim4)
        self.MLP_last = torch.nn.Linear(dim4, dim4)
        # """

        # new
        # self.MLP_two_nodes = torch.nn.Linear(dim1 + dim2, dim4)
        # self.MLP_last = torch.nn.Linear(dim4 * 2, dim4)

        # self.graph_dim = dim5
        # self.graph = nn.Sequential(nn.Linear(dim5 * dim5, dim5 * dim5), nn.ReLU(), nn.Linear(dim5* dim5, dim5))

    def forward(self, x1, x2, x3):
        # # MLP(u;v;w)
        # x = torch.cat([x1, x2, x3], dim=-1)
        # x = self.layer_norm(x)
        # h = self.act(self.fc1(x))
        # return self.fc2(h)

        # # MLP(MLP(v;w) + MLP(u;w))
        # f_uw = torch.cat([x1, x3], 1)
        # f_vw = torch.cat([x2, x3], 1)
        # # encodings = self.trainable_embedding_distance_agg(torch.cat([uw+vw, torch.abs(uw-vw)], 1))
        # f_uw = self.act(self.MLP_two_nodes(f_uw))
        # f_vw = self.act(self.MLP_two_nodes(f_vw))
        
        # h = torch.cat([f_uw+f_vw, torch.abs(f_uw-f_vw)], 1)
        # h = self.fc2(self.act(self.MLP_last(h)))

        # Original
        # """
        f_uw = torch.cat([x1, x2, x3], dim=-1)
        f_vw = torch.cat([x2, x1, x3], dim=-1)
        f_uw = self.act(self.MLP_two_nodes(f_uw))
        f_vw = self.act(self.MLP_two_nodes(f_vw))
        h = f_uw + f_vw
        h = self.fc2(self.act(self.MLP_last(h)))
        # """

        # p = (h.view(-1, self.graph_dim, 1) * h.view(-1, 1, self.graph_dim)).view(-1, self.graph_dim * self.graph_dim)
        # final_graph = self.graph(p)
        return h
        # return h #, final_graph

class ScaledDotProductAttention(torch.nn.Module):
    ''' Scaled Dot-Product Attention '''

    def __init__(self, temperature, attn_dropout=0.1):
        super().__init__()
        self.temperature = temperature
        self.dropout = torch.nn.Dropout(attn_dropout)
        self.softmax = torch.nn.Softmax(dim=2)

    def forward(self, q, k, v, mask=None):
        # q: [B*N_src*n_head, 1, d_k]; k: [B*N_src*n_head, num_neighbors, d_k]
        # v: [B*N_src*n_head, num_neighbors, d_v], mask: [B*N_src*n_head, 1, num_neighbors]
        attn = torch.bmm(q, k.transpose(-1, -2))  # [B*N_src*n_head, 1, num_neighbors]
        attn = attn / self.temperature

        if mask is not None:
            attn = attn.masked_fill(mask, -1e10)

        attn = self.softmax(attn) # [n * b, l_q, l_k]
        attn = self.dropout(attn) # [n * b, l_v, d]

        output = torch.bmm(attn, v)  # [B*N_src*n_head, 1, d_v]

        return output, attn


class MultiHeadAttention(nn.Module):
    ''' Multi-Head Attention module '''

    def __init__(self, n_head, d_model, d_k, d_v, dropout=0.1):
        super().__init__()

        self.n_head = n_head
        self.d_k = d_k
        self.d_v = d_v

        self.w_qs = nn.Linear(d_model, n_head * d_k, bias=False)
        self.w_ks = nn.Linear(d_model, n_head * d_k, bias=False)
        self.w_vs = nn.Linear(d_model, n_head * d_v, bias=False)
        nn.init.normal_(self.w_qs.weight, mean=0, std=np.sqrt(2.0 / (d_model + d_k)))
        nn.init.normal_(self.w_ks.weight, mean=0, std=np.sqrt(2.0 / (d_model + d_k)))
        nn.init.normal_(self.w_vs.weight, mean=0, std=np.sqrt(2.0 / (d_model + d_v)))

        self.attention = ScaledDotProductAttention(temperature=np.power(d_k, 0.5), attn_dropout=dropout)
        self.layer_norm = nn.LayerNorm(d_model)

        self.fc = nn.Linear(n_head * d_v, d_model)

        nn.init.xavier_normal_(self.fc.weight)

        self.dropout = nn.Dropout(dropout)


    def forward(self, q, k, v, mask=None):

        d_k, d_v, n_head = self.d_k, self.d_v, self.n_head

        B, N_src, _ = q.size() # [B, N_src, model_dim]
        B, N_ngh, _ = k.size() # [B, N_ngh, model_dim]
        B, N_ngh, _ = v.size() # [B, N_ngh, model_dim]
        assert(N_ngh % N_src == 0)
        num_neighbors = int(N_ngh / N_src)
        residual = q

        q = self.w_qs(q).view(B, N_src, 1, n_head, d_k)  # [B, N_src, 1, n_head, d_k]
        k = self.w_ks(k).view(B, N_src, num_neighbors, n_head, d_k)  # [B, N_src, num_neighbors, n_head, d_k]
        v = self.w_vs(v).view(B, N_src, num_neighbors, n_head, d_v)  # [B, N_src, num_neighbors, n_head, d_k]

        q = q.transpose(2, 3).contiguous().view(B*N_src*n_head, 1, d_k)  # [B*N_src*n_head, 1, d_k]
        k = k.transpose(2, 3).contiguous().view(B*N_src*n_head, num_neighbors, d_k)  # [B*N_src*n_head, num_neighbors, d_k]
        v = v.transpose(2, 3).contiguous().view(B*N_src*n_head, num_neighbors, d_v)  # [B*N_src*n_head, num_neighbors, d_v]
        mask = mask.view(B*N_src, 1, num_neighbors).repeat(n_head, 1, 1) # [B*N_src*n_head, 1, num_neighbors]
        output, attn_map = self.attention(q, k, v, mask=mask) # output: [B*N_src*n_head, 1, d_v], attn_map: [B*N_src*n_head, 1, num_neighbors]

        output = output.view(B, N_src, n_head*d_v)  # [B, N_src, n_head*d_v]
        output = self.dropout(self.fc(output))  # [B, N_src, model_dim]
        output = self.layer_norm(output + residual)  # [B, N_src, model_dim]
        attn_map = attn_map.view(B, N_src, n_head, num_neighbors)
        return output, attn_map


class MapBasedMultiHeadAttention(nn.Module):
    ''' Multi-Head Attention module '''

    def __init__(self, n_head, d_model, d_k, d_v, dropout=0.1):
        super().__init__()

        self.n_head = n_head
        self.d_k = d_k
        self.d_v = d_v

        self.wq_node_transform = nn.Linear(d_model, n_head * d_k, bias=False)
        self.wk_node_transform = nn.Linear(d_model, n_head * d_k, bias=False)
        self.wv_node_transform = nn.Linear(d_model, n_head * d_k, bias=False)

        self.layer_norm = nn.LayerNorm(d_model)

        self.fc = nn.Linear(n_head * d_v, d_model)

        self.act = nn.LeakyReLU(negative_slope=0.2)
        self.weight_map = nn.Linear(2 * d_k, 1, bias=False)

        nn.init.xavier_normal_(self.fc.weight)

        self.dropout = torch.nn.Dropout(dropout)
        self.softmax = torch.nn.Softmax(dim=2)

        self.dropout = nn.Dropout(dropout)

    def forward(self, q, k, v, mask=None):

        d_k, d_v, n_head = self.d_k, self.d_v, self.n_head

        sz_b, len_q, _ = q.size()

        sz_b, len_k, _ = k.size()
        sz_b, len_v, _ = v.size()

        residual = q

        q = self.wq_node_transform(q).view(sz_b, len_q, n_head, d_k)

        k = self.wk_node_transform(k).view(sz_b, len_k, n_head, d_k)

        v = self.wv_node_transform(v).view(sz_b, len_v, n_head, d_v)

        q = q.permute(2, 0, 1, 3).contiguous().view(-1, len_q, d_k) # (n*b) x lq x dk
        q = torch.unsqueeze(q, dim=2) # [(n*b), lq, 1, dk]
        q = q.expand(q.shape[0], q.shape[1], len_k, q.shape[3]) # [(n*b), lq, lk, dk]

        k = k.permute(2, 0, 1, 3).contiguous().view(-1, len_k, d_k) # (n*b) x lk x dk
        k = torch.unsqueeze(k, dim=1) # [(n*b), 1, lk, dk]
        k = k.expand(k.shape[0], len_q, k.shape[2], k.shape[3]) # [(n*b), lq, lk, dk]

        v = v.permute(2, 0, 1, 3).contiguous().view(-1, len_v, d_v) # (n*b) x lv x dv

        mask = mask.repeat(n_head, 1, 1) # (n*b) x lq x lk

        # Map based Attention
        # output, attn = self.attention(q, k, v, mask=mask)
        q_k = torch.cat([q, k], dim=3) # [(n*b), lq, lk, dk * 2]
        attn = self.weight_map(q_k).squeeze(dim=3) # [(n*b), lq, lk]

        if mask is not None:
            attn = attn.masked_fill(mask, -1e10)

        attn = self.softmax(attn) # [n * b, l_q, l_k]
        attn = self.dropout(attn) # [n * b, l_q, l_k]

        # [n * b, l_q, l_k] * [n * b, l_v, d_v] >> [n * b, l_q, d_v]
        output = torch.bmm(attn, v)

        output = output.view(n_head, sz_b, len_q, d_v)

        output = output.permute(1, 2, 0, 3).contiguous().view(sz_b, len_q, -1) # b x lq x (n*dv)

        output = self.dropout(self.act(self.fc(output)))
        output = self.layer_norm(output + residual)

        return output, attn


def expand_last_dim(x, num):
    view_size = list(x.size()) + [1]
    expand_size = list(x.size()) + [num]
    return x.view(view_size).expand(expand_size)


class TimeEncode(torch.nn.Module):
    def __init__(self, expand_dim, factor=5):
        super(TimeEncode, self).__init__()

        self.time_dim = expand_dim
        self.factor = factor
        self.basis_freq = torch.nn.Parameter((torch.from_numpy(1 / 10 ** np.linspace(0, 9, self.time_dim))).float())
        self.phase = torch.nn.Parameter(torch.zeros(self.time_dim).float())


    def forward(self, ts):
        # ts: [N, L]
        batch_size = ts.size(0)
        seq_len = ts.size(1)

        ts = ts.view(batch_size, seq_len, 1)  # [N, L, 1]
        map_ts = ts * self.basis_freq.view(1, 1, -1)  # [N, L, time_dim]
        map_ts += self.phase.view(1, 1, -1)

        harmonic = torch.cos(map_ts)

        return harmonic #self.dense(harmonic)


class PosEncode(torch.nn.Module):
    def __init__(self, expand_dim, seq_len):
        super().__init__()

        self.pos_embeddings = nn.Embedding(num_embeddings=seq_len, embedding_dim=expand_dim)

    def forward(self, ts):
        # ts: [N, L]
        order = ts.argsort()
        ts_emb = self.pos_embeddings(order)
        return ts_emb


class EmptyEncode(torch.nn.Module):
    def __init__(self, expand_dim):
        super().__init__()
        self.expand_dim = expand_dim

    def forward(self, ts):
        out = torch.zeros_like(ts).float()
        out = torch.unsqueeze(out, dim=-1)
        out = out.expand(out.shape[0], out.shape[1], self.expand_dim)
        return out


class LSTMPool(torch.nn.Module):
    def __init__(self, feat_dim, edge_dim, time_dim):
        super(LSTMPool, self).__init__()
        self.feat_dim = feat_dim
        self.time_dim = time_dim
        self.edge_dim = edge_dim

        self.att_dim = feat_dim + edge_dim + time_dim

        self.act = torch.nn.ReLU()

        self.lstm = torch.nn.LSTM(input_size=self.att_dim,
                                  hidden_size=self.feat_dim,
                                  num_layers=1,
                                  batch_first=True)
        self.merger = MergeLayer(feat_dim, feat_dim, feat_dim, feat_dim)

    def forward(self, src, src_t, seq, seq_t, seq_e, mask):
        # seq [B, N, D]
        # mask [B, N]
        seq_x = torch.cat([seq, seq_e, seq_t], dim=2)

        _, (hn, _) = self.lstm(seq_x)

        hn = hn[-1, :, :] #hn.squeeze(dim=0)

        out = self.merger.forward(hn, src)
        return out, None


class MeanPool(torch.nn.Module):
    def __init__(self, feat_dim, edge_dim):
        super(MeanPool, self).__init__()
        self.edge_dim = edge_dim
        self.feat_dim = feat_dim
        self.act = torch.nn.ReLU()
        self.merger = MergeLayer(edge_dim + feat_dim, feat_dim, feat_dim, feat_dim)

    def forward(self, src, src_t, seq, seq_t, seq_e, mask):
        # seq [B, N, D]
        # mask [B, N]
        src_x = src
        seq_x = torch.cat([seq, seq_e], dim=2) #[B, N, De + D]
        hn = seq_x.mean(dim=1) #[B, De + D]
        output = self.merger(hn, src_x)
        return output, None


class AttnModel(torch.nn.Module):
    """Attention based temporal layers
    """
    def __init__(self, feat_dim, edge_dim, time_dim, pos_dim, model_dim,
                 attn_mode='prod', n_head=2, drop_out=0.1):
        """
        args:
          feat_dim: dim for the node features
          edge_dim: dim for the temporal edge features
          time_dim: dim for the time encoding
          attn_mode: choose from 'prod' and 'map'
          n_head: number of heads in attention
          drop_out: probability of dropping a neural.
        """
        super(AttnModel, self).__init__()

        self.feat_dim = feat_dim
        self.edge_dim = edge_dim
        self.time_dim = time_dim
        self.pos_dim = pos_dim
        self.model_dim = model_dim

        self.merger = MergeLayer(self.model_dim, feat_dim, feat_dim, feat_dim)

        assert(self.model_dim % n_head == 0)
        self.logger = logging.getLogger(__name__)
        self.attn_mode = attn_mode

        if attn_mode == 'prod':
            self.multi_head_target = MultiHeadAttention(n_head,
                                             d_model=self.model_dim,
                                             d_k=self.model_dim // n_head,
                                             d_v=self.model_dim // n_head,
                                             dropout=drop_out)
            self.logger.info('Using scaled prod attention')

        elif attn_mode == 'map':
            self.multi_head_target = MapBasedMultiHeadAttention(n_head,
                                             d_model=self.model_dim,
                                             d_k=self.model_dim // n_head,
                                             d_v=self.model_dim // n_head,
                                             dropout=drop_out)
            self.logger.info('Using map based attention')
        else:
            raise ValueError('attn_mode can only be prod or map')

    def forward(self, src, src_t, src_p, seq, seq_t, seq_e, seq_p, mask):
        """"Attention based temporal attention forward pass
        args:
          src: float Tensor of shape [B, N_src, D]
          src_t: float Tensor of shape [B, N_src, Dt], Dt == D
          seq: float Tensor of shape [B, N_ngh, D]
          seq_t: float Tensor of shape [B, N_ngh, Dt]
          seq_e: float Tensor of shape [B, N_ngh, De], De == D
          mask: boolean Tensor of shape [B, N_ngh], where the true value indicate a null value in the sequence.
        returns:
          output, weight
          output: float Tensor of shape [B, D]
          weight: float Tensor of shape [B, N]
        """

        batch, N_src, _ = src.shape
        N_ngh = seq.shape[1]
        device = src.device
        src_e = torch.zeros((batch, N_src, self.edge_dim)).float().to(device)
        src_p_pad, seq_p_pad = src_p, seq_p
        if src_p is None:
            src_p_pad = torch.zeros((batch, N_src, self.pos_dim)).float().to(device)
            seq_p_pad = torch.zeros((batch, N_ngh, self.pos_dim)).float().to(device)
        q = torch.cat([src, src_e, src_t, src_p_pad], dim=2) # [B, N_src, D + De + Dt] -> [B, N_src, D]
        k = torch.cat([seq, seq_e, seq_t, seq_p_pad], dim=2) # [B, N_ngh, D + De + Dt] -> [B, N_ngh, D]
        output, attn = self.multi_head_target(q=q, k=k, v=k, mask=mask) # output: [B, N_src, D + De + Dt], attn: [B, N_src, n_head, num_neighbors]
        output = self.merger(output, src)
        return output, attn


class CAWN(torch.nn.Module):
    def __init__(self, n_feat, e_feat, agg='tree',
                 attn_mode='prod', use_time='time', attn_agg_method='attn',
                 pos_dim=0, pos_enc='spd', walk_pool='attn', walk_n_head=8, walk_mutual=False,
                 num_layers=3, n_head=4, drop_out=0.1, num_neighbors=20, cpu_cores=1,
                 verbosity=1, get_checkpoint_path=None, walk_linear_out=False, interpretation=False):
        super(CAWN, self).__init__()
        self.logger = logging.getLogger(__name__)
        self.verbosity = verbosity

        # subgraph extraction hyper-parameters
        self.num_neighbors, self.num_layers = process_sampling_numbers(num_neighbors, num_layers)
        self.ngh_finder = None

        # features
        self.n_feat_th = torch.nn.Parameter(torch.from_numpy(n_feat.astype(np.float32)), requires_grad=False)
        self.e_feat_th = torch.nn.Parameter(torch.from_numpy(e_feat.astype(np.float32)), requires_grad=False)

        # label
        self.num_class = 4

        # train mode or test mode
        self.test = True

        # dimensions of 4 elements: node, edge, time, position
        self.feat_dim = self.n_feat_th.shape[1]  # node feature dimension
        self.e_feat_dim = self.e_feat_th.shape[1]  # edge feature dimension
        self.time_dim = self.feat_dim  # default to be time feature dimension
        self.pos_dim = pos_dim  # position feature dimension
        self.pos_enc = pos_enc
        self.model_dim = self.feat_dim + self.e_feat_dim + self.time_dim + self.pos_dim
        print("model dim, posdim", self.model_dim, self.pos_dim)
        self.logger.info('neighbors: {}, node dim: {}, edge dim: {}, pos dim: {}, edge dim: {}'.format(self.num_neighbors, self.feat_dim, self.e_feat_dim, self.pos_dim, self.time_dim))

        # aggregation method
        self.agg = agg

        # walk-based attention/summation model hyperparameters
        self.walk_pool = walk_pool
        self.walk_n_head = walk_n_head
        self.walk_mutual = walk_mutual
        self.walk_linear_out = walk_linear_out

        # dropout for both tree and walk based model
        self.dropout_p = drop_out

        # embedding layers and encoders
        self.edge_raw_embed = torch.nn.Embedding.from_pretrained(self.e_feat_th, padding_idx=0, freeze=True)
        # self.source_edge_embed = nn.parameter(torch.tensor()self.e_feat_dim)
        self.node_raw_embed = torch.nn.Embedding.from_pretrained(self.n_feat_th, padding_idx=0, freeze=True)
        self.time_encoder = self.init_time_encoder(use_time, seq_len=self.num_neighbors[0])
        self.position_encoder = PositionEncoder(enc_dim=self.pos_dim, num_layers=self.num_layers, ngh_finder=self.ngh_finder,
                                                cpu_cores=cpu_cores, verbosity=verbosity, logger=self.logger, enc=self.pos_enc)

        # attention model
        if self.agg == 'tree':
            self.attn_model_list = self.init_attn_model_list(attn_agg_method, attn_mode, n_head, drop_out)
        elif self.agg == 'walk':
            self.random_walk_attn_model = self.init_random_walk_attn_model()
        else:
            raise NotImplementedError('{} forward propagation strategy not implemented.'.format(self.agg))

        # final projection layer
        if interpretation:
            self.interpretation = True
            self.affinity_score = finalClassifier_inter(self.feat_dim, self.feat_dim, self.feat_dim, self.feat_dim, 4)
        else:
            self.interpretation = False
            self.affinity_score = finalClassifier(self.feat_dim, self.feat_dim, self.feat_dim, self.feat_dim, 4) # cls_tri, opn_tri, wedge, neg
        #torch.nn.Bilinear(self.feat_dim, self.feat_dim, 1, bias=True)

        self.get_checkpoint_path = get_checkpoint_path

        self.flag_for_cur_edge = True  # flagging whether the current edge under computation is real edges, for data analysis
        # self.common_node_percentages = {'pos': [], 'neg': []}
        # self.walk_encodings_scores = {'encodings': [], 'scores': []}

    def init_attn_model_list(self, attn_agg_method, attn_mode, n_head, drop_out):
        if attn_agg_method == 'attn':
            self.logger.info('Aggregation uses attention model')
            attn_model_list = torch.nn.ModuleList([AttnModel(self.feat_dim, self.e_feat_dim, self.time_dim,
                                                             self.pos_dim, self.model_dim,
                                                             attn_mode=attn_mode, n_head=n_head, drop_out=drop_out)
                                                   for _ in range(self.num_layers)])
        elif attn_agg_method == 'lstm':
            self.logger.info('Aggregation uses LSTM model')
            attn_model_list = torch.nn.ModuleList([LSTMPool(self.feat_dim,
                                                                 self.feat_dim,
                                                                 self.feat_dim) for _ in range(self.num_layers)])
        elif attn_agg_method == 'mean':
            self.logger.info('Aggregation uses constant mean model')
            attn_model_list = torch.nn.ModuleList([MeanPool(self.feat_dim,
                                                                 self.feat_dim) for _ in range(self.num_layers)])
        else:
            raise NotImplementedError('invalid agg_method value, use attn or lstm')
        return attn_model_list

    def init_random_walk_attn_model(self):
        random_walk_attn_model = RandomWalkAttention(feat_dim=self.model_dim, pos_dim=self.pos_dim,
                                                     model_dim=self.model_dim, out_dim=self.feat_dim,
                                                     walk_pool=self.walk_pool,
                                                     n_head=self.walk_n_head, mutual=self.walk_mutual,
                                                     dropout_p=self.dropout_p, logger=self.logger, walk_linear_out=self.walk_linear_out)
        return random_walk_attn_model

    def init_time_encoder(self, use_time, seq_len):
        if use_time == 'time':
            self.logger.info('Using time encoding')
            time_encoder = TimeEncode(expand_dim=self.time_dim)
        elif use_time == 'pos':
            assert(seq_len is not None)
            self.logger.info('Using positional encoding')
            time_encoder = PosEncode(expand_dim=self.time_dim, seq_len=seq_len)
        elif use_time == 'empty':
            self.logger.info('Using empty encoding')
            time_encoder = EmptyEncode(expand_dim=self.time_dim)
        else:
            raise ValueError('invalid time option!')
        return time_encoder

    def contrast(self, src_idx_l_1, src_idx_l_2, tgt_idx_l, cut_time_l_pos, e_idx_l_pos):
        '''
        1. grab subgraph for src, tgt, bgd
        2. add positional encoding for src & tgt nodes
        3. forward propagate to get src embeddings and tgt embeddings (and finally pos_score (shape: [batch, ]))
        4. forward propagate to get src embeddings and bgd embeddings (and finally neg_score (shape: [batch, ]))
        '''
        start = time.time()
        # cut_time_l_pos += 1e-8 # original is integer. 1e7, so the delta t can be 0.0001
        subgraph_src_1_pos = self.grab_subgraph(src_idx_l_1, cut_time_l_pos, e_idx_l=e_idx_l_pos)
        subgraph_src_2_pos = self.grab_subgraph(src_idx_l_2, cut_time_l_pos, e_idx_l=e_idx_l_pos)
        subgraph_tgt = self.grab_subgraph(tgt_idx_l, cut_time_l_pos, e_idx_l=None)
        end = time.time()
        if self.verbosity > 1:
            self.logger.info('grab subgraph for the minibatch, time eclipsed: {} seconds'.format(str(end-start)))
        self.flag_for_cur_edge = True
        # if self.interpretation:
        score, walk_bucket = self.forward(src_idx_l_1, src_idx_l_2, tgt_idx_l, cut_time_l_pos, (subgraph_src_1_pos, subgraph_src_2_pos, subgraph_tgt), test=self.test)
        # else:
        #     # walk_bucket is None
        #     score, walk_bucket = self.forward(src_idx_l_1, src_idx_l_2, tgt_idx_l, cut_time_l_pos, (subgraph_src_1_pos, subgraph_src_2_pos, subgraph_tgt), test=self.test)
        # pos_score = self.forward(src_idx_l, tgt_idx_l, cut_time_l, (subgraph_src, subgraph_tgt), test=test)
        self.flag_for_cur_edge = False
        # neg_score1 = self.forward(src_idx_l, bgd_idx_l, cut_time_l, (subgraph_src, subgraph_bgd), test=test)
        # neg_score2 = self.forward(tgt_idx_l, bgd_idx_l, cut_time_l, (subgraph_tgt, subgraph_bgd))
        # return pos_score.sigmoid(), (neg_score1.sigmoid() + neg_score2.sigmoid())/2.0
        # return pos_score.sigmoid(), neg_score.sigmoid()
        # if self.interpretation:
        return score, walk_bucket
        # else:
        #     return score, None
        # original sigmoid, because we use CrossEntropyLoss, so no softmax
        # return torch.nn.functional.softmax(score)

    def forward(self, src_idx_l_1, src_idx_l_2, tgt_idx_l, cut_time_l, subgraphs=None, test=False):
        if subgraphs is not None:
            subgraph_src_1, subgraph_src_2, subgraph_tgt = subgraphs
        else: # not used in our code but is still a useful branch when negative sample is not provided
            subgraph_src_1 = self.grab_subgraph(src_idx_l_1, cut_time_l, e_idx_l=None)  # TODO: self.grab_subgraph(), with e_idx_l
            subgraph_src_2 = self.grab_subgraph(src_idx_l_2, cut_time_l, e_idx_l=None)
            subgraph_tgt = self.grab_subgraph(tgt_idx_l, cut_time_l, e_idx_l=None)
        self.position_encoder.init_internal_data(src_idx_l_1, src_idx_l_2, tgt_idx_l, cut_time_l, subgraph_src_1, subgraph_src_2, subgraph_tgt)
        if self.agg == 'walk':  #TODO: can we do this later to save position coding time, since walk-based has too much redundancy?
            # print('we do walk')
            subgraph_src_1 = self.subgraph_tree2walk(src_idx_l_1, cut_time_l, subgraph_src_1)
            subgraph_src_2 = self.subgraph_tree2walk(src_idx_l_2, cut_time_l, subgraph_src_2)
            subgraph_tgt = self.subgraph_tree2walk(tgt_idx_l, cut_time_l, subgraph_tgt)
        src_1_walk_pattern, src_embed_1 = self.forward_msg(src_idx_l_1, cut_time_l, subgraph_src_1, test=self.test, interpretation=self.interpretation)
        src_2_walk_pattern,src_embed_2 = self.forward_msg(src_idx_l_2, cut_time_l, subgraph_src_2, test=self.test, interpretation=self.interpretation)
        tgt_walk_pattern, tgt_embed = self.forward_msg(tgt_idx_l, cut_time_l, subgraph_tgt, test=self.test, interpretation=self.interpretation)
        # if self.agg == 'walk' and self.walk_mutual:
        #     src_embed_1, src_embed_2, tgt_embed = self.tune_msg(src_embed_1, src_embed_2, tgt_embed)
        feature_dim = src_embed_1.shape[-1]
        
        if self.interpretation:
            if self.test:
                src_1_walks_score = self.affinity_score(src_embed_1.view(-1, feature_dim), None, None)
                src_2_walks_score = self.affinity_score(src_embed_2.view(-1, feature_dim), None, None)
                tgt_walks_score = self.affinity_score(tgt_embed.view(-1, feature_dim), None, None)
                score = (src_1_walks_score + src_2_walks_score + tgt_walks_score).view(len(src_idx_l_1), -1, self.num_class).mean(1)
            # print(score.size())
                return score, (src_1_walks_score, src_2_walks_score, tgt_walks_score, src_1_walk_pattern, src_2_walk_pattern, tgt_walk_pattern)
            else:
                score = self.affinity_score(src_embed_1.mean(1), src_embed_2.mean(1), tgt_embed.mean(1)) #.squeeze(dim=-1)
                return score, None
        else:
            score = self.affinity_score(src_embed_1, src_embed_2, tgt_embed) # .squeeze(dim=-1) # score_walk shape: [B, M]
            # score = score.squeeze(dim=-1)
            return score, None
            # src_1_walks_score = None
            # src_2_walks_score = None
            # tgt_walks_score = None
        
        # score_graph = score_graph.squeeze(dim=-1)
        # score.squeeze_(dim=-1)
        # if test:
        #     self.walk_encodings_scores['scores'].append(score_walk)

        # return score# , score_graph

    def grab_subgraph(self, src_idx_l, cut_time_l, e_idx_l=None):
        subgraph = self.ngh_finder.find_k_hop(self.num_layers, src_idx_l, cut_time_l, num_neighbors=self.num_neighbors, e_idx_l=e_idx_l)
        return subgraph

    def subgraph_tree2walk(self, src_idx_l, cut_time_l, subgraph_src):
        # put src nodes and extracted subgraph together
        node_records, eidx_records, t_records = subgraph_src
        node_records_tmp = [np.expand_dims(src_idx_l, 1)] + node_records
        eidx_records_tmp = [np.zeros_like(node_records_tmp[0])] + eidx_records
        t_records_tmp = [np.expand_dims(cut_time_l, 1)] + t_records

        # use the list to construct a new matrix
        new_node_records = self.subgraph_tree2walk_one_component(node_records_tmp)
        new_eidx_records = self.subgraph_tree2walk_one_component(eidx_records_tmp)
        new_t_records = self.subgraph_tree2walk_one_component(t_records_tmp)
        return new_node_records, new_eidx_records, new_t_records

    def subgraph_tree2walk_one_component(self, record_list):
        batch, n_walks, walk_len, dtype = record_list[0].shape[0], record_list[-1].shape[-1], len(record_list), record_list[0].dtype
        record_matrix = np.empty((batch, n_walks, walk_len), dtype=dtype)
        for hop_idx, hop_record in enumerate(record_list):
            assert(n_walks % hop_record.shape[-1] == 0)
            record_matrix[:, :, hop_idx] = np.repeat(hop_record, repeats=n_walks // hop_record.shape[-1], axis=1)
        return record_matrix

    def forward_msg(self, src_idx_l, cut_time_l, subgraph_src, test=False, interpretation=False):
        node_records, eidx_records, t_records = subgraph_src
        # NOTE: we assume raw node features are static in this problem
        # 1. initialize 0-layer hidden embeddings with raw node features of all hops (later with positional encodings as well)
        # 2. get time encodings for all hops
        # 3. get edge features for all in-between hops
        # 4. iterate over hidden embeddings for each layer
        hidden_embeddings, masks = self.init_hidden_embeddings(src_idx_l, node_records)  # length self.num_layers+1
        time_features = self.retrieve_time_features(cut_time_l, t_records)  # length self.num_layers+1
        edge_features = self.retrieve_edge_features(eidx_records)  # length self.num_layers
        position_features, walk_pattern = self.retrieve_position_features(src_idx_l, node_records, cut_time_l, t_records, test=test, interpretation=interpretation)  # length self.num_layers+1, core contribution
        if self.agg == 'tree':
            n_layer = self.num_layers
            for layer in range(n_layer):
                hidden_embeddings = self.forward_msg_layer(hidden_embeddings, time_features[:n_layer+1-layer],
                                                           edge_features[:n_layer-layer], position_features[:n_layer+1-layer],
                                                           masks[:n_layer-layer], self.attn_model_list[layer])
            final_node_embeddings = hidden_embeddings[0].squeeze(1)
        elif self.agg == 'walk':
            # Notice that eidx_records[:, :, 1] may be all None
            # random walk branch logic:
            # 1. get the feature matrix shaped [batch, n_walk, len_walk + 1, node_dim + edge_dim + time_dim + pos_dim]
            # 2. feed the matrix forward to LSTM, then transformer, now shaped [batch, n_walk, transformer_model_dim]
            # 3. aggregate and collapse dim=1 (using set operation), now shaped [batch, out_dim]
            final_node_embeddings = self.forward_msg_walk(hidden_embeddings, time_features, edge_features, position_features, masks)
        else:
            raise NotImplementedError('{} forward propagation strategy not implemented.'.format(self.agg))
        return walk_pattern, final_node_embeddings

    def tune_msg(self, src_embed_1, src_embed_2, tgt_embed):
        return self.random_walk_attn_model.mutual_query( src_embed_1, src_embed_2, tgt_embed)

    def init_hidden_embeddings(self, src_idx_l, node_records):
        device = self.n_feat_th.device
        if self.agg == 'tree':
            hidden_embeddings, masks = [], []
            hidden_embeddings.append(self.node_raw_embed(torch.from_numpy(np.expand_dims(src_idx_l, 1)).long().to(device)))
            for i in range(len(node_records)):
                batch_node_idx = torch.from_numpy(node_records[i]).long().to(device)
                hidden_embeddings.append(self.node_raw_embed(batch_node_idx))
                masks.append(batch_node_idx == 0)
        elif self.agg == 'walk':
            node_records_th = torch.from_numpy(node_records).long().to(device)
            hidden_embeddings = self.node_raw_embed(node_records_th)  # shape [batch, n_walk, len_walk+1, node_dim]
            masks = (node_records_th != 0).sum(dim=-1).long()  # shape [batch, n_walk], here the masks means differently: it records the valid length of each walk
        else:
            raise NotImplementedError('{} forward propagation strategy not implemented.'.format(self.agg))
        return hidden_embeddings, masks

    def retrieve_time_features(self, cut_time_l, t_records):
        device = self.n_feat_th.device
        batch = len(cut_time_l)
        if self.agg == 'tree':
            first_time_stamp = np.expand_dims(cut_time_l, 1)
            time_features = [self.time_encoder(torch.from_numpy(np.zeros_like(first_time_stamp)).float().to(device))]
            standard_timestamps = np.expand_dims(first_time_stamp, 2)
            for layer_i in range(len(t_records)):
                t_record = t_records[layer_i]
                time_delta = standard_timestamps - t_record.reshape(batch, -1, self.num_neighbors[layer_i])
                time_delta = time_delta.reshape(batch, -1)
                time_delta = torch.from_numpy(time_delta).float().to(device)
                time_features.append(self.time_encoder(time_delta))
                standard_timestamps = np.expand_dims(t_record, 2)
        elif self.agg == 'walk':
            t_records_th = torch.from_numpy(t_records).float().to(device)
            t_records_th = t_records_th.select(dim=-1, index=0).unsqueeze(dim=2) - t_records_th
            n_walk, len_walk = t_records_th.size(1), t_records_th.size(2)
            time_features = self.time_encoder(t_records_th.view(batch, -1)).view(batch, n_walk, len_walk,
                                                                                 self.time_encoder.time_dim)
        else:
            raise NotImplementedError('{} forward propagation strategy not implemented.'.format(self.agg))
        return time_features

    def retrieve_edge_features(self, eidx_records):
        # Notice that if subgraph is tree, then len(eidx_records) is just the number of hops, excluding the src node
        # but if subgraph is walk, then eidx_records contains the random walks of length len_walk+1, including the src node
        device = self.n_feat_th.device
        if self.agg == 'tree':
            edge_features = []
            for i in range(len(eidx_records)):
                batch_edge_idx = torch.from_numpy(eidx_records[i]).long().to(device)
                edge_features.append(self.edge_raw_embed(batch_edge_idx))
        elif self.agg == 'walk':
            eidx_records_th = torch.from_numpy(eidx_records).to(device)
            eidx_records_th[:, :, 0] = 0   # NOTE: this will NOT be mixed with padded 0's since those paddings are denoted by masks and will be ignored later in lstm
            edge_features = self.edge_raw_embed(eidx_records_th)  # shape [batch, n_walk, len_walk+1, edge_dim]
        else:
            raise NotImplementedError('{} forward propagation strategy not implemented.'.format(self.agg))
        return edge_features

    def forward_msg_layer(self, hidden_embeddings, time_features, edge_features, position_features, masks, attn_m):
        assert(len(hidden_embeddings) == len(time_features)) 
        assert(len(hidden_embeddings) == (len(edge_features) + 1)) 
        assert(len(masks) == len(edge_features))
        assert(len(hidden_embeddings) == len(position_features))
        new_src_embeddings = []
        for i in range(len(edge_features)):
            src_embedding = hidden_embeddings[i]
            src_time_feature = time_features[i]
            src_pos_feature = position_features[i]
            ngh_embedding = hidden_embeddings[i+1]
            ngh_time_feature = time_features[i+1]
            ngh_edge_feature = edge_features[i]
            ngh_pos_feature = position_features[i+1]
            ngh_mask = masks[i]
            # NOTE: n_neighbor_support = n_source_support * num_neighbor this layer
            # new_src_embedding shape: [batch, n_source_support, feat_dim]
            # attn_map shape: [batch, n_source_support, n_head, num_neighbors]
            new_src_embedding, attn_map = attn_m(src_embedding,  # shape [batch, n_source_support, feat_dim]
                                                 src_time_feature,  # shape [batch, n_source_support, time_feat_dim]
                                                 src_pos_feature, # shape [batch, n_source_support, pos_dim]
                                                 ngh_embedding,  # shape [batch, n_neighbor_support, feat_dim]
                                                 ngh_time_feature,  # shape [batch, n_neighbor_support, time_feat_dim]
                                                 ngh_edge_feature,  # shape [batch, n_neighbor_support, edge_feat_dim]
                                                 ngh_pos_feature, # shape [batch, n_neighbor_support, pos_dim]
                                                 ngh_mask)  # shape [batch, n_neighbor_support]

            new_src_embeddings.append(new_src_embedding)
        return new_src_embeddings

    def forward_msg_walk(self, hidden_embeddings, time_features, edge_features, position_features, masks):
        return self.random_walk_attn_model.forward_one_node(hidden_embeddings, time_features, edge_features,
                                                            position_features, masks)

    def retrieve_position_features(self, src_idx_l, node_records, cut_time_l, t_records, test=False, interpretation=False):
        start = time.time()
        encode = self.position_encoder
        walk_pattern = None
        if self.agg == 'tree':
            if encode.enc_dim == 0:
                return [None]*(len(node_records)+1)
            position_feature, common_nodes = encode(np.expand_dims(src_idx_l, 1), np.expand_dims(cut_time_l, 1))
            position_features = [position_feature]
            for i in range(len(node_records)):
                position_feature, common_nodes = encode(node_records[i], t_records[i])
                position_features.append(position_feature)
                # self.update_common_node_percentages(common_nodes)
        elif self.agg == 'walk':
            if encode.enc_dim == 0:
                return None
            batch, n_walk, len_walk = node_records.shape
            new_node_records = self.position_encoder.anonymize(node_records)
            node_records_r, t_records_r = new_node_records.reshape(batch, -1), t_records.reshape(batch, -1)
            # if test:
            #     self.walk_encodings_scores['encodings'].append(walk_encodings)
            position_features, common_nodes, walk_encodings, walk_pattern = encode(node_records_r, t_records_r, test, interpretation)
            position_features = position_features.view(batch, n_walk, len_walk, self.pos_dim)
            # self.update_common_node_percentages(common_nodes)
            # if test:
            #     self.walk_encodings_scores['encodings'].append(walk_encodings)
        else:
            raise NotImplementedError('{} forward propagation strategy not implemented.'.format(self.agg))
        end = time.time()
        if self.verbosity > 1:
            self.logger.info('encode positions encodings for the minibatch, time eclipsed: {} seconds'.format(str(end-start)))
        return position_features, walk_pattern

    def update_ngh_finder(self, ngh_finder):
        self.ngh_finder = ngh_finder
        self.position_encoder.ngh_finder = ngh_finder

    def update_common_node_percentages(self, common_node_percentage):
        if self.flag_for_cur_edge:
            self.common_node_percentages['pos'].append(common_node_percentage)
        else:
            self.common_node_percentages['neg'].append(common_node_percentage)

    def save_common_node_percentages(self, dir):
        torch.save(self.common_node_percentages, dir + '/common_node_percentages.pt')

    def save_walk_encodings_scores(self, dir):
        torch.save(self.walk_encodings_scores, dir + '/walk_encodings_scores.pt')


class PositionEncoder(nn.Module):
    '''
    Note that encoding initialization and lookup is done on cpu but encoding (post) projection is on device
    '''
    def __init__(self, num_layers, enc='spd', enc_dim=2, ngh_finder=None, verbosity=1, cpu_cores=1, logger=None):
        super(PositionEncoder, self).__init__()
        self.enc = enc
        self.enc_dim = enc_dim
        self.num_layers = num_layers
        self.nodetime2emb_maps = None
        self.projection = nn.Linear(1, 1)  # reserved for when the internal position encoding does not match input
        self.cpu_cores = cpu_cores
        self.ngh_finder = ngh_finder
        self.verbosity = verbosity
        self.logger = logger

        """
        for interpretation, add the bucket part
        """
        self.pattern = {}
        self.pattern_num = 0

        if self.enc == 'spd':
            # self.trainable_embedding = nn.Embedding(num_embeddings=self.num_layers+2, embedding_dim=self.enc_dim) # [0, 1, ... num_layers, inf]
            # self.trainable_embedding_src = nn.Embedding(num_embeddings=self.num_layers+2, embedding_dim=self.enc_dim) # [0, 1, ... num_layers, inf] TODO: relu?
            # self.trainable_embedding_tgt = nn.Embedding(num_embeddings=self.num_layers+2, embedding_dim=self.enc_dim) # [0, 1, ... num_layers, inf] TODO: relu?
            # self.trainable_embedding_concat = nn.Sequential(nn.Linear(in_features=self.enc_dim * 2, out_features=self.enc_dim))
            
            # final
            self.trainable_embedding_distance = nn.Embedding(num_embeddings=self.num_layers+2, embedding_dim=self.enc_dim)
            self.trainable_embedding_distance_MLP = nn.Sequential(nn.Linear(in_features=2 * self.enc_dim, out_features=self.enc_dim))
            self.trainable_embedding_distance_agg = nn.Sequential(nn.Linear(in_features=self.enc_dim * 2, out_features=self.enc_dim), nn.ReLU(),
                                                                  nn.Linear(in_features=self.enc_dim, out_features=self.enc_dim), nn.ReLU(),
                                                                #   nn.Linear(in_features=self.enc_dim, out_features=self.enc_dim), nn.ReLU(),
                                                                  )

        else:
            assert(self.enc in ['lp', 'saw'])
            # self.trainable_embedding = nn.Sequential(nn.Linear(in_features=self.num_layers+1, out_features=self.enc_dim),
            #                                          nn.ReLU(),
            #                                          nn.Linear(in_features=self.enc_dim, out_features=self.enc_dim))  # landing prob at [0, 1, ... num_layers]
            # self.trainable_embedding_src = nn.Sequential(nn.Linear(in_features=self.num_layers+1, out_features=self.enc_dim),
            #                                              nn.ReLU(),
            #                                              )  # landing prob at [0, 1, ... num_layers]
            # self.trainable_embedding_tgt = nn.Sequential(nn.Linear(in_features=self.num_layers+1, out_features=self.enc_dim),
            #                                              nn.ReLU(),
            #                                              )  # landing prob at [0, 1, ... num_layers]
            # self.trainable_embedding_concat = nn.Sequential(nn.Linear(in_features=self.enc_dim * 2, out_features=self.enc_dim))

            # self.trainable_embedding_src_2 = nn.Sequential(nn.Linear(in_features=self.num_layers+1, out_features=self.enc_dim),
            #                                              nn.ReLU(),
            #                                              )
            # # self.trainable_embedding_uvw = nn.Sequential(nn.Linear(in_features=self.enc_dim * 3, out_features=self.enc_dim), nn.ReLU(),
            # #                                             #  nn.Linear(in_features=self.enc_dim, out_features=self.enc_dim)
            # #                                              )
            # self.trainable_embedding_uvw = nn.Sequential(nn.Linear(in_features=(self.num_layers+1) * 3, out_features=self.enc_dim), nn.ReLU(),
            #                                              nn.Linear(in_features=self.enc_dim, out_features=self.enc_dim), nn.ReLU()
            #                                              ) # MLP(u,v,w) + MLP(v,u,w)
            # self.trainable_embedding_uvw_MLP = nn.Sequential(nn.Linear(in_features=self.enc_dim, out_features=self.enc_dim)) # MLP(u,v,w) + MLP(v,u,w)

            # MLP(MLP(u, w) + MLP(v, w), abs(MLP(u, w) - MLP(v, w)))
            self.trainable_embedding_distance = nn.Sequential(nn.Linear(in_features=2 * (self.num_layers+1), out_features=self.enc_dim), nn.ReLU())
            self.trainable_embedding_distance_agg = nn.Sequential(nn.Linear(in_features=self.enc_dim * 2, out_features=self.enc_dim), nn.ReLU(),
                                                                  nn.Linear(in_features=self.enc_dim, out_features=self.enc_dim), nn.ReLU(),
                                                                #   nn.Linear(in_features=self.enc_dim, out_features=self.enc_dim), nn.ReLU(),
                                                                  )
        self.logger.info("Distance encoding: {}".format(self.enc))

    # def init_internal_data(self, src_idx_l, tgt_idx_l, cut_time_l, subgraph_src, subgraph_tgt):
    def init_internal_data(self, src_idx_l_1, src_idx_l_2, tgt_idx_l, cut_time_l, subgraph_src_1, subgraph_src_2, subgraph_tgt):
        if self.enc_dim == 0:
            return
        start = time.time()
        # initialize internal data structure to index node positions
        # self.nodetime2emb_maps = self.collect_pos_mapping_ptree(src_idx_l, tgt_idx_l, cut_time_l, subgraph_src,
        #                                                         subgraph_tgt)
        self.nodetime2emb_maps = self.collect_pos_mapping_ptree(src_idx_l_1, src_idx_l_2, tgt_idx_l, cut_time_l, subgraph_src_1, subgraph_src_2,
                                                                subgraph_tgt)
        end = time.time()
        if self.verbosity > 1:
            self.logger.info('init positions encodings for the minibatch, time eclipsed: {} seconds'.format(str(end-start)))

    # def collect_pos_mapping_ptree(self, src_idx_l, tgt_idx_l, cut_time_l, subgraph_src, subgraph_tgt):
    def collect_pos_mapping_ptree(self, src_idx_l_1, src_idx_l_2, tgt_idx_l, cut_time_l, subgraph_src_1, subgraph_src_2, subgraph_tgt):
        # Return:
        # nodetime2idx_maps: a list of dict {(node index, rounded time string) -> index in embedding look up matrix}
        if self.cpu_cores == 1:
            subgraph_src_node_1, _, subgraph_src_ts_1 = subgraph_src_1  # only use node index and timestamp to identify a node in temporal graph
            subgraph_src_node_2, _, subgraph_src_ts_2 = subgraph_src_2
            subgraph_tgt_node, _, subgraph_tgt_ts = subgraph_tgt
            nodetime2emb_maps = {}
            for row in range(len(src_idx_l_1)):
                src_1 = src_idx_l_1[row]
                src_2 = src_idx_l_2[row]
                tgt = tgt_idx_l[row]
                cut_time = cut_time_l[row]
                src_neighbors_node_1 = [k_hop_neighbors[row] for k_hop_neighbors in subgraph_src_node_1]
                src_neighbors_ts_1 = [k_hop_neighbors[row] for k_hop_neighbors in subgraph_src_ts_1]
                src_neighbors_node_2 = [k_hop_neighbors[row] for k_hop_neighbors in subgraph_src_node_2]
                src_neighbors_ts_2 = [k_hop_neighbors[row] for k_hop_neighbors in subgraph_src_ts_2]
                tgt_neighbors_node = [k_hop_neighbors[row] for k_hop_neighbors in subgraph_tgt_node]
                tgt_neighbors_ts = [k_hop_neighbors[row] for k_hop_neighbors in subgraph_tgt_ts]
                nodetime2emb_map = PositionEncoder.collect_pos_mapping_ptree_sample(src_1, src_2, tgt, cut_time,
                                                                   src_neighbors_node_1, src_neighbors_ts_1,
                                                                   src_neighbors_node_2, src_neighbors_ts_2,
                                                                   tgt_neighbors_node, tgt_neighbors_ts, batch_idx=row, enc=self.enc) # why batch_idx?
                nodetime2emb_maps.update(nodetime2emb_map)
        else:
            # multiprocessing version, no significant gain though
            cores = self.cpu_cores
            if cores in [-1, 0]:
                cores = mp.cpu_count()
            pool = mp.Pool(processes=cores)
            nodetime2emb_maps = pool.map(PositionEncoder.collect_pos_mapping_ptree_sample_mp,
                                         [(src_idx_l_1, src_idx_l_2, tgt_idx_l, cut_time_l, subgraph_src_1, subgraph_src_2, subgraph_tgt, row, self.enc) for row in range(len(src_idx_l_1))],
                                         chunksize=len(src_idx_l_1)//cores+1)
            pool.close()
        return nodetime2emb_maps

    @staticmethod
    def collect_pos_mapping_ptree_sample(src_1, src_2, tgt, cut_time, src_neighbors_node_1, src_neighbors_ts_1, src_neighbors_node_2, src_neighbors_ts_2,
                                         tgt_neighbors_node, tgt_neighbors_ts, batch_idx, enc='spd'):
        """
        This function has the potential of being written in numba by using numba.typed.Dict!
        """
        n_hop = len(src_neighbors_node_1)
        makekey = nodets2key
        nodetime2emb = {}
        if enc == 'spd':
            for k in range(n_hop-1, -1, -1):
                for src_node_1, src_ts_1, src_node_2, src_ts_2, tgt_node, tgt_ts in zip(src_neighbors_node_1[k], src_neighbors_ts_1[k],
                                                                                        src_neighbors_node_2[k], src_neighbors_ts_2[k],
                                                                                        tgt_neighbors_node[k], tgt_neighbors_ts[k]):

                    src_key_1, src_key_2, tgt_key = makekey(batch_idx, src_node_1, src_ts_1), makekey(batch_idx, src_node_2, src_ts_2), makekey(batch_idx, tgt_node, tgt_ts)
                    # src_ts, tgt_ts = PositionEncoder.float2str(src_ts), PositionEncoder.float2str(tgt_ts)
                    # src_key, tgt_key = (src_node, src_ts), (tgt_node, tgt_ts)
                    if src_key_1 not in nodetime2emb:
                        nodetime2emb[src_key_1] = [k+1, 2*n_hop, 2*n_hop]  # 2*n_hop for disconnected case
                    else:
                        nodetime2emb[src_key_1][0] = k+1

                    if src_key_2 not in nodetime2emb:
                        nodetime2emb[src_key_2] = [2*n_hop, k+1 , 2*n_hop]  # 2*n_hop for disconnected case
                    else:
                        nodetime2emb[src_key_2][1] = k+1

                    if tgt_key not in nodetime2emb:
                        nodetime2emb[tgt_key] = [2*n_hop, 2*n_hop, k+1]
                    else:
                        nodetime2emb[tgt_key][2] = k+1
            # add two end nodes
            src_key_1 = makekey(batch_idx, src_1, cut_time)
            src_key_2 = makekey(batch_idx, src_2, cut_time)
            tgt_key = makekey(batch_idx, tgt, cut_time)
            # src_key = (src, PositionEncoder.float2str(cut_time))
            # tgt_key = (tgt, PositionEncoder.float2str(cut_time))
            if src_key_1 in nodetime2emb:
                nodetime2emb[src_key_1][0] = 0
            else:
                nodetime2emb[src_key_1] = [0, 2*n_hop, 2*n_hop]
            
            if src_key_2 in nodetime2emb:
                nodetime2emb[src_key_2][1] = 0
            else:
                nodetime2emb[src_key_2] = [2*n_hop, 0, 2*n_hop]

            if tgt_key in nodetime2emb:
                nodetime2emb[tgt_key][2] = 0
            else:
                nodetime2emb[tgt_key] = [2*n_hop, 2*n_hop, 0]
            null_key = makekey(batch_idx, 0, 0.0)
            nodetime2emb[null_key] = [2 * n_hop, 2 * n_hop, 2 * n_hop]
            # nodetime2emb[(0, PositionEncoder.float2str(0.0))] = [2*n_hop, 2*n_hop] # Fix a big bug with 0.0! Also, very important to keep null node far away from the two end nodes!
        elif enc == 'lp':
            # landing probability encoding, n_hop+1 types of probabilities for each node
            # src_neighbors_node, src_neighbors_ts = [[src]] + src_neighbors_node, [[cut_time]] + src_neighbors_ts
            src_neighbors_node_1, src_neighbors_ts_1 = [[src_1]] + src_neighbors_node_1, [[cut_time]] + src_neighbors_ts_1 
            src_neighbors_node_2, src_neighbors_ts_2 = [[src_2]] + src_neighbors_node_2, [[cut_time]] + src_neighbors_ts_2
            tgt_neighbors_node, tgt_neighbors_ts = [[tgt]] + tgt_neighbors_node, [[cut_time]] + tgt_neighbors_ts
            for k in range(n_hop+1):
                k_hop_total = len(src_neighbors_node_1[k])
                # for src_node, src_ts, tgt_node, tgt_ts in zip(src_neighbors_node[k], src_neighbors_ts[k],
                #                                               tgt_neighbors_node[k], tgt_neighbors_ts[k]):
                for src_node_1, src_ts_1, src_node_2, src_ts_2, tgt_node, tgt_ts in zip(src_neighbors_node_1[k], src_neighbors_ts_1[k],
                                                                                        src_neighbors_node_2[k], src_neighbors_ts_2[k],
                                                                                        tgt_neighbors_node[k], tgt_neighbors_ts[k]):
                    # src_key, tgt_key = makekey(batch_idx, src_node, src_ts), makekey(batch_idx, tgt_node, tgt_ts)
                    src_key_1, src_key_2, tgt_key = makekey(batch_idx, src_node_1, src_ts_1), makekey(batch_idx, src_node_2, src_ts_2), makekey(batch_idx, tgt_node, tgt_ts)
                    # src_ts, tgt_ts = PositionEncoder.float2str(src_ts), PositionEncoder.float2str(tgt_ts)
                    # src_key, tgt_key = (src_node, src_ts), (tgt_node, tgt_ts)
                    if src_key_1 not in nodetime2emb:
                        nodetime2emb[src_key_1] = np.zeros((3, n_hop+1), dtype=np.float32)
                    if src_key_2 not in nodetime2emb:
                        nodetime2emb[src_key_2] = np.zeros((3, n_hop+1), dtype=np.float32)
                    if tgt_key not in nodetime2emb:
                        nodetime2emb[tgt_key] = np.zeros((3, n_hop+1), dtype=np.float32)
                    nodetime2emb[src_key_1][0, k] += 1/k_hop_total  # convert into landing probabilities by normalizing with k hop sampling number
                    nodetime2emb[src_key_2][1, k] += 1/k_hop_total  # convert into landing probabilities by normalizing with k hop sampling number
                    nodetime2emb[tgt_key][2, k] += 1/k_hop_total  # convert into landing probabilities by normalizing with k hop sampling number
            null_key = makekey(batch_idx, 0, 0.0)
            nodetime2emb[null_key] = np.zeros((3, n_hop + 1), dtype=np.float32)
            # nodetime2emb[(0, PositionEncoder.float2str(0.0))] = np.zeros((2, n_hop+1), dtype=np.float32)
        else:
            assert(enc == 'saw')  # self-based anonymous walk, no mutual distance encoding
            # src_neighbors_node_1, src_neighbors_ts_1 = [[src_1]] + src_neighbors_node_1, [[cut_time]] + src_neighbors_ts_1
            # src_neighbors_node_2, src_neighbors_ts_2 = [[src_2]] + src_neighbors_node_2, [[cut_time]] + src_neighbors_ts_2
            # tgt_neighbors_node, tgt_neighbors_ts = [[tgt]] + tgt_neighbors_node, [[cut_time]] + tgt_neighbors_ts
            # src_seen_nodes2label_1 = {}
            # src_seen_nodes2label_2 = {}
            # tgt_seen_nodes2label = {}
            # for k in range(n_hop + 1):
            #     # for src_node, src_ts, tgt_node, tgt_ts in zip(src_neighbors_node[k], src_neighbors_ts[k],
            #     #                                               tgt_neighbors_node[k], tgt_neighbors_ts[k]):
            #     for src_node_1, src_ts_1, src_node_2, src_ts_2, tgt_node, tgt_ts in zip(src_neighbors_node_1[k], src_neighbors_ts_1[k],
            #                                                                             src_neighbors_node_2[k], src_neighbors_ts_2[k],
            #                                                                             tgt_neighbors_node[k], tgt_neighbors_ts[k]):
            #         # src_key, tgt_key = makekey(batch_idx, src_node, src_ts), makekey(batch_idx, tgt_node, tgt_ts)
            #         src_key_1, src_key_2, tgt_key = makekey(batch_idx, src_node_1, src_ts_1), makekey(batch_idx, src_node_2, src_ts_2), makekey(batch_idx, tgt_node, tgt_ts)
            #         # src_ts, tgt_ts = PositionEncoder.float2str(src_ts), PositionEncoder.float2str(tgt_ts)
            #         # src_key, tgt_key = (src_node, src_ts), (tgt_node, tgt_ts)

            #         # encode src node tree
            #         if src_key_1 not in nodetime2emb:
            #             nodetime2emb[src_key_1] = np.zeros((n_hop + 1, ), dtype=np.float32)
            #         if src_node_1 not in src_seen_nodes2label_1:
            #             new_src_node_label = k
            #             src_seen_nodes2label_1[src_key_1] = k
            #         else:
            #             new_src_node_label = src_seen_nodes2label_1[src_node_1]
            #         nodetime2emb[src_key_1][new_src_node_label] = 1

            #         # encode src node tree
            #         if src_key_2 not in nodetime2emb:
            #             nodetime2emb[src_key_2] = np.zeros((n_hop + 1, ), dtype=np.float32)
            #         if src_node_2 not in src_seen_nodes2label_2:
            #             new_src_node_label = k
            #             src_seen_nodes2label_2[src_key_2] = k
            #         else:
            #             new_src_node_label = src_seen_nodes2label_2[src_node_2]
            #         nodetime2emb[src_key_2][new_src_node_label] = 1

            #         # encode tgt node tree
            #         if tgt_key not in nodetime2emb:
            #             nodetime2emb[tgt_key] = np.zeros((n_hop + 1, ), dtype=np.float32)
            #         if tgt_node not in tgt_seen_nodes2label:
            #             new_tgt_node_label = k
            #             tgt_seen_nodes2label[tgt_node] = k
            #         else:
            #             new_tgt_node_label = tgt_seen_nodes2label[tgt_node]
            #         nodetime2emb[tgt_key][new_tgt_node_label] = 1
            # null_key = makekey(batch_idx, 0, 0.0)
            # nodetime2emb[null_key] = np.zeros((n_hop + 1, ), dtype=np.float32)
            # ======================================

            # nodetime2emb[(0, PositionEncoder.float2str(0.0))] = np.zeros((n_hop + 1, ), dtype=np.float32)
        # for key, value in nodetime2emb.items():
        #     nodetime2emb[key] = torch.tensor(value)
        return nodetime2emb

    def anonymize(self, node_records):
        if self.enc == 'saw':
            new_node_records = anonymize(node_records, *node_records.shape)
        else:
            new_node_records = node_records
        return new_node_records

    def forward(self, node_record, t_record, test, interpretation):
        '''
        accept two numpy arrays each of shape [batch, k-hop-support-number], corresponding to node indices and timestamps respectively
        return Torch.tensor: position features of shape [batch, k-hop-support-number, position_dim]
        return Torch.tensor: position features of shape [batch, k-hop-support-number, position_dim]
        '''
        # encodings = []
        device = next(self.projection.parameters()).device
        # float2str = PositionEncoder.float2str
        batch = node_record.shape[0]
        if self.enc != 'saw':
            batched_keys = make_batched_keys(node_record, t_record)
            unique, inv = np.unique(batched_keys, return_inverse=True)
            unordered_encodings = np.array([self.nodetime2emb_maps[key] for key in unique])
            encodings = unordered_encodings[inv.reshape(batch, -1), :]
            encodings = torch.tensor(encodings).to(device)
            walk_encodings = encodings.view(encodings.shape[0], -1, self.num_layers+1, *encodings.shape[2:])
            # walk_encodings = encodings.view(encodings.shape[0], -1, encodings.shape[-1], *encodings.shape[2:])
            """
            This is walk encoding bucket part.
            For interpretation, we can build a bucket for each pattern and put all the encodings into the bucket.
            And then we return the bucket.
            1) The bucket should be global for each batch, therefore the bucket is a self. in positionEncoder
            2) There are some symmetrical property when we have three nodes, therefore compare the first two nodes.
            First make it easy, and ignore the symmetric problems. #TODO: make the symmetric part
            walk_encodings: batch, walk_size, walk_length, (u,v,w), position_encoding_dim(3 for spd, walk_length for lp)
            """
            walk_pattern = None
            if (test) and (interpretation):
                walk_pattern = np.zeros([walk_encodings.shape[0], walk_encodings.shape[1]], dtype=np.int32)
                walk_encodings_cpu = walk_encodings.cpu().numpy()
                for idx_batch, batch in enumerate(walk_encodings_cpu):
                    for idx_walk, walk in enumerate(batch):
                        # print(walk)
                        walk = np.array2string(walk)
                        if walk not in self.pattern:
                            self.pattern[walk] = self.pattern_num
                            self.pattern_num += 1
                        walk_pattern[idx_batch, idx_walk] = self.pattern[walk]
        else:
            one_hot_base = np.eye(self.num_layers+1)
            encodings = one_hot_base[node_record]
            encodings = torch.tensor(encodings).to(device)
            walk_encodings = encodings.view(batch, -1, self.num_layers+1, self.num_layers+1)
        #         # for batch_idx, (n_l, ts_l) in enumerate(zip(node_record, t_record)):
        #     # encoding = [self.nodetime2emb_maps[batch_idx][(n, float2str(ts))] for n, ts in zip(n_l, ts_l)]
        #     # encodings.append(torch.stack(encoding))  # shape [support_n, 2] / [support_n, 2, num_layers+1]
        #     lookup_func = np.vectorize(self.nodetime2emb_maps[batch_idx].get)
        #     encodings = lookup_func(np.array(zip(node_record, [float2str(ts) for ts in t_record])))
        # encodings = torch.stack(encodings).to(device)  # shape [B, support_n, 2] / [B, support_n, 2, num_layers+1]
        # common_nodes = (((encodings.sum(-1) > 0).sum(-1) == 2).sum().float() / (encodings.shape[0] * encodings.shape[1])).item()
        common_nodes = 0
        encodings = self.get_trainable_encodings(encodings)
        return encodings, common_nodes, walk_encodings, walk_pattern

    @staticmethod
    def collect_pos_mapping_ptree_sample_mp(args):
        # src_idx_l, tgt_idx_l, cut_time_l, subgraph_src, subgraph_tgt, row, enc = args
        src_idx_l_1, src_idx_l_2, tgt_idx_l, cut_time_l, subgraph_src_1, subgraph_src_2, subgraph_tgt, row, enc = args
        subgraph_src_node_1, _, subgraph_src_ts_1 = subgraph_src_1  # only use node index and timestamp to identify a node in temporal graph
        subgraph_src_node_2, _, subgraph_src_ts_2 = subgraph_src_2
        subgraph_tgt_node, _, subgraph_tgt_ts = subgraph_tgt
        
        src_1 = src_idx_l_1[row]
        src_2 = src_idx_l_2[row]
        tgt = tgt_idx_l[row]
        cut_time = cut_time_l[row]
        src_neighbors_node_1 = [k_hop_neighbors[row] for k_hop_neighbors in subgraph_src_node_1]
        src_neighbors_ts_1 = [k_hop_neighbors[row] for k_hop_neighbors in subgraph_src_ts_1]
        src_neighbors_node_2 = [k_hop_neighbors[row] for k_hop_neighbors in subgraph_src_node_2]
        src_neighbors_ts_2 = [k_hop_neighbors[row] for k_hop_neighbors in subgraph_src_ts_2]
        tgt_neighbors_node = [k_hop_neighbors[row] for k_hop_neighbors in subgraph_tgt_node]
        tgt_neighbors_ts = [k_hop_neighbors[row] for k_hop_neighbors in subgraph_tgt_ts]
        
        # subgraph_src_node, _, subgraph_src_ts = subgraph_src  # only use node index and timestamp to identify a node in temporal graph
        # subgraph_tgt_node, _, subgraph_tgt_ts = subgraph_tgt
        # src = src_idx_l[row]
        # tgt = tgt_idx_l[row]
        # cut_time = cut_time_l[row]
        # src_neighbors_node = [k_hop_neighbors[row] for k_hop_neighbors in subgraph_src_node]
        # src_neighbors_ts = [k_hop_neighbors[row] for k_hop_neighbors in subgraph_src_ts]
        # tgt_neighbors_node = [k_hop_neighbors[row] for k_hop_neighbors in subgraph_tgt_node]
        # tgt_neighbors_ts = [k_hop_neighbors[row] for k_hop_neighbors in subgraph_tgt_ts]
        nodetime2emb_map = PositionEncoder.collect_pos_mapping_ptree_sample(src_1, src_2, tgt, cut_time,
                                                                            src_neighbors_node_1, src_neighbors_ts_1,
                                                                            src_neighbors_node_2, src_neighbors_ts_2,
                                                                            tgt_neighbors_node, tgt_neighbors_ts, enc=enc)
        # nodetime2emb_map = PositionEncoder.collect_pos_mapping_ptree_sample(src, tgt, cut_time,
        #                                                                     src_neighbors_node, src_neighbors_ts,
        #                                                                     tgt_neighbors_node, tgt_neighbors_ts, enc=enc)
        return nodetime2emb_map

    def get_trainable_encodings(self, encodings):
        '''
        Args:
            encodings: a device tensor of shape [batch, support_n, 2] / [batch, support_n, 2, L+1]
        Returns:  a device tensor of shape [batch, pos_dim]
        '''
        if self.enc == 'spd':
            encodings[encodings > (self.num_layers+0.5)] = self.num_layers + 1
            """
            previous
            """
            # # print('encoding size  ', encodings.size())
            # sum_src_1_2 = self.trainable_embedding_src(encodings[:,:,0].long()) + self.trainable_embedding_src(encodings[:,:,1].long())#.sum(-2)
            # tgt = self.trainable_embedding_tgt(encodings[:,:,2].long())# .sum(-2)
            # # print(sum_src_1_2.size(), tgt.size())
            # # print('encoding size  ', sum_src_1_2.size(), tgt.size(), torch.cat([sum_src_1_2, tgt], 2).size())
            # encodings = self.trainable_embedding_concat(torch.cat([sum_src_1_2, tgt], 2))# .squeeze()
            # # print('encoding size  ', encodings.size())
            # # encodings = self.trainable_embedding(encodings.long())  # now shape [batch, support_n, 2, pos_dim]
            # # encodings = encodings.sum(dim=-2)  # now shape [batch, support_n, pos_dim]

            """
            final
            """
            
            u = self.trainable_embedding_distance(encodings[:,:,0].long()).float()
            v = self.trainable_embedding_distance(encodings[:,:,1].long()).float()
            w = self.trainable_embedding_distance(encodings[:,:,2].long()).float()
            
            # print(u.size())
            # MLP
            d_uw = self.trainable_embedding_distance_MLP(torch.cat([u,w], 2))
            d_vw = self.trainable_embedding_distance_MLP(torch.cat([v,w], 2))
            encodings = self.trainable_embedding_distance_agg(torch.cat([d_uw+d_vw, torch.abs(d_uw-d_vw)], 2))

            # u+v; w
            # encodings = self.trainable_embedding_distance_agg(torch.cat([u+v, w], 2))
        elif self.enc == 'lp':
            # encodings = self.trainable_embedding(encodings.float())   # now shape [batch, support_n, 2, pos_dim]
            # encodings = encodings.sum(dim=-2)  # now shape [batch, support_n, pos_dim]
            # encodings = encodings.view()  # now shape [batch, support_n, pos_dim]
            # feature for three nodes: src_1 src_2 tgt
            # concat(src_1 + src_2, tgt)
            
            # # MLP((u + v) ; w)
            # encodings = encodings.float()
            # sum_src_1_2 = self.trainable_embedding_src(encodings[:,:,0,:] + encodings[:,:,1,:])
            # tgt = self.trainable_embedding_tgt(encodings[:,:,2,:])
            # encodings = self.trainable_embedding_concat(torch.cat([sum_src_1_2, tgt], 2))# .squeeze()

            # # # MLP(u,v,w) + MLP(v,u,w)
            # encodings = encodings.float()
            # # print(encodings[:,:,0,:].size())
            # u = encodings[:,:,0,:]
            # v = encodings[:,:,1,:]
            # w = encodings[:,:,2,:]
            # # u = self.trainable_embedding_src(encodings[:,:,0,:])
            # # v = self.trainable_embedding_src_2(encodings[:,:,1,:])
            # # w = self.trainable_embedding_tgt(encodings[:,:,2,:])
            # encodings_1 = self.trainable_embedding_uvw(torch.cat([u, v, w], 2))
            # encodings_2 = self.trainable_embedding_uvw(torch.cat([v, u, w], 2))
            # # encodings_1 = self.trainable_embedding_uvw(torch.cat([u, v, w], 2))
            # # encodings_2 = self.trainable_embedding_uvw(torch.cat([v, u, w], 2))
            # encodings = self.trainable_embedding_uvw_MLP(encodings_1 + encodings_2)

            encodings = encodings.float()
            u = encodings[:,:,0,:]
            v = encodings[:,:,1,:]
            w = encodings[:,:,2,:]
            d_uw = self.trainable_embedding_distance(torch.cat([u,w], 2))
            d_vw = self.trainable_embedding_distance(torch.cat([v,w], 2))
            encodings = self.trainable_embedding_distance_agg(torch.cat([d_uw+d_vw, torch.abs(d_uw-d_vw)], 2))
        else:
            assert(self.enc == 'saw')
            encodings = self.trainable_embedding(encodings.float())  # now shape [batch, support_n, pos_dim]
        return encodings

    # @staticmethod
    # @jit(nopython=True)
    # def float2str(ts):
    #     return str(int(round(ts)))
    #     # return str(round(ts, PRECISION))
    #
    # @staticmethod
    # @jit(nopython=True)
    # def nodets2key(batch: int, node: int, ts: float):
    #     key = '-'.join([str(batch), str(node), PositionEncoder.float2str(ts)])
    #     return key
    #
    # @staticmethod
    # @jit(nopython=True)
    # def make_batched_keys(node_record, t_record):
    #     batch, support = node_record.shape
    #     batch_matrix = np.tile(np.expand_dims(np.arange(batch), 1), (1, support))
    #     batched_keys = np.array([PositionEncoder.nodets2key(b, n, t) for b, n, t in zip(batch_matrix.ravel(), node_record.ravel(), t_record.ravel())]).reshape(batch, support)
    #     return batched_keys


class RandomWalkAttention(nn.Module):
    '''
    RandomWalkAttention have two modules: lstm + tranformer-self-attention
    '''
    def __init__(self, feat_dim, pos_dim, model_dim, out_dim, logger, walk_pool='attn', mutual=False, n_head=8, dropout_p=0.1, walk_linear_out=False):
        '''
        masked flags whether or not use only valid temporal walks instead of full walks including null nodes
        '''
        super(RandomWalkAttention, self).__init__()
        self.feat_dim = feat_dim
        self.pos_dim = pos_dim
        self.model_dim = model_dim
        self.attn_dim = self.model_dim//2  # half the model dim to save computation cost for attention
        self.out_dim = out_dim
        self.walk_pool = walk_pool
        self.mutual = mutual
        self.n_head = n_head
        self.dropout_p = dropout_p
        self.logger = logger

        self.feature_encoder = FeatureEncoder(self.feat_dim, self.model_dim, self.dropout_p)  # encode all types of features along each temporal walk
        # triangle
        self.position_encoder = FeatureEncoder(self.pos_dim, self.pos_dim, self.dropout_p)  # encode specifially spatio-temporal features along each temporal walk
        self.projector = nn.Sequential(nn.Linear(self.feature_encoder.model_dim+self.position_encoder.model_dim, self.attn_dim),  # notice that self.feature_encoder.model_dim may not be exactly self.model_dim is its not even number because of the usage of bi-lstm
                                       nn.ReLU(), nn.Dropout(self.dropout_p))  # TODO: whether to add #[, nn.Dropout())]?
        self.self_attention = TransformerEncoderLayer(d_model=self.attn_dim, nhead=self.n_head,
                                                      dim_feedforward=4*self.attn_dim, dropout=self.dropout_p,
                                                      activation='relu')
        if self.mutual:
            self.mutual_attention_src2tgt = TransformerDecoderLayer(d_model=self.attn_dim, nhead=self.n_head,
                                                                    dim_feedforward=4*self.model_dim,
                                                                    dropout=self.dropout_p,
                                                                    activation='relu')
            self.mutual_attention_tgt2src = TransformerDecoderLayer(d_model=self.attn_dim, nhead=self.n_head,
                                                                    dim_feedforward=4*self.model_dim,
                                                                    dropout=self.dropout_p,
                                                                    activation='relu')
        self.pooler = SetPooler(n_features=self.attn_dim, out_features=self.out_dim, dropout_p=self.dropout_p, walk_linear_out=walk_linear_out)
        self.logger.info('bi-lstm actual encoding dim: {} + {}, attention dim: {}, attention heads: {}'.format(self.feature_encoder.model_dim, self.position_encoder.model_dim, self.attn_dim, self.n_head))

    def forward_one_node(self, hidden_embeddings, time_features, edge_features, position_features, masks=None):
        '''
        Input shape [batch, n_walk, len_walk, *_dim]
        Return shape [batch, n_walk, feat_dim]
        '''
        combined_features = self.aggregate(hidden_embeddings, time_features, edge_features, position_features)
        combined_features = self.feature_encoder(combined_features, masks)
        if self.pos_dim > 0:
            position_features = self.position_encoder(position_features, masks)
            combined_features = torch.cat([combined_features, position_features], dim=-1)
        X = self.projector(combined_features)
        """
        Here X is [batch, n_walk, feat_dim]
        For interpretation, we need to return X before pooling
        """
        if self.walk_pool == 'sum':
            X = self.pooler(X, agg='mean')  # we are actually doing mean pooling since sum has numerical issues
            return X
        else:
            X = self.self_attention(X)
            if not self.mutual:
                X = self.pooler(X, agg='mean') # we are actually doing mean pooling since sum has numerical issues
            return X

    def mutual_query(self, src_embed_1, src_embed_2, tgt_embed):
        '''
        Input shape: [batch, n_walk, feat_dim]
        '''
        
        src_emb = self.mutual_attention_src2tgt(src_embed, tgt_embed)
        tgt_emb = self.mutual_attention_tgt2src(tgt_embed, src_embed)
        src_emb = self.pooler(src_emb)
        tgt_emb = self.pooler(tgt_emb)
        return src_emb, tgt_emb

    def aggregate(self, hidden_embeddings, time_features, edge_features, position_features):
        batch, n_walk, len_walk, _ = hidden_embeddings.shape
        device = hidden_embeddings.device
        if position_features is None:
            assert(self.pos_dim == 0)
            combined_features = torch.cat([hidden_embeddings, time_features, edge_features], dim=-1)
        else:
            combined_features = torch.cat([hidden_embeddings, time_features, edge_features, position_features], dim=-1)
        combined_features = combined_features.to(device)
        assert(combined_features.size(-1) == self.feat_dim)
        return combined_features


class FeatureEncoder(nn.Module):
    def __init__(self, in_features, hidden_features, dropout_p=0.1):
        super(FeatureEncoder, self).__init__()
        self.hidden_features_one_direction = hidden_features//2
        self.model_dim = self.hidden_features_one_direction * 2  # notice that we are using bi-lstm
        if self.model_dim == 0:  # meaning that this encoder will be use less
            return
        self.lstm_encoder = nn.LSTM(input_size=in_features, hidden_size=self.hidden_features_one_direction, batch_first=True, bidirectional=True)
        self.dropout = nn.Dropout(dropout_p)

    def forward(self, X, mask=None):
        batch, n_walk, len_walk, feat_dim = X.shape
        X = X.view(batch*n_walk, len_walk, feat_dim)
        if mask is not None:
            lengths = mask.view(batch*n_walk)
            X = pack_padded_sequence(X, lengths, batch_first=True, enforce_sorted=False)
        encoded_features = self.lstm_encoder(X)[0]
        if mask is not None:
            encoded_features, lengths = pad_packed_sequence(encoded_features, batch_first=True)
        encoded_features = encoded_features.select(dim=1, index=-1).view(batch, n_walk, self.model_dim)
        encoded_features = self.dropout(encoded_features)
        return encoded_features

# Use different LSTM for each step
class FeatureEncoder_DifferentRNN(nn.Module):
    def __init__(self, in_features, hidden_features, length, dropout_p=0.1):
        super(FeatureEncoder_DifferentRNN, self).__init__()
        """
        length : int, m length = m+1 nodes = m+1 LSTM
        """
        self.hidden_features_one_direction = hidden_features//2
        self.model_dim = self.hidden_features_one_direction * 2  # notice that we are using bi-lstm
        if self.model_dim == 0:  # meaning that this encoder will be use less
            return
        self.lstm_encoder_1 = nn.LSTM(input_size=in_features, hidden_size=self.hidden_features_one_direction, batch_first=True, bidirectional=False)
        self.lstm_encoder = []
        self.lstm_encoder.append(self.lstm_encoder_1)
        for i in range(length - 2):
            self.lstm_encoder.append(nn.LSTM(input_size=self.hidden_features_one_direction, hidden_size=self.hidden_features_one_direction, batch_first=True, bidirectional=False))
        self.lstm_encoder.append(nn.LSTM(input_size=self.hidden_features_one_direction, hidden_size=self.hidden_features_one_direction, batch_first=True, bidirectional=False))
        self.dropout = nn.Dropout(dropout_p)

    def forward(self, X, mask=None):
        batch, n_walk, len_walk, feat_dim = X.shape
        X = X.view(batch*n_walk, len_walk, feat_dim)
        if mask is not None:
            lengths = mask.view(batch*n_walk)
            X = pack_padded_sequence(X, lengths, batch_first=True, enforce_sorted=False)
        for i in self.lstm_encoder:
            X = i(X)[0]
        encoded_features = X
        if mask is not None:
            encoded_features, lengths = pad_packed_sequence(encoded_features, batch_first=True)
        encoded_features = encoded_features.select(dim=1, index=-1).view(batch, n_walk, self.model_dim)
        encoded_features = self.dropout(encoded_features)
        return encoded_features



class SetPooler(nn.Module):
    """
    Implement similar ideas to the Deep Set
    """
    def __init__(self, n_features, out_features, dropout_p=0.1, walk_linear_out=False):
        super(SetPooler, self).__init__()
        self.mean_proj = nn.Linear(n_features, n_features)
        self.max_proj = nn.Linear(n_features, n_features)
        self.attn_weight_mat = nn.Parameter(torch.zeros((2, n_features, n_features)), requires_grad=True)
        nn.init.xavier_uniform_(self.attn_weight_mat.data[0])
        nn.init.xavier_uniform_(self.attn_weight_mat.data[1])
        self.dropout = nn.Dropout(dropout_p)
        self.out_proj = nn.Sequential(nn.Linear(n_features, out_features), nn.ReLU(), self.dropout)
        self.walk_linear_out = walk_linear_out

    def forward(self, X, agg='sum'):
        if self.walk_linear_out:  # for explainability, postpone summation to merger function
            return self.out_proj(X)
        if agg == 'sum':
            return self.out_proj(X.sum(dim=-2))
        else:
            assert(agg == 'mean')
            return self.out_proj(X.mean(dim=-2))
        # code block below is trying to combine mean and max pooling with attention but is currently problematic
        # batch, n_walk, feat_dim = X.shape
        # X_mean = self.mean_proj(X).mean(dim=-2)
        # X_max = self.max_proj(X).max(dim=-2)[0]
        # X_mean_max = torch.stack([X_mean, X_max])  # shape: [2, batch, feat_dim]
        # weights = torch.bmm(torch.bmm(X_mean_max,
        #                     self.attn_weight_mat).view(2*batch, 1, feat_dim),
        #           X_mean_max.view(2*batch, feat_dim, 1)).view(2, batch, 1)
        # weights = torch.softmax(weights, dim=0)
        # X_agg = (weights * X_mean_max).sum(dim=0)
        # X_agg = self.dropout(self.out_proj(self.dropout(X_agg)))
        # return X_agg