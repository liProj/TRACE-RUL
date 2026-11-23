"""Faithful re-implementation of AIDGN (Model/AIDGN.py of the AIDGN-RUL release).

Parameter names are identical so the released checkpoints load with strict=True.
Quirks of the release are kept on purpose (softmax instead of sigmoid in Eq. 5,
`view` instead of transpose for the second pathway, abs() on the output).
"""
import torch
from torch import nn
import torch.nn.functional as F


class nconv(nn.Module):
    def forward(self, x, A):
        return torch.einsum('bfnm,bmnv->bfvm', (x, A)).contiguous()


class linear(nn.Module):
    def __init__(self, c_in, c_out):
        super().__init__()
        self.mlp = nn.Conv2d(c_in, c_out, kernel_size=(1, 1), bias=True)

    def forward(self, x):
        return self.mlp(x)


class gcn(nn.Module):
    def __init__(self, c_in, c_out, dropout, support_len=1, order=2):
        super().__init__()
        self.nconv = nconv()
        self.mlp = linear((order * support_len + 1) * c_in, c_out)
        self.order = order

    def forward(self, x, bases):
        out = [x]
        for b in bases:
            x1 = self.nconv(x, b)
            out.append(x1)
            for _ in range(2, self.order + 1):
                x1 = self.nconv(x1, b)
                out.append(x1)
        return self.mlp(torch.cat(out, dim=1))


class AIDGN(nn.Module):
    def __init__(self, feature_num=14, sequence_len=50, hidden_dim=64, AATE_dim=10, dropout=0.1,
                 nlayer=1, hop=1, use_age=True):
        super().__init__()
        self.C, self.M, self.n_layer, self.AATE_dim = feature_num, 1, nlayer, AATE_dim
        self.use_age = use_age
        fc_layer_dim, feature_fc_layer_dim = hidden_dim * 4, feature_num * 4
        self.dropout = nn.Dropout(dropout)
        self.ll1, self.ll2 = nn.ModuleList(), nn.ModuleList()
        self.m_gate1, self.m_gate2 = nn.ModuleList(), nn.ModuleList()
        for _ in range(nlayer):
            self.ll1.append(nn.Linear(sequence_len, AATE_dim))
            self.ll2.append(nn.Linear(sequence_len, AATE_dim))
            self.m_gate1.append(nn.Sequential(nn.Linear(sequence_len + AATE_dim, 1), nn.ReLU()))
            self.m_gate2.append(nn.Sequential(nn.Linear(sequence_len + AATE_dim, 1), nn.ReLU()))
        self.gconv = nn.ModuleList([gcn(sequence_len, hidden_dim, self.dropout, support_len=1, order=hop)
                                    for _ in range(nlayer)])
        self.project = nn.Linear(sequence_len, AATE_dim)
        self.temporal_agg = nn.Sequential(nn.Linear(hidden_dim, hidden_dim))
        self.decoder = nn.Sequential(nn.Linear(hidden_dim, fc_layer_dim), nn.ReLU(), nn.Linear(fc_layer_dim, 1))
        self.channel_decoder = nn.Sequential(nn.Linear(feature_num, feature_fc_layer_dim), nn.ReLU(),
                                             nn.Linear(feature_fc_layer_dim, 1))

    def forward(self, x, occ):
        x = x.permute(0, 2, 1)
        B, C, N = x.shape
        x = x.unsqueeze(2).expand(-1, -1, self.M, -1)
        occ = occ.transpose(1, 2)
        if not self.use_age:
            occ = torch.zeros_like(occ)
        occ = self.project(occ)
        occ = occ.repeat(self.M, 1, C, 1)
        for layer in range(self.n_layer):
            if layer > 0:
                x_last = x.clone()
            aate = occ.view(B, self.M, C, self.AATE_dim)
            aate_t = occ.view(B, self.M, self.AATE_dim, C)
            m1 = self.m_gate1[layer](torch.cat([x, aate.permute(0, 2, 1, 3)], dim=-1))
            m2 = self.m_gate2[layer](torch.cat([x, aate_t.permute(0, 3, 1, 2)], dim=-1))
            e1 = self.dropout(F.softmax(F.relu(m1 * self.ll1[layer](x)), dim=-1))
            e2 = self.dropout(F.softmax(F.relu(m2 * self.ll2[layer](x)), dim=-1))
            e1 = aate + e1.permute(0, 2, 1, 3)
            e2 = aate_t + e2.permute(0, 2, 3, 1)
            adp = F.softmax(F.relu(torch.matmul(e1, e2)), dim=-1)
            x = self.gconv[layer](x.permute(0, 3, 1, 2), [adp])
            x = F.relu(self.dropout(x)).permute(0, 2, 3, 1)
            if layer > 0:
                x = x_last + x
        x = self.temporal_agg(x.mean(dim=2))
        out = self.decoder(self.dropout(x)).permute(0, 2, 1)
        out = self.channel_decoder(out).permute(0, 2, 1)
        return torch.abs(out.squeeze(-1))
