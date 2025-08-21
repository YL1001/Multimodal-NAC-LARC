import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATv2Conv
import math


class DeepGATBlock(nn.Module):
    def __init__(self, hidden_dim, heads=4, edge_dim=1, num_layers=1):
        super().__init__()
        self.layers = nn.ModuleList([
            GATv2Conv(
                hidden_dim,
                hidden_dim // heads,
                heads=heads,
                edge_dim=edge_dim,
                add_self_loops=False,
                dropout=0.1
            )
            for i in range(num_layers)
        ])
        self.norms = nn.ModuleList([nn.LayerNorm(hidden_dim) for _ in range(num_layers)])
        self.res_weights = nn.Parameter(torch.ones(num_layers))

    def forward(self, x, edge_index, edge_attr):
        identity = x
        for i, (gat, norm) in enumerate(zip(self.layers, self.norms)):
            x = gat(x, edge_index, edge_attr)
            x = norm(x + self.res_weights[i] * identity)
            x = F.gelu(x)
            identity = x
        return x


class MultiHeadCrossAttention(nn.Module):
    def __init__(self, hidden_dim, num_heads=4):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads

        self.q_proj = nn.Linear(hidden_dim, hidden_dim)
        self.k_proj = nn.Linear(hidden_dim, hidden_dim)
        self.v_proj = nn.Linear(hidden_dim, hidden_dim)
        self.out_proj = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, feat1, feat2):
        Q = self._reshape_to_heads(self.q_proj(feat1))
        K = self._reshape_to_heads(self.k_proj(feat2))
        V = self._reshape_to_heads(self.v_proj(feat2))

        attn = torch.einsum('nhd,nkd->nhk', Q, K) / math.sqrt(self.head_dim)
        attn = F.softmax(attn, dim=-1)
        out = torch.einsum('nhk,nkd->nhd', attn, V)
        out = self._reshape_from_heads(out)
        return self.out_proj(out)

    def _reshape_to_heads(self, x):
        return x.view(-1, self.num_heads, self.head_dim)

    def _reshape_from_heads(self, x):
        return x.contiguous().view(-1, self.num_heads * self.head_dim)


class BioGAT(nn.Module):
    def __init__(
        self,
        func_dim=10,
        expr_dim=1,
        hidden_dim=512,
        gat_heads=4,
        cross_heads=4,
        gat_layers=1,
        dropout=0.3,
        gene_n=1
    ):
        super().__init__()
        self._init_dynamic_params()

        self.func_encoder = nn.Sequential(
            nn.Linear(func_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.LayerNorm(hidden_dim)
        )
        self.expr_encoder = nn.Sequential(
            nn.Linear(expr_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.LayerNorm(hidden_dim)
        )

        self.func_gat1 = DeepGATBlock(hidden_dim, gat_heads, edge_dim=1, num_layers=gat_layers)
        self.expr_gat1 = DeepGATBlock(hidden_dim, gat_heads, edge_dim=1, num_layers=gat_layers)

        self.cross_att_func1 = MultiHeadCrossAttention(hidden_dim, cross_heads)
        self.cross_att_expr1 = MultiHeadCrossAttention(hidden_dim, cross_heads)

        self.feature_squeeze = nn.Sequential(
            nn.Linear(gene_n, gene_n // 2),
            nn.Tanh(),
            nn.Linear(gene_n // 2, 1, bias=False)
        )

        self.classifier = nn.Sequential(
            nn.Linear(2 * hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )

    def _init_dynamic_params(self):
        for name, param in self.named_parameters():
            if 'weight' in name:
                if 'gat' in name:
                    nn.init.xavier_normal_(param, gain=nn.init.calculate_gain('relu'))
                elif 'attention' in name:
                    nn.init.kaiming_uniform_(param, mode='fan_out', nonlinearity='sigmoid')

    def forward(self, data, plan):
        filtered_edge_index = data.edge_index
        filtered_edge_attr = data.edge_attr

        func_feat = self.func_encoder(data.gene_feature)
        expr_feat = self.expr_encoder(data.x)

        func_feat = self.func_gat1(func_feat, filtered_edge_index, filtered_edge_attr)
        expr_feat = self.expr_gat1(expr_feat, filtered_edge_index, filtered_edge_attr)

        func_feat = func_feat + self.cross_att_func1(func_feat, expr_feat)
        expr_feat = expr_feat + self.cross_att_expr1(expr_feat, func_feat)

        combined = torch.cat([func_feat, expr_feat], dim=1)

        batch_mask = data.batch
        unique_batches = torch.unique(batch_mask)
        grouped_features = []
        for batch_idx in unique_batches:
            group = combined[(batch_mask == batch_idx)]
            grouped_features.append(group)
        grouped_features = torch.stack(grouped_features, dim=0)
        reshaped_features = grouped_features.permute(0, 2, 1)

        graph_feat = self.feature_squeeze(reshaped_features).squeeze()

        return self.classifier(graph_feat)