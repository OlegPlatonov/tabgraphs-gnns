import torch
from torch import nn
from modules import (ResidualModuleWrapper, FeedForwardModule, GCNModule, SAGEModule, GATModule, GATSepModule,
                     TransformerAttentionModule, TransformerAttentionSepModule)
from plr_embeddings import PLREmbeddings


MODULES = {
    'ResNet': [FeedForwardModule],
    'GCN': [GCNModule],
    'SAGE': [SAGEModule],
    'GAT': [GATModule],
    'GAT-sep': [GATSepModule],
    'GT': [TransformerAttentionModule, FeedForwardModule],
    'GT-sep': [TransformerAttentionSepModule, FeedForwardModule]
}


NORMALIZATION = {
    'None': nn.Identity,
    'LayerNorm': nn.LayerNorm,
    'BatchNorm': nn.BatchNorm1d
}


class Model(nn.Module):
    def __init__(self, model_name, num_layers, input_dim, hidden_dim, output_dim, hidden_dim_multiplier, num_heads,
                 normalization, dropout, use_plr, num_numeric_inputs, plr_n_frequencies, plr_frequency_scale,
                 plr_d_embedding, use_plr_lite):
        super().__init__()

        normalization = NORMALIZATION[normalization]

        self.use_plr = use_plr
        if use_plr:
            self.plr_embeddings = PLREmbeddings(n_features=num_numeric_inputs, n_frequencies=plr_n_frequencies,
                                                frequency_scale=plr_frequency_scale, d_embedding=plr_d_embedding,
                                                lite=use_plr_lite)
            self.num_numeric_inputs = num_numeric_inputs
            input_dim = input_dim - num_numeric_inputs + num_numeric_inputs * plr_d_embedding

        self.input_linear = nn.Linear(in_features=input_dim, out_features=hidden_dim)
        self.dropout = nn.Dropout(p=dropout)
        self.act = nn.GELU()

        self.residual_modules = nn.ModuleList()
        for _ in range(num_layers):
            for module in MODULES[model_name]:
                residual_module = ResidualModuleWrapper(module=module,
                                                        normalization=normalization,
                                                        dim=hidden_dim,
                                                        hidden_dim_multiplier=hidden_dim_multiplier,
                                                        num_heads=num_heads,
                                                        dropout=dropout)

                self.residual_modules.append(residual_module)

        self.output_normalization = normalization(hidden_dim)
        self.output_linear = nn.Linear(in_features=hidden_dim, out_features=output_dim)

    def forward(self, graph, x):
        if self.use_plr:
            x_num = x[:, :self.num_numeric_inputs]
            x_num_embedded = self.plr_embeddings(x_num).flatten(start_dim=1)
            x = torch.cat([x_num_embedded, x[:, self.num_numeric_inputs:]], axis=1)

        x = self.input_linear(x)
        x = self.dropout(x)
        x = self.act(x)

        for residual_module in self.residual_modules:
            x = residual_module(graph, x)

        x = self.output_normalization(x)
        x = self.output_linear(x).squeeze(1)

        return x
