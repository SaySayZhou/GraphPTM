import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
from torch_scatter import scatter
from torch_geometric.nn import TransformerConv, GCNConv, GATConv


class Attention(nn.Module):
    def __init__(self, in_features, hidden_dim):
        super(Attention, self).__init__()
        self.query = nn.Linear(in_features, hidden_dim)
        self.key = nn.Linear(in_features, hidden_dim)
        self.value = nn.Linear(in_features, hidden_dim)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        query = self.query(x)
        key = self.key(x)
        value = self.value(x)
        attention_scores = torch.matmul(query, key.transpose(-2, -1)) / torch.sqrt(
            torch.tensor(query.shape[-1]).float())
        attention_weights = self.softmax(attention_scores)
        attention_output = torch.matmul(attention_weights, value)
        return attention_output


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super(MultiHeadAttention, self).__init__()
        self.num_heads = num_heads
        self.d_model = d_model

        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"

        self.d_k = d_model // num_heads

        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)

    def scaled_dot_product_attention(self, Q, K, V, mask=None):
        attn_scores = torch.matmul(Q, K.transpose(-2, -1)) / np.sqrt(self.d_k)
        if mask is not None:
            attn_scores = attn_scores.masked_fill(mask == 0, -1e9)
        attn_probs = torch.softmax(attn_scores, dim=-1)
        output = torch.matmul(attn_probs, V)
        return output

    def split_heads(self, x):
        batch_size, seq_length, d_model = x.size()
        return x.view(batch_size, seq_length, self.num_heads, self.d_k).transpose(1, 2)

    def combine_heads(self, x):
        batch_size, _, seq_length, d_k = x.size()
        return x.transpose(1, 2).contiguous().view(batch_size, seq_length, self.d_model)

    def forward(self, Q, K, V, mask=None):
        Q = self.split_heads(self.W_q(Q))
        K = self.split_heads(self.W_k(K))
        V = self.split_heads(self.W_v(V))

        attn_output = self.scaled_dot_product_attention(Q, K, V, mask)
        output = self.W_o(self.combine_heads(attn_output))
        return output


class GNNTrans(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        self.loss_func = nn.BCELoss()
        self.ln1 = nn.LayerNorm(hidden_dim)

        self.gcn_convs = torch.nn.ModuleList(
            [GCNConv(in_channels=input_dim, out_channels=hidden_dim, heads=1)] +
            [GCNConv(in_channels=hidden_dim, out_channels=hidden_dim, heads=1)
             for _ in range(num_layers - 1)]
        )
        self.atten_mlp = Attention(hidden_dim, hidden_dim)

        self.mlp = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(hidden_dim, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.3),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )


    def get_conv_result_GCN(self, x, edge_index):
        """
        GCNConv feature
        """
        for i in range(self.num_layers):
            x = self.gcn_convs[i](x=x, edge_index=edge_index)
            x = self.ln1(x)  # LayerNormal
            # x = self.atten_mlp(x=x)
            x = F.relu(x, inplace=True)
        return x

    def forward(self, data):
        # print('data:', data)
        """
        data:
            DataBatch(x=[2624, 1], y=[64], 
            edge_index1=[2, 107584], edge_index2=[2, 10176], 
            emb=[2624, 1024], 
            seq=[64], uniprot=[64], unique_id=[64], batch=[2624], ptr=[65])
        """
        x, edge_index, batch = data.emb, data.edge_index1, data.batch
        idx = (data.ptr + int(len(data.seq[0]) / 2))[:-1]

        x = self.get_conv_result_GCN(x, edge_index)
        x = self.atten_mlp(x=x)

        x = x[idx]

        # print("before FCN x'shape is : ",x.shape)
        out = self.mlp(x)
        return out

    def loss(self, pred, label):
        pred, label = pred.reshape(-1), label.reshape(-1)
        return self.loss_func(pred, label)


if __name__ == '__main__':
    from torch_geometric.loader import DataLoader

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    train_ls, test_ls = torch.load('./data/output/human/N_Gly/input/seq_data.pt')

    train_loader = DataLoader(train_ls, batch_size=64)
    test_loader = DataLoader(test_ls, batch_size=64)

    print('-----len train_loader.dataset:', len(train_loader.dataset))
    print('-----len test_loader.dataset:', len(test_loader.dataset))

    model = GNNTrans(input_dim=1024, hidden_dim=256, num_layers=3).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=3e-5)
    train_epoch_loss = 0.0
    for batch_idx, batch in enumerate(train_loader):
        batch = batch.to(device)
        output = model(batch)
        print(output)
        print(output.shape)
        break
    # train_epoch_loss /= len(train_loader)
