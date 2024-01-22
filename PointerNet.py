import torch
from torch_geometric.nn import GATv2Conv


class GATEncoder(torch.nn.Module):
    """
    Graph encoder based on two layers of GAT.
    """
    def __init__(self,
                 input_size: int,
                 hidden_size: int = 64,
                 embed_size: int = 128,
                 n_heads: int = 3,
                 leaky_slope: float = 0.15):
        """
        Constructor.

        Args:
            input_size: Number of features in each node/operation.
            hidden_size: Hidden units in the first layer.
            embed_size: Number of dimensions in the output embeddings.
            n_heads: Number of heads in GAT.
            leaky_slope: Slope in the leaky ReLU.
        """
        super(GATEncoder, self).__init__()
        self.embedding1 = GATv2Conv(in_channels=input_size,
                                    out_channels=hidden_size,
                                    dropout=0,
                                    heads=n_heads,
                                    concat=True,
                                    add_self_loops=False,
                                    negative_slope=leaky_slope)
        self.embedding2 = GATv2Conv(in_channels=hidden_size * n_heads + input_size,
                                    out_channels=embed_size,
                                    dropout=0,
                                    heads=n_heads,
                                    concat=False,
                                    add_self_loops=False,
                                    negative_slope=leaky_slope)
        #
        self.out_size = input_size + embed_size

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor):
        """
        Forward pass.

        Args:
            x: The features for each node. Shape: (num nodes, input_size).
            edge_index: edge index of the graph. Shape: (2, num edges).
        Return:
            The node embeddings.
        """
        #
        h1 = torch.relu(self.embedding1(x, edge_index))
        h = torch.cat([x, h1], dim=-1)
        #
        h2 = torch.relu(self.embedding2(h, edge_index))

        return torch.cat([x, h2], dim=-1)


class MHADecoder(torch.nn.Module):

    def __init__(self,
                 encoder_size: int,
                 context_size: int,
                 hidden_size: int = 64,
                 mem_size: int = 128,
                 clf_size: int = 128,
                 leaky_slope: float = 0.15,
                 n_heads: int = 3):
        """
        Constructor.

        Args:
            encoder_size: Number of features in the output of the encoder.
            context_size: Number of features in the state.
            hidden_size: Number of hidden dimensions in the memory network.
            mem_size: Number of output dimension from the memory network
            clf_size: Number of hidden dimension in the classifier.
            leaky_slope: Slope in the leaky ReLU.
        """
        super(MHADecoder, self).__init__()
        # Memory net
        self.linear1 = torch.nn.Linear(context_size, hidden_size * n_heads)
        self.linear2 = torch.nn.Linear(hidden_size * n_heads, mem_size)
        self.self_attn = torch.nn.MultiheadAttention(
            hidden_size * n_heads,
            num_heads=n_heads,
            dropout=0.,
            batch_first=True
        )

        # Classifier net
        self.act = torch.nn.LeakyReLU(leaky_slope)
        self.linear3 = torch.nn.Linear(encoder_size + mem_size, clf_size)
        self.linear4 = torch.nn.Linear(clf_size, 1)

    def forward(self, embed_x: torch.Tensor, state: torch.Tensor):
        """
        Forward pass.

        Args:
            embed_x: (batch size, num jobs, num features)
            state: (batch size, num jobs, num features)
        Return:
            Probability of each job.
        """
        # Make job states
        x1 = self.linear1(state)
        x2 = x1 + self.self_attn(x1, x1, x1)[0]
        x2 = torch.relu(self.linear2(x2))

        # Make probability of selecting each job
        xx = torch.concat([embed_x, x2], dim=-1)
        xx = self.act(self.linear3(xx))
        return self.linear4(xx).squeeze(-1)
