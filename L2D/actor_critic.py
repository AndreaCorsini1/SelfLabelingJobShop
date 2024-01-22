import torch
import torch.nn.functional as F
from torch import nn


class MLP(nn.Module):
    def __init__(self, num_layers, input_dim, hidden_dim, output_dim):
        """

        Args:
            num_layers: number of layers in the neural networks (EXCLUDING the
                input layer). If num_layers=1, this reduces to linear model.
            input_dim: dimensionality of input features
            hidden_dim: dimensionality of hidden units at ALL layers
            output_dim: number of classes for prediction
        """
        super(MLP, self).__init__()

        self.linear_or_not = True  # default is linear model
        self.num_layers = num_layers

        if num_layers < 1:
            raise ValueError("number of layers should be positive!")
        elif num_layers == 1:
            # Linear model
            self.linear = nn.Linear(input_dim, output_dim)
        else:
            # Multi-layer model
            self.linear_or_not = False
            self.linears = torch.nn.ModuleList()
            self.batch_norms = torch.nn.ModuleList()
            self.linears.append(nn.Linear(input_dim, hidden_dim))
            for layer in range(num_layers - 2):
                self.linears.append(nn.Linear(hidden_dim, hidden_dim))
            self.linears.append(nn.Linear(hidden_dim, output_dim))
            for layer in range(num_layers - 1):
                self.batch_norms.append(nn.BatchNorm1d((hidden_dim)))

    def forward(self, x):
        if self.linear_or_not:
            # If linear model
            return self.linear(x)
        else:
            # If MLP
            h = x
            for layer in range(self.num_layers - 1):
                h = F.relu(self.batch_norms[layer](self.linears[layer](h)))
            return self.linears[self.num_layers - 1](h)


class GraphCNN(nn.Module):
    def __init__(self,
                 num_layers,
                 num_mlp_layers,
                 input_dim,
                 hidden_dim,
                 # final_dropout,
                 learn_eps,
                 neighbor_pooling_type,
                 device):
        """

        Args:
            num_layers: number of layers in the neural networks (INCLUDING the
                input layer).
            num_mlp_layers: number of layers in mlps (EXCLUDING the input layer)
            input_dim: dimensionality of input features
            hidden_dim: dimensionality of hidden units at ALL layers
            learn_eps: If True, learn epsilon to distinguish center nodes from
                neighboring nodes. If False, aggregate neighbors and center
                nodes altogether.
            neighbor_pooling_type: how to aggregate neighbors (mean, average, or max)
            device: which device to use
        """
        super(GraphCNN, self).__init__()

        self.device = device
        self.num_layers = num_layers
        self.neighbor_pooling_type = neighbor_pooling_type
        self.learn_eps = learn_eps
        # common out the eps if you do not need to use it, otherwise the it
        # will cause error "not in the computational graph"
        # if self.learn_eps:
        #     self.eps = nn.Parameter(torch.zeros(self.num_layers - 1))

        # List of MLPs
        self.mlps = torch.nn.ModuleList()

        # List of batchnorms applied to the output of MLP (input of the final
        # prediction linear layer)
        self.batch_norms = torch.nn.ModuleList()
        for layer in range(self.num_layers - 1):
            if layer == 0:
                self.mlps.append(
                    MLP(num_mlp_layers, input_dim, hidden_dim, hidden_dim))
            else:
                self.mlps.append(
                    MLP(num_mlp_layers, hidden_dim, hidden_dim, hidden_dim))

            self.batch_norms.append(nn.BatchNorm1d(hidden_dim))

    def next_layer_eps(self, h, layer, padded_neighbor_list=None,
                       Adj_block=None):
        # pooling neighboring nodes and center nodes separately by epsilon reweighting.

        if self.neighbor_pooling_type == "max":
            # If max pooling
            pooled = self.maxpool(h, padded_neighbor_list)
        else:
            # If sum or average pooling
            pooled = torch.mm(Adj_block, h)
            if self.neighbor_pooling_type == "average":
                # If average pooling
                degree = torch.mm(Adj_block,
                                  torch.ones((Adj_block.shape[0], 1)).to(
                                      self.device))
                pooled = pooled / degree

        # Reweights the center node representation when aggregating it with its neighbors
        pooled = pooled + (1 + self.eps[layer]) * h
        pooled_rep = self.mlps[layer](pooled)
        h = self.batch_norms[layer](pooled_rep)

        # non-linearity
        h = F.relu(h)
        return h

    def next_layer(self, h, layer, padded_neighbor_list=None, Adj_block=None):

        # pooling neighboring nodes and center nodes altogether
        if self.neighbor_pooling_type == "max":
            # If max pooling
            pooled = self.maxpool(h, padded_neighbor_list)
        else:
            # If sum or average pooling
            # print(Adj_block.dtype)
            # print(h.dtype)
            pooled = torch.mm(Adj_block, h)
            if self.neighbor_pooling_type == "average":
                # If average pooling
                degree = torch.mm(Adj_block,
                                  torch.ones((Adj_block.shape[0], 1)).to(
                                      self.device))
                pooled = pooled / degree
        # representation of neighboring and center nodes
        pooled_rep = self.mlps[layer](pooled)
        h = self.batch_norms[layer](pooled_rep)

        # non-linearity
        h = F.relu(h)
        return h

    def forward(self,
                x,
                graph_pool,
                padded_nei,
                adj):

        x_concat = x
        graph_pool = graph_pool

        if self.neighbor_pooling_type == "max":
            padded_neighbor_list = padded_nei
        else:
            Adj_block = adj

        # list of hidden representation at each layer (including input)
        h = x_concat

        for layer in range(self.num_layers - 1):
            if self.neighbor_pooling_type == "max" and self.learn_eps:
                h = self.next_layer_eps(h, layer,
                                        padded_neighbor_list=padded_neighbor_list)
            elif not self.neighbor_pooling_type == "max" and self.learn_eps:
                h = self.next_layer_eps(h, layer, Adj_block=Adj_block)
            elif self.neighbor_pooling_type == "max" and not self.learn_eps:
                h = self.next_layer(h, layer,
                                    padded_neighbor_list=padded_neighbor_list)
            elif not self.neighbor_pooling_type == "max" and not self.learn_eps:
                h = self.next_layer(h, layer, Adj_block=Adj_block)

        h_nodes = h.clone()
        # print(graph_pool.shape, h.shape)
        pooled_h = torch.sparse.mm(graph_pool, h)
        # pooled_h = graph_pool.spmm(h)

        return pooled_h, h_nodes


class MLPActor(nn.Module):
    def __init__(self, num_layers, input_dim, hidden_dim, output_dim):
        """

        Args:
            num_layers: number of layers in the neural networks (EXCLUDING the
                input layer). If num_layers=1, this reduces to linear model.
            input_dim: dimensionality of input features
            hidden_dim: dimensionality of hidden units at ALL layers
            output_dim: number of classes for prediction
        """
        super(MLPActor, self).__init__()

        self.linear_or_not = True  # default is linear model
        self.num_layers = num_layers

        if num_layers < 1:
            raise ValueError("number of layers should be positive!")
        elif num_layers == 1:
            # Linear model
            self.linear = nn.Linear(input_dim, output_dim)
        else:
            # Multi-layer model
            self.linear_or_not = False
            self.linears = torch.nn.ModuleList()
            self.linears.append(nn.Linear(input_dim, hidden_dim))
            for layer in range(num_layers - 2):
                self.linears.append(nn.Linear(hidden_dim, hidden_dim))
            self.linears.append(nn.Linear(hidden_dim, output_dim))

    def forward(self, x):
        if self.linear_or_not:
            # If linear model
            return self.linear(x)
        else:
            # If MLP
            h = x
            for layer in range(self.num_layers - 1):
                h = torch.tanh(self.linears[layer](h))
            return self.linears[self.num_layers - 1](h)


class MLPCritic(nn.Module):
    def __init__(self, num_layers, input_dim, hidden_dim, output_dim):
        """

        Args:
            num_layers: number of layers in the neural networks (EXCLUDING the
                input layer). If num_layers=1, this reduces to linear model.
            input_dim: dimensionality of input features
            hidden_dim: dimensionality of hidden units at ALL layers
            output_dim: number of classes for prediction
        """
        super(MLPCritic, self).__init__()

        self.linear_or_not = True  # default is linear model
        self.num_layers = num_layers

        if num_layers < 1:
            raise ValueError("number of layers should be positive!")
        elif num_layers == 1:
            # Linear model
            self.linear = nn.Linear(input_dim, output_dim)
        else:
            # Multi-layer model
            self.linear_or_not = False
            self.linears = torch.nn.ModuleList()
            self.linears.append(nn.Linear(input_dim, hidden_dim))
            for layer in range(num_layers - 2):
                self.linears.append(nn.Linear(hidden_dim, hidden_dim))
            self.linears.append(nn.Linear(hidden_dim, output_dim))

    def forward(self, x):
        if self.linear_or_not:
            # If linear model
            return self.linear(x)
        else:
            # If MLP
            h = x
            for layer in range(self.num_layers - 1):
                h = torch.tanh(self.linears[layer](h))
            return self.linears[self.num_layers - 1](h)


class ActorCritic(nn.Module):
    def __init__(self,
                 n_j,
                 n_m,
                 # feature extraction net unique attributes:
                 num_layers,
                 learn_eps,
                 neighbor_pooling_type,
                 input_dim,
                 hidden_dim,
                 # feature extraction net MLP attributes:
                 num_mlp_layers_feature_extract,
                 # actor net MLP attributes:
                 num_mlp_layers_actor,
                 hidden_dim_actor,
                 # actor net MLP attributes:
                 num_mlp_layers_critic,
                 hidden_dim_critic,
                 # actor/critic/feature_extraction shared attribute
                 device
                 ):
        super(ActorCritic, self).__init__()
        # job size for problems, no business with network
        self.n_j = n_j
        # machine size for problems, no business with network
        self.n_m = n_m
        self.device = device

        self.feature_extract = GraphCNN(num_layers=num_layers,
                                        num_mlp_layers=num_mlp_layers_feature_extract,
                                        input_dim=input_dim,
                                        hidden_dim=hidden_dim,
                                        learn_eps=learn_eps,
                                        neighbor_pooling_type=neighbor_pooling_type,
                                        device=device).to(device)
        self.actor = MLPActor(num_mlp_layers_actor, hidden_dim*2, hidden_dim_actor, 1).to(device)
        self.critic = MLPCritic(num_mlp_layers_critic, hidden_dim, hidden_dim_critic, 1).to(device)

    def forward(self,
                x,
                graph_pool,
                padded_nei,
                adj,
                candidate,
                mask,
                ):

        h_pooled, h_nodes = self.feature_extract(x=x,
                                                 graph_pool=graph_pool,
                                                 padded_nei=padded_nei,
                                                 adj=adj)
        # prepare policy feature: concat omega feature with global feature
        dummy = candidate.unsqueeze(-1).expand(-1, self.n_j, h_nodes.size(-1))
        candidate_feature = torch.gather(h_nodes.reshape(dummy.size(0), -1, dummy.size(-1)), 1, dummy)
        h_pooled_repeated = h_pooled.unsqueeze(1).expand_as(candidate_feature)

        # concatenate feature
        concateFea = torch.cat((candidate_feature, h_pooled_repeated), dim=-1)
        candidate_scores = self.actor(concateFea)

        # perform mask
        mask_reshape = mask.reshape(candidate_scores.size())
        candidate_scores[mask_reshape] = float('-inf')

        pi = F.softmax(candidate_scores, dim=1)
        v = self.critic(h_pooled)
        return pi, v
