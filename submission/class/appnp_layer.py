"""APPNP and PPNP layers."""

import math
import torch
from torch_sparse import spmm
from scipy import sparse
import numpy as np
def create_adjacency_matrix(graph):
    """
    Creating a sparse adjacency matrix.
    :param graph: NetworkX object.
    :return A: Adjacency matrix.
    """
    index_1 = [edge[0] for edge in graph.edges()] + [edge[1] for edge in graph.edges()]
    index_2 = [edge[1] for edge in graph.edges()] + [edge[0] for edge in graph.edges()]
    values = [1 for edge in index_1]
    node_count = max(max(index_1)+1, max(index_2)+1)
    A = sparse.coo_matrix((values, (index_1, index_2)), shape=(node_count, node_count), dtype=np.float32)
    return A

def uniform(size, tensor):
    """
    Uniform weight initialization.
    :param size: Size of the tensor.
    :param tensor: Tensor initialized.
    """
    stdv = 1.0 / math.sqrt(size)
    if tensor is not None:
        tensor.data.uniform_(-stdv, stdv)

def create_adjacency_matrix(graph):
    """
    Creating a sparse adjacency matrix.
    :param graph: NetworkX object.
    :return A: Adjacency matrix.
    """
    index_1 = [edge[0] for edge in graph.edges()] + [edge[1] for edge in graph.edges()]
    index_2 = [edge[1] for edge in graph.edges()] + [edge[0] for edge in graph.edges()]
    values = [1 for edge in index_1]
    node_count = max(max(index_1)+1, max(index_2)+1)
    A = sparse.coo_matrix((values, (index_1, index_2)), shape=(node_count, node_count), dtype=np.float32)
    return A

def normalize_adjacency_matrix(A, I):
    """
    Creating a normalized adjacency matrix with self loops.
    :param A: Sparse adjacency matrix.
    :param I: Identity matrix.
    :return A_tile_hat: Normalized adjacency matrix.
    """
    A_tilde = A + I
    degrees = A_tilde.sum(axis=0)[0].tolist()
    D = sparse.diags(degrees, [0])
    D = D.power(-0.5)
    A_tilde_hat = D.dot(A_tilde).dot(D)
    return A_tilde_hat

def create_propagator_matrix(graph, alpha, model):
    """
    Creating  apropagation matrix.
    :param graph: NetworkX graph.
    :param alpha: Teleport parameter.
    :param model: Type of model exact or approximate.
    :return propagator: Propagator matrix Dense torch matrix /
    dict with indices and values for sparse multiplication.
    """
    A = create_adjacency_matrix(graph)
    I = sparse.eye(A.shape[0])
    A_tilde_hat = normalize_adjacency_matrix(A, I)
    if model == "exact":
        propagator = (I-(1-alpha)*A_tilde_hat).todense()
        propagator = alpha*torch.inverse(torch.FloatTensor(propagator))
    else:
        propagator = dict()
        A_tilde_hat = sparse.coo_matrix(A_tilde_hat)
        indices = np.concatenate([A_tilde_hat.row.reshape(-1, 1), A_tilde_hat.col.reshape(-1, 1)], axis=1).T
        propagator["indices"] = torch.LongTensor(indices)
        propagator["values"] = torch.FloatTensor(A_tilde_hat.data)
    return propagator

class DenseFullyConnected(torch.nn.Module):
    """
    Abstract class for PageRank and Approximate PageRank networks.
    :param in_channels: Number of input channels.
    :param out_channels: Number of output channels.
    :param density: Feature matrix structure.
    """
    def __init__(self, in_channels, out_channels):
        super(DenseFullyConnected, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.define_parameters()
        self.init_parameters()

    def define_parameters(self):
        """
        Defining the weight matrices.
        """
        self.weight_matrix = torch.nn.Parameter(torch.Tensor(self.in_channels, self.out_channels))
        self.bias = torch.nn.Parameter(torch.Tensor(self.out_channels))

    def init_parameters(self):
        """
        Initializing weights.
        """
        torch.nn.init.xavier_uniform_(self.weight_matrix)
        uniform(self.out_channels, self.bias)

    def forward(self, features):
        """
        Doing a forward pass.
        :param features: Feature matrix.
        :return filtered_features: Convolved features.
        """
        filtered_features = torch.mm(features, self.weight_matrix)
        filtered_features = filtered_features + self.bias
        return filtered_features

class SparseFullyConnected(torch.nn.Module):
    """
    Abstract class for PageRank and Approximate PageRank networks.
    :param in_channels: Number of input channels.
    :param out_channels: Number of output channels.
    :param density: Feature matrix structure.
    """
    def __init__(self, in_channels, out_channels):
        super(SparseFullyConnected, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.define_parameters()
        self.init_parameters()

    def define_parameters(self):
        """
        Defining the weight matrices.
        """
        self.weight_matrix = torch.nn.Parameter(torch.Tensor(self.in_channels, self.out_channels))
        self.bias = torch.nn.Parameter(torch.Tensor(self.out_channels))

    def init_parameters(self):
        """
        Initializing weights.
        """
        torch.nn.init.xavier_uniform_(self.weight_matrix)
        uniform(self.out_channels, self.bias)

    def forward(self, feature_indices, feature_values,number_of_features):
        """
        Making a forward pass.
        :param feature_indices: Non zero value indices.
        :param feature_values: Matrix values.
        :return filtered_features: Output features.
        """
        number_of_nodes = torch.max(feature_indices[0]).item()+1
        number_of_features = torch.max(feature_indices[1]).item()+1
        filtered_features = spmm(index = feature_indices,
                                 value = feature_values,
                                 m = number_of_nodes,
                                 n = number_of_features,
                                 matrix = self.weight_matrix)
        filtered_features = filtered_features + self.bias
        return filtered_features

class APPNPModel(torch.nn.Module):
    """
    APPNP Model Class.
    :param args: Arguments object.
    :param number_of_labels: Number of target labels.
    :param number_of_features: Number of input features.
    :param graph: NetworkX graph.
    :param device: CPU or GPU.
    """
    def __init__(self, number_of_labels, number_of_features, graph, device):
        super(APPNPModel, self).__init__()
        self.number_of_labels = number_of_labels
        self.number_of_features = number_of_features
        self.graph = graph
        self.device = device
        self.setup_layers()
        self.setup_propagator()

    def setup_layers(self):
        """
        Creating layers.
        """
        self.layer_1 = SparseFullyConnected(self.number_of_features, 64)
        self.layer_2 = DenseFullyConnected(64, self.number_of_labels)

    def setup_propagator(self):
        """
        Defining propagation matrix (Personalized Pagrerank or adjacency).
        """
        self.propagator = create_propagator_matrix(self.graph,0.1, "exact")
        self.propagator = self.propagator.to(self.device)

    def forward(self, feature_indices, feature_values):
        """
        Making a forward propagation pass.
        :param feature_indices: Feature indices for feature matrix.
        :param feature_values: Values in the feature matrix.
        :return self.predictions: Predicted class label log softmaxes.
        """
        feature_values = torch.nn.functional.dropout(feature_values,
                                                     p=0.5,
                                                     training=self.training)
        latent_features_1 = self.layer_1(feature_indices, feature_values,self.number_of_features)

        latent_features_1 = torch.nn.functional.relu(latent_features_1)

        latent_features_1 = torch.nn.functional.dropout(latent_features_1,
                                                        p=0.5,
                                                        training=self.training)

        latent_features_2 = self.layer_2(latent_features_1)
        self.predictions = torch.nn.functional.dropout(self.propagator,
                                                       p=0.5,
                                                       training=self.training)

        self.predictions = torch.mm(self.predictions, latent_features_2)
        self.predictions = torch.nn.functional.log_softmax(self.predictions, dim=1)
        return self.predictions
