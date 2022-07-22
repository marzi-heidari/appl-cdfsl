# This code is modified from https://github.com/jakesnell/prototypical-networks 

import numpy as np
import torch
import torch.nn as nn
from scipy.special import softmax
from torch.autograd import Variable

from methods.meta_template import MetaTemplate

from pykeops.torch import LazyTensor


def KMeans(x, K=10, Niter=10, verbose=True):
    """Implements Lloyd's algorithm for the Euclidean metric."""

    N, D = x.shape  # Number of samples, dimension of the ambient space

    c = x[:K, :].clone()  # Simplistic initialization for the centroids

    x_i = LazyTensor(x.view(N, 1, D))  # (N, 1, D) samples
    c_j = LazyTensor(c.view(1, K, D))  # (1, K, D) centroids

    # K-means loop:
    # - x  is the (N, D) point cloud,
    # - cl is the (N,) vector of class labels
    # - c  is the (K, D) cloud of cluster centroids
    for i in range(Niter):
        # E step: assign points to the closest cluster -------------------------
        D_ij = ((x_i - c_j) ** 2).sum(-1)  # (N, K) symbolic squared distances
        cl = D_ij.argmin(dim=1).long().view(-1)  # Points -> Nearest cluster

        # M step: update the centroids to the normalized cluster average: ------
        # Compute the sum of points per cluster:
        c.zero_()
        c.scatter_add_(0, cl[:, None].repeat(1, D), x)

        # Divide by the number of points per cluster:
        Ncl = torch.bincount(cl, minlength=K).type_as(c).view(K, 1)
        c /= Ncl  # in-place division to compute the average

    return cl, c


def init_weights(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight)
        m.bias.data.fill_(0.01)


class Perceptron(torch.nn.Module):
    def __init__(self, input_dim, output_dim):
        super(Perceptron, self).__init__()
        self.fc = nn.Linear(input_dim, output_dim)
        self.relu = torch.nn.ReLU()

    def forward(self, x):
        x = self.fc(x)
        output = self.relu(x)
        return output


class ProtoNet(MetaTemplate):
    def __init__(self, model_func, n_way, n_support):
        super(ProtoNet, self).__init__(model_func, n_way, n_support)
        self.loss_fn = nn.CrossEntropyLoss()
        self.p_prime = None
        self.p_prime_init = False
        self.alpha = 0.5
        self.n_shot_prime = 5
        self.prototype_net = Perceptron(self.n_shot_prime * 640, 640)
        if self.n_shot > 5:
            self.prototype_net = self.prototype_net.cuda(0)

    def set_forward(self, x, finetune=False, query_labels=None, is_feature=False, shuffle=False):
        z_support, z_query = self.parse_feature(x, is_feature)
        if shuffle:
            randperm = torch.randperm(z_support.size()[1])
            z_support = z_support[:, randperm, :]
        z_support = z_support.contiguous()
        if self.n_shot > 5:
            z_clust_center = torch.zeros((self.n_way, self.n_shot_prime, z_support.shape[2])).to(z_support.device)
            for iter in range(self.n_way):
                _, z_clust_center[iter, 0:self.n_shot_prime, :] = KMeans(z_support[iter, :, :], self.n_shot_prime)
            support_view_for_prototype_net = z_clust_center.view(self.n_way, self.n_shot_prime * 640)
            support_view_for_prototype_net = support_view_for_prototype_net.cuda(0)
        else:
            support_view_for_prototype_net = z_support.view(self.n_way, self.n_shot_prime * 640)

        z_proto = self.prototype_net(support_view_for_prototype_net)
        z_query = z_query.contiguous().view(self.n_way * self.n_query, -1)
        query_to_proto_distance = euclidean_dist(z_query, z_proto)
        n_support = self.n_shot
        support_to_proto_distance = euclidean_dist(z_support.contiguous().view(self.n_way * n_support, -1),
                                                   z_proto)
        numpy = torch.ones(1).cuda()
        push_loss = numpy / euclidean_dist(z_proto, z_proto).sum()
        if finetune:
            target = torch.from_numpy(np.repeat(range(self.n_way), n_support))
            pull_loss = support_to_proto_distance[list(range(self.n_way * n_support)), target].sum()
        else:
            labels = torch.from_numpy(np.repeat(range(self.n_way), self.n_query))
            pull_loss = query_to_proto_distance[list(range(self.n_way * self.n_query)), labels].sum()
        return -query_to_proto_distance, push_loss, pull_loss, -support_to_proto_distance

    def set_forward_loss(self, x, finetune=False, shuffle=False):
        p_xj_test, prototypes_dist, pull_term, support_scores = self.set_forward(
            x, finetune, shuffle=shuffle)
        if self.n_shot == 1:
            n_support = self.n_shot_prime
        else:
            n_support = self.n_shot
        y_support = torch.from_numpy(np.repeat(range(self.n_way), n_support))
        if finetune:
            if self.p_prime_init:
                p = self.alpha * p_xj_test.data.cpu().numpy() + (1 - self.alpha) * self.p_prime
                self.p_prime = p
                soft_p = softmax(p, axis=1)
                index = []
                for i in range(np.shape(soft_p)[0]):
                    if soft_p[i].max() > 0.4:
                        index.append(i)
                y_query = torch.from_numpy(p.argmax(axis=1))[index]
                y_query = Variable(y_query.cuda())
                loss = self.loss_fn(p_xj_test[index], y_query) \
                       + self.loss_fn(support_scores,
                                      Variable(y_support.cuda())) + prototypes_dist + 1e-3 * pull_term
                return loss
            else:
                self.p_prime_init = True
                self.p_prime = p_xj_test.data.cpu().numpy()
                return self.loss_fn(support_scores, Variable(y_support.cuda())) + prototypes_dist \
                       + 1e-3 * pull_term
        else:
            y_query = torch.from_numpy(np.repeat(range(self.n_way), self.n_query))
            cuda_y_query = y_query.cuda()
            y_query = Variable(cuda_y_query)
            return self.loss_fn(p_xj_test, y_query) \
                   + 1e-3 * pull_term \
                   + prototypes_dist


def euclidean_dist(x, y):
    # x: N x D
    # y: M x D
    n = x.size(0)
    m = y.size(0)
    d = x.size(1)
    assert d == y.size(1)

    x = x.unsqueeze(1).expand(n, m, d)
    y = y.unsqueeze(0).expand(n, m, d)

    return torch.pow(x - y, 2).sum(2)
