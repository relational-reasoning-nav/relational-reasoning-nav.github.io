import torch
import math
import numpy as np
import torch.nn.functional as F
from torch.autograd import Variable
from sklearn.cluster import KMeans
from sklearn.neighbors import NearestNeighbors


def reparameterized(mu, std, device):
    epsilon = Variable(torch.FloatTensor(std.size()).normal_()).to(device)
    return epsilon.mul(std).add_(mu)


def cal_ade(pred_traj, gt_traj):
    error_sum_xy = torch.sqrt(torch.sum(torch.pow(pred_traj - gt_traj, 2), dim=-1))
    error_node = torch.mean(error_sum_xy, dim=-1)
    error = torch.mean(error_node, dim=-1)
    return error


def cal_fde(pred_traj, gt_traj):
    error = torch.linalg.norm(pred_traj[:, -1] - gt_traj[:, -1], axis=-1)
    return torch.mean(error, axis=-1)


def create_hg(
    node_list: np.array,
    n_cluster: int,
    n_nearest_neighbor: int,
    n_nearest_cluster: int,
) -> list:
    hypergraph = []
    C = KMeans(n_clusters=n_cluster).fit(node_list)
    cluster_list = []
    for i in range(n_cluster):
        cluster_list.append([j for j, x in enumerate(C.labels_) if x == i])

    nearest_neighbors = NearestNeighbors(
        n_neighbors=n_nearest_neighbor, algorithm="ball_tree"
    ).fit(node_list)
    _, indices = nearest_neighbors.kneighbors(node_list)
    cluster_centers = C.cluster_centers_
    for i in range(indices.shape[0]):
        hyperedge = []

        for j in range(indices.shape[1]):
            hyperedge.append(indices[i, j])

        node_pos = node_list[i]
        dist_list = []
        for j in range(n_cluster):
            center_pos = cluster_centers[j]
            dist = math.hypot(center_pos[0] - node_pos[0], center_pos[1] - node_pos[1])
            dist_list.append(dist)
        min_value = min(dist_list)
        min_index = dist_list.index(min_value)
        for j in cluster_list[min_index]:
            hyperedge.append(j)

        hypergraph.append(hyperedge)
    return hypergraph


def softmax(input, axis=1, dim=0):
    trans_input = input.transpose(axis, 0).contiguous()
    soft_max_1d = F.softmax(trans_input, dim=dim)
    return soft_max_1d.transpose(axis, 0)


def sample_gumbel(shape, eps=1e-10):
    U = torch.rand(shape).float()
    return -torch.log(eps - torch.log(U + eps))


def gumbel_softmax_sample(logits, tau=1, dim=0, eps=1e-10):
    gumbel_noise = sample_gumbel(logits.size(), eps=eps)
    if logits.is_cuda:
        gumbel_noise = gumbel_noise.cuda()
    y = logits + Variable(gumbel_noise)
    return softmax(y / tau, axis=-1, dim=dim)


def gumbel_softmax(logits, tau=1, dim=0, hard=False, eps=1e-10):
    y_soft = gumbel_softmax_sample(logits, tau=tau, dim=dim, eps=eps)
    if hard:
        shape = logits.size()
        _, k = y_soft.data.max(-1)
        y_hard = torch.zeros(*shape)
        if y_soft.is_cuda:
            y_hard = y_hard.cuda()
        y_hard = y_hard.zero_().scatter_(-1, k.view(shape[:-1] + (1,)), 1.0)
        y = Variable(y_hard - y_soft.data) + y_soft
    else:
        y = y_soft
    return y


def sample_core(weights, mu):
    original_mu_shape = mu.shape
    weights = weights.reshape(-1, weights.shape[-1])
    mu = mu.reshape(-1, mu.shape[-2], mu.shape[-1])
    categorical_distribution = torch.distributions.categorical.Categorical(weights)
    category = categorical_distribution.sample()
    selected_mu = torch.zeros(mu.shape[0], mu.shape[2])
    for i in range(category.shape[0]):
        selected_mu[i] = mu[i, category[i]]
    if len(original_mu_shape) == 4:
        selected_mu = selected_mu.reshape(
            original_mu_shape[0], original_mu_shape[1], original_mu_shape[-1]
        )
    return selected_mu


def encode_onehot(labels):
    classes = set(labels)
    classes_dict = {c: np.identity(len(classes))[i, :] for i, c in enumerate(classes)}
    labels_onehot = np.array(list(map(classes_dict.get, labels)), dtype=np.int32)
    return labels_onehot


def kl_categorical_uniform(preds, num_atoms, num_edge_types, add_const=False, eps=1e-16):
    kl_div = preds * torch.log(preds + eps)
    if add_const:
        const = np.log(num_edge_types)
        kl_div += const
    return kl_div.sum() / (num_atoms * preds.size(0))


def nll_loss(mu, target, alpha, sigma):
    original_mu_shape = mu.shape
    nll_all = torch.zeros(original_mu_shape[0], original_mu_shape[1])
    if mu.is_cuda:
        nll_all = nll_all.cuda()
    for core_index in range(mu.shape[3]):
        this_mu = mu[:, :, :, core_index, :]
        this_sigma = sigma[:, :, :, core_index, :]
        this_alpha = alpha[:, :, :, core_index]
        temp = torch.sum((this_mu - target) ** 2 / (2 * this_sigma), dim=-1)
        this_nll = this_alpha * temp
        this_nll = torch.sum(this_nll, dim=-1)
        nll_all += this_nll
    return nll_all


def compute_ade(output, target, type="sum"):
    if type == "sum":
        diff = output - target
        diff = diff**2
        diff_1 = torch.mean(torch.sqrt(torch.sum(diff, dim=3))) * 0.3
    elif type == "no_sum":
        diff = output - target
        diff = diff**2
        diff_1 = torch.mean(torch.sqrt(torch.sum(diff, dim=3)), dim=2) * 0.3
    return diff_1


def compute_fde(output, target, type="sum"):
    if type == "sum":
        diff = output - target
        diff = diff**2
        diff_2 = torch.sum(torch.sqrt(torch.sum(diff[:, :, -1, :2], dim=2))) * 0.3
    elif type == "no_sum":
        diff = output - target
        diff = diff**2
        diff_2 = torch.sqrt(torch.sum(diff[:, :, -1, :2], dim=2)) * 0.3
    return diff_2


def generate_mask(valid_list, max_num):
    mask = torch.zeros(valid_list.shape[0], max_num)
    for i in range(valid_list.shape[0]):
        mask[i, : valid_list[i]] = 1
    return mask


def custom_softmax(input, axis=1):
    trans_input = input.transpose(axis, 0).contiguous()
    soft_max_1d = F.softmax(trans_input)
    return soft_max_1d.transpose(axis, 0)


def reshape_tensor(tensor: torch.Tensor, seq_length: int, nenv: int) -> torch.Tensor:
    shape = tensor.size()[1:]
    return tensor.unsqueeze(0).reshape((seq_length, nenv, *shape))
