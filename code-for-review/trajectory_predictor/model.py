import math
import torch
import random
import itertools
import networkx as nx
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from utils.utils import gumbel_softmax, custom_softmax, sample_core
from trajectory_predictor.encoder import MLP


class TrojectoryPredictor(nn.Module):
    def __init__(
        self,
        n_in: int,
        n_fc_out: int,
        n_hid: int,
        tau: float,
        hard: bool,
        n_layers: int = 1,
        do_prob: float = 0.0,
    ):
        super(TrojectoryPredictor, self).__init__()
        self.n_in = n_in
        self.n_fc_out = n_fc_out
        self.n_hid = n_hid
        self.n_layers = n_layers
        self.do_prob = do_prob
        self.mlp1 = MLP(n_in, n_fc_out, n_fc_out, do_prob)
        self.mlp2 = MLP(n_fc_out, n_fc_out, n_fc_out, do_prob)

        self.hidden_r = nn.Linear(n_hid, n_hid, bias=False)
        self.hidden_i = nn.Linear(n_hid, n_hid, bias=False)
        self.hidden_h = nn.Linear(n_hid, n_hid, bias=False)

        self.input_r = nn.Linear(2 * n_fc_out, n_hid, bias=True)
        self.input_i = nn.Linear(2 * n_fc_out, n_hid, bias=True)
        self.input_n = nn.Linear(2 * n_fc_out, n_hid, bias=True)

        self.out_fc1 = nn.Linear(n_hid, n_hid)
        self.out_fc2 = nn.Linear(n_hid, n_hid)
        self.out_fc3 = nn.Linear(n_hid, n_in)

        self.tau = tau
        self.hard = hard

    def init_weights(self):
        for m in self.modules():
            nn.init.xavier_normal_(m.weight.data)
            m.bias.data.fill_(0.1)

    def edge2node(self, x, rel_rec, rel_send):
        incoming = torch.matmul(rel_rec.transpose(1, 2), x)
        return incoming / incoming.size(1)

    def node2edge(self, x, rel_rec, rel_send):
        receivers = torch.matmul(rel_rec, x)
        senders = torch.matmul(rel_send, x)
        edges = torch.cat([receivers, senders], dim=2)
        return edges

    def forward(
        self,
        x: torch.Tensor = None,
        encoder1 = None,
        encoder2 = None,
        decoder = None,
        inputs = None,
        rel_rec1 = None,
        rel_send1 = None,
        rel_rec2 = None,
        rel_send2 = None,
        total_pred_steps = None,
        encoder_timesteps = None,
        recompute_gap = None,
        var = None,
        agent_types = None,
        pre_train: bool = False,
    ):
        if x is not None:
            return self._forward_trajectory(x)
        else:
            return self._forward_graph(
                encoder1,
                encoder2,
                decoder,
                inputs,
                rel_rec1,
                rel_send1,
                rel_rec2,
                rel_send2,
                total_pred_steps,
                encoder_timesteps,
                recompute_gap,
                var,
                agent_types,
                pre_train,
            )

    def _forward_trajectory(self, x: torch.Tensor):
        batch_size = x.size(0)
        batch_idx = 0
        w_list = []
        mu_list = []
        pred_traj = []
        current_frame = x[batch_idx, :, 0].to(self.device, dtype=torch.float32)
        h = torch.zeros((1, self.n_node, self.h_dim)).to(self.device, dtype=torch.float32)

        nll_loss = torch.tensor([0]).to(self.device, dtype=torch.float32)
        kld_loss1 = torch.tensor([0]).to(self.device, dtype=torch.float32)
        kld_loss2 = torch.tensor([0]).to(self.device, dtype=torch.float32)

        for frame_idx in range(self.len_traj - 1):
            if frame_idx < self.hf_split - 1:
                g1, q1, edge_cluster1, edge_feature1 = self.encoder1(current_frame, h[batch_idx])
                
                if frame_idx % 5 == 0:
                    he = self.he_distribute(q1)
                    he = F.normalize(he - torch.min(he), p=1, dim=0)
                    
                g2, _, edge_cluster2, edge_feature2 = self.encoder2(
                    current_frame, h[batch_idx], he
                )

                inp = torch.unsqueeze(
                    torch.cat(
                        (
                            torch.matmul(g1, edge_feature1),
                            torch.matmul(g2, edge_feature2),
                            current_frame,
                        ),
                        dim=-1,
                    ),
                    dim=1,
                )
                _, h = self.gru(inp, h)
                current_frame = x[batch_idx, :, frame_idx + 1].to(
                    self.device, dtype=torch.float32
                )

            if frame_idx >= self.hf_split - 1:
                next_frame = x[batch_idx, :, frame_idx + 1]
                g1, q1, edge_cluster1, edge_feature1 = self.encoder1(current_frame, h[batch_idx])
                
                if frame_idx % 5 == 0:
                    he = self.he_distribute(q1)
                    he = F.normalize(he - torch.min(he), p=1, dim=0)
                    
                g2, _, edge_cluster2, edge_feature2 = self.encoder2(
                    current_frame, h[batch_idx], he
                )

                inp = torch.unsqueeze(
                    torch.cat(
                        (
                            torch.matmul(g1, edge_feature1),
                            torch.matmul(g2, edge_feature2),
                            current_frame,
                        ),
                        dim=-1,
                    ),
                    dim=1,
                )
                _, h = self.gru(inp, h)

                w_list_step = []
                mu_list_step = []
                next_frame_ = torch.zeros(current_frame.size()).to(self.device)
                
                for i in range(self.n_gaussian):
                    w = self.out_w[i](h[0])
                    w = F.normalize(w - torch.min(w), p=1, dim=0)
                    mu = self.out_mu[i](h[0])
                    w_list_step.append(w)
                    mu_list_step.append(mu)
                    next_frame_ += w * self._reparameterized(
                        current_frame + mu,
                        1e-2 * torch.ones(mu.size()).to(self.device),
                    )
                    
                w_list.append(torch.stack(w_list_step, dim=0))
                mu_list.append(torch.stack(mu_list_step, dim=0))

                nll_loss += nn.MSELoss()(next_frame_, next_frame)
                kld_loss1 += self._kld_gaussian(
                    edge_cluster1,
                    torch.zeros(1).to(self.device),
                    torch.tensor(
                        [1 / self.n_edge_type] * self.n_edge_type, dtype=torch.float32
                    ).to(self.device),
                    torch.zeros(1).to(self.device),
                )
                kld_loss2 += self._kld_gaussian(
                    edge_cluster2,
                    torch.zeros(1).to(self.device),
                    torch.tensor(
                        [1 / self.n_edge_type] * self.n_edge_type, dtype=torch.float32
                    ).to(self.device),
                    torch.zeros(1).to(self.device),
                )

                current_frame = next_frame_.detach()
                pred_traj.append(next_frame_.detach())

        pred_traj = torch.permute(torch.stack(pred_traj), (1, 0, 2))
        ade = self._cal_ade(x[batch_idx, :, self.hf_split :], pred_traj)
        fde = self._cal_fde(x[batch_idx, :, self.hf_split :], pred_traj)
        
        return pred_traj, nll_loss, kld_loss1, kld_loss2, ade, fde

    def _forward_graph(
        self,
        encoder1,
        encoder2,
        decoder,
        inputs,
        rel_rec1,
        rel_send1,
        rel_rec2,
        rel_send2,
        total_pred_steps,
        encoder_timesteps,
        recompute_gap,
        var,
        agent_types,
        pre_train=False,
    ):
        graph = encoder1.forward(inputs, rel_rec1, rel_send1)
        graph[:, :, :] = graph[0, 0]
        x1 = self.mlp1(graph)
        x1 = self.edge2node(x1, rel_rec1, rel_send1)
        x1 = self.mlp2(x1)
        x1 = self.node2edge(x1, rel_rec1, rel_send1)

        hidden = torch.zeros((graph.size(0), graph.size(1), self.n_hid))
        if inputs.is_cuda:
            hidden = hidden.cuda()
        r = torch.sigmoid(self.input_r(x1))
        i = torch.sigmoid(self.input_i(x1))
        n = torch.tanh(self.input_n(x1))
        hidden = (1 - i) * n + i * hidden

        output_g_graph = F.dropout(F.relu(self.out_fc1(hidden)), p=self.do_prob)
        output_g_graph = F.dropout(F.relu(self.out_fc2(output_g_graph)), p=self.do_prob)
        output_g_graph = self.out_fc3(output_g_graph)
        output_g_prob = custom_softmax(output_g_graph, -1)
        output_g_graph = gumbel_softmax(output_g_graph, tau=self.tau, hard=self.hard)

        output_hg_graph = None
        if not pre_train:
            h_graph = encoder2.forward(inputs, rel_rec2, rel_send2)
            h_graph[:, :, :] = h_graph[0, 0]
            x2 = self.mlp1(h_graph)
            x2 = self.edge2node(x2, rel_rec2, rel_send2)
            x2 = self.mlp2(x2)
            x2 = self.node2edge(x2, rel_rec2, rel_send2)

            h_hidden = torch.zeros((h_graph.size(0), h_graph.size(1), self.n_hid))
            if inputs.is_cuda:
                h_hidden = h_hidden.cuda()
            r = torch.sigmoid(self.input_r(x2))
            i = torch.sigmoid(self.input_i(x2))
            n = torch.tanh(self.input_n(x2))
            h_hidden = (1 - i) * n + i * h_hidden

            output_hg_graph = F.dropout(F.relu(self.out_fc1(h_hidden)), p=self.do_prob)
            output_hg_graph = F.dropout(F.relu(self.out_fc2(output_hg_graph)), p=self.do_prob)
            output_hg_graph = self.out_fc3(output_hg_graph)
            output_hg_graph = gumbel_softmax(output_hg_graph, tau=self.tau, hard=self.hard)

        time_steps_left = total_pred_steps - encoder_timesteps - recompute_gap
        output_traj, alphas, mus, sigmas = decoder.forward(
            inputs,
            output_g_graph,
            rel_rec1,
            rel_send1,
            output_hg_graph,
            rel_rec2,
            rel_send2,
            encoder_timesteps + recompute_gap,
            var,
            burn_in=True,
            burn_in_steps=encoder_timesteps,
            pre_train=pre_train,
        )

        output_traj = output_traj[:, :, -recompute_gap:, :]
        alphas = alphas[:, :, -recompute_gap:, :]
        mus = mus[:, :, -recompute_gap:, :, :]
        sigmas = sigmas[:, :, -recompute_gap:, :, :]

        if recompute_gap < encoder_timesteps:
            inputs = torch.cat(
                (inputs[:, :, -(encoder_timesteps - recompute_gap):, :], output_traj),
                dim=2,
            )
        else:
            inputs = output_traj[:, :, -encoder_timesteps:, :]

        output_lists = {
            "g_graphs": [output_g_graph],
            "hg_graphs": [output_hg_graph],
            "probs": [output_g_prob],
            "trajs": [output_traj],
            "alphas": [alphas],
            "mus": [mus],
            "sigmas": [sigmas],
        }

        num_new_graph = math.ceil(
            (total_pred_steps - encoder_timesteps) / recompute_gap
        ) - 1

        for _ in range(num_new_graph):
            graph = encoder1.forward(inputs, rel_rec1, rel_send1)
            graph[:, :, :] = graph[0, 0]
            x1 = self.mlp1(graph)
            x1 = self.edge2node(x1, rel_rec1, rel_send1)
            x1 = self.mlp2(x1)
            x1 = self.node2edge(x1, rel_rec1, rel_send1)

            r = torch.sigmoid(self.input_r(x1))
            i = torch.sigmoid(self.input_i(x1))
            n = torch.tanh(self.input_n(x1))
            hidden = (1 - i) * n + i * hidden

            output_g_graph = F.dropout(F.relu(self.out_fc1(hidden)), p=self.do_prob)
            output_g_graph = F.dropout(F.relu(self.out_fc2(output_g_graph)), p=self.do_prob)
            output_g_graph = self.out_fc3(output_g_graph)
            output_g_prob = custom_softmax(output_g_graph, -1)
            output_g_graph = gumbel_softmax(output_g_graph, tau=self.tau, hard=self.hard)

            output_hg_graph = None
            if not pre_train:
                h_graph = encoder2.forward(inputs, rel_rec2, rel_send2)
                h_graph[:, :, :] = h_graph[0, 0]
                x2 = self.mlp1(h_graph)
                x2 = self.edge2node(x2, rel_rec2, rel_send2)
                x2 = self.mlp2(x2)
                x2 = self.node2edge(x2, rel_rec2, rel_send2)

                r = torch.sigmoid(self.input_r(x2))
                i = torch.sigmoid(self.input_i(x2))
                n = torch.tanh(self.input_n(x2))
                h_hidden = (1 - i) * n + i * h_hidden

                output_hg_graph = F.dropout(F.relu(self.out_fc1(h_hidden)), p=self.do_prob)
                output_hg_graph = F.dropout(F.relu(self.out_fc2(output_hg_graph)), p=self.do_prob)
                output_hg_graph = self.out_fc3(output_hg_graph)
                output_hg_graph = gumbel_softmax(output_hg_graph, tau=self.tau, hard=self.hard)

            output_traj, alphas, mus, sigmas = decoder.forward(
                inputs,
                output_g_graph,
                rel_rec1,
                rel_send1,
                output_hg_graph,
                rel_rec2,
                rel_send2,
                encoder_timesteps + recompute_gap,
                var,
                burn_in=True,
                burn_in_steps=encoder_timesteps,
                pre_train=pre_train,
            )

            if time_steps_left >= recompute_gap:
                output_traj = output_traj[:, :, -recompute_gap:, :]
                alphas = alphas[:, :, -recompute_gap:, :]
                mus = mus[:, :, -recompute_gap:, :, :]
                sigmas = sigmas[:, :, -recompute_gap:, :, :]
            else:
                output_traj = output_traj[:, :, -time_steps_left:, :]
                alphas = alphas[:, :, -time_steps_left:, :]
                mus = mus[:, :, -time_steps_left:, :, :]
                sigmas = sigmas[:, :, -time_steps_left:, :, :]

            time_steps_left -= recompute_gap
            if recompute_gap < encoder_timesteps:
                inputs = torch.cat(
                    (inputs[:, :, -(encoder_timesteps - recompute_gap):, :], output_traj),
                    dim=2,
                )
            else:
                inputs = output_traj[:, :, -encoder_timesteps:, :]

            output_lists["g_graphs"].append(output_g_graph)
            output_lists["hg_graphs"].append(output_hg_graph)
            output_lists["probs"].append(output_g_prob)
            output_lists["trajs"].append(output_traj)
            output_lists["alphas"].append(alphas)
            output_lists["mus"].append(mus)
            output_lists["sigmas"].append(sigmas)

        return (
            output_lists["trajs"],
            output_lists["g_graphs"],
            output_lists["hg_graphs"],
            output_lists["probs"],
            output_lists["alphas"],
            output_lists["mus"],
            output_lists["sigmas"],
        )

    def _reparameterized(
        self, 
        mu: torch.Tensor, 
        std: torch.Tensor,
    ) -> torch.Tensor:
        epsilon = Variable(torch.FloatTensor(std.size()).normal_()).to(self.device)
        return epsilon.mul(std).add_(mu)

    def _nll_gaussian(
        self,
        mu_: torch.Tensor,
        mu: torch.Tensor,
        w: torch.Tensor,
        std: torch.Tensor,
    ) -> torch.Tensor:
        nll_all = torch.tensor([0]).to(self.device, dtype=torch.float32)
        
        for gaussian_idx in range(self.n_gaussian):
            this_mu_ = mu_[:, :, gaussian_idx]
            this_w = w[:, :, gaussian_idx, 0]
            temp = torch.sum((this_mu_ - mu) ** 2 / (2 * std), dim=-1)
            this_nll = this_w * temp
            this_nll = torch.sum(this_nll, dim=-1)
            this_nll = torch.sum(this_nll, dim=-1)
            nll_all += this_nll
            
        return nll_all

    def _kld_gaussian(
        self,
        mu_1: torch.Tensor,
        logvar_1: torch.Tensor,
        mu_2: torch.Tensor,
        logvar_2: torch.Tensor,
    ) -> torch.Tensor:
        num_nodes = mu_1.size()[0]
        kld_element = (
            2 * (logvar_2 - logvar_1)
            + (torch.exp(2 * logvar_1) + torch.pow(mu_1 - mu_2, 2))
            / torch.exp(2 * logvar_2)
        )
        return (0.5 / num_nodes) * torch.mean(torch.sum(kld_element, dim=1), dim=0)

    def _cal_ade(
        self, 
        pred_traj: torch.Tensor, 
        gt_traj: torch.Tensor,
    ) -> torch.Tensor:
        error_sum_xy = torch.sqrt(torch.sum(torch.pow(pred_traj - gt_traj, 2), dim=-1))
        error_node = torch.mean(error_sum_xy, dim=-1)
        error = torch.mean(error_node, dim=-1)
        return error

    def _cal_fde(
        self, 
        pred_traj: torch.Tensor, 
        gt_traj: torch.Tensor,
    ) -> torch.Tensor:
        error = torch.linalg.norm(pred_traj[:, -1] - gt_traj[:, -1], axis=-1)
        return torch.mean(error, axis=-1)

    def _best_of_n(
        self,
        next_frame: torch.Tensor,
        w_list: torch.Tensor,
        mu_list: torch.Tensor,
        n_sample: int,
    ) -> torch.Tensor:
        sample_loss = torch.tensor(0).to(self.device, dtype=torch.float32)
        
        for node_idx in range(self.n_node):
            sample_idx = random.choices(
                population=range(self.n_gaussian),
                weights=w_list[:, node_idx],
                k=n_sample,
            )
            sample_loss_list = []
            
            for i in sample_idx:
                loss = nn.MSELoss()(
                    self._reparameterized(
                        mu_list[i],
                        1e-3 * torch.ones(mu_list[i].size()).to(self.device),
                    ),
                    next_frame,
                )
                sample_loss_list.append(loss)
                
            sample_loss += torch.min(torch.stack(sample_loss_list, dim=0))
            
        return sample_loss
