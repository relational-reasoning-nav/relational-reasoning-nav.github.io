import math
import torch
import random
import itertools
import networkx as nx
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from utils.utils import gumbel_softmax, custom_softmax, sample_core


class RNNDecoder(nn.Module):
    def __init__(
        self,
        n_in_node,
        edge_types,
        n_hid,
        num_cores=None,
        do_prob=0.0,
        skip_first=False,
        env_flag=False,
        n_env_in=0,
    ):
        super(RNNDecoder, self).__init__()
        self.num_cores = num_cores
        self.dim = n_in_node
        self.env_flag = env_flag
        self.msg_out_shape = n_hid
        self.skip_first_edge_type = skip_first

        self.msg_fc1_g = nn.ModuleList(
            [nn.Linear(2 * n_hid, n_hid) for _ in range(edge_types)]
        )
        self.msg_fc1_hg = nn.ModuleList(
            [nn.Linear(2 * n_hid, n_hid) for _ in range(edge_types)]
        )
        self.msg_fc2_g = nn.ModuleList([nn.Linear(n_hid, n_hid) for _ in range(edge_types)])
        self.msg_fc2_hg = nn.ModuleList(
            [nn.Linear(n_hid, n_hid) for _ in range(edge_types)]
        )

        self.hidden_r_g = nn.Linear(n_hid, n_hid, bias=False)
        self.hidden_i_g = nn.Linear(n_hid, n_hid, bias=False)
        self.hidden_h_g = nn.Linear(n_hid, n_hid, bias=False)

        self.input_r_g = nn.Linear(n_in_node, n_hid, bias=True)
        self.input_i_g = nn.Linear(n_in_node, n_hid, bias=True)
        self.input_n_g = nn.Linear(n_in_node, n_hid, bias=True)

        self.hidden_r_hg = nn.Linear(n_hid, n_hid, bias=False)
        self.hidden_i_hg = nn.Linear(n_hid, n_hid, bias=False)
        self.hidden_h_hg = nn.Linear(n_hid, n_hid, bias=False)

        self.input_r_hg = nn.Linear(n_in_node, n_hid, bias=True)
        self.input_i_hg = nn.Linear(n_in_node, n_hid, bias=True)
        self.input_n_hg = nn.Linear(n_in_node, n_hid, bias=True)

        self.out_fc1 = nn.Linear(n_hid * 2, n_hid)
        self.out_fc2 = nn.Linear(n_hid, n_hid)
        self.W_alpha = nn.Linear(n_hid, self.num_cores)
        self.W_mu = nn.Linear(n_hid, self.num_cores * n_in_node)
        self.W_sigma = nn.Linear(n_hid, self.num_cores * n_in_node)

        self.dropout_prob = do_prob

    def single_step_forward(
        self,
        inputs,
        rel_rec_g,
        rel_send_g,
        rel_type_g,
        rel_rec_hg,
        rel_send_hg,
        rel_type_hg,
        hidden_g,
        hidden_hg,
        var,
        pre_train,
    ):
        receivers = torch.matmul(rel_rec_g, hidden_g)
        senders = torch.matmul(rel_send_g, hidden_g)
        pre_msg = torch.cat([receivers, senders], dim=-1)
        all_msgs = Variable(
            torch.zeros(pre_msg.size(0), pre_msg.size(1), self.msg_out_shape)
        )

        if inputs.is_cuda:
            all_msgs = all_msgs.cuda()

        if self.skip_first_edge_type:
            start_idx = 1
            norm = float(len(self.msg_fc2_g)) - 1.0
        else:
            start_idx = 0
            norm = float(len(self.msg_fc2_g))

        for i in range(start_idx, len(self.msg_fc2_g)):
            msg = torch.tanh(self.msg_fc1_g[i](pre_msg))
            msg = F.dropout(msg, p=self.dropout_prob)
            msg = torch.tanh(self.msg_fc2_g[i](msg))
            msg = msg * rel_type_g[:, :, i : i + 1]
            all_msgs += msg / norm

        agg_msgs = all_msgs.transpose(-2, -1).matmul(rel_rec_g).transpose(-2, -1)
        agg_msgs = agg_msgs.contiguous() / inputs.size(2)

        r = torch.sigmoid(self.input_r_g(inputs) + self.hidden_r_g(agg_msgs))
        i = torch.sigmoid(self.input_i_g(inputs) + self.hidden_i_g(agg_msgs))
        n = torch.tanh(self.input_n_g(inputs) + r * self.hidden_h_g(agg_msgs))
        hidden_g = (1 - i) * n + i * hidden_g

        if pre_train:
            hidden_hg = torch.zeros(hidden_g.size())
            if inputs.is_cuda:
                hidden_hg = hidden_hg.cuda()
        if not pre_train:
            receivers = torch.matmul(rel_rec_hg, hidden_hg)
            senders = torch.matmul(rel_send_hg, hidden_hg)
            pre_msg = torch.cat([receivers, senders], dim=-1)
            all_msgs = Variable(
                torch.zeros(pre_msg.size(0), pre_msg.size(1), self.msg_out_shape)
            )
            if inputs.is_cuda:
                all_msgs = all_msgs.cuda()

            if self.skip_first_edge_type:
                start_idx = 1
                norm = float(len(self.msg_fc2_hg)) - 1.0
            else:
                start_idx = 0
                norm = float(len(self.msg_fc2_hg))

            for i in range(start_idx, len(self.msg_fc2_hg)):
                msg = torch.tanh(self.msg_fc1_hg[i](pre_msg))
                msg = F.dropout(msg, p=self.dropout_prob)
                msg = torch.tanh(self.msg_fc2_hg[i](msg))
                msg = msg * rel_type_hg[:, :, i : i + 1]
                all_msgs += msg / norm

            agg_msgs = all_msgs.transpose(-2, -1).matmul(rel_rec_hg).transpose(-2, -1)
            agg_msgs = agg_msgs.contiguous() / inputs.size(2)

            r = torch.sigmoid(self.input_r_hg(inputs) + self.hidden_r_hg(agg_msgs))
            i = torch.sigmoid(self.input_i_hg(inputs) + self.hidden_i_hg(agg_msgs))
            n = torch.tanh(self.input_n_hg(inputs) + r * self.hidden_h_hg(agg_msgs))
            hidden_hg = (1 - i) * n + i * hidden_hg

        pred = F.dropout(
            F.relu(self.out_fc1(torch.cat((hidden_g, hidden_hg), dim=-1))),
            p=self.dropout_prob,
        )
        pred = F.dropout(F.relu(self.out_fc2(pred)), p=self.dropout_prob)
        alpha = self.W_alpha(pred)
        alpha = F.softmax(alpha, dim=-1)
        mu = self.W_mu(pred)
        mu = mu.reshape(mu.shape[0], mu.shape[1], self.num_cores, self.dim)
        sigma = torch.ones((mu.shape[0], mu.shape[1], self.num_cores, self.dim)) * 1.0
        if inputs.is_cuda:
            sigma = sigma.cuda()

        pred = sample_core(alpha, mu)
        for i in range(mu.shape[2]):
            mu[:, :, i, :] += inputs
        if inputs.is_cuda:
            pred = pred.cuda()

        pred = inputs + pred
        return pred, alpha, mu, sigma, hidden_g, hidden_hg

    def forward(
        self,
        data,
        rel_type_g,
        rel_rec_g,
        rel_send_g,
        rel_type_hg,
        rel_rec_hg,
        rel_send_hg,
        output_steps,
        var,
        pred_steps=1,
        burn_in=False,
        burn_in_steps=1,
        dynamic_graph=False,
        encoder=None,
        temp=None,
        env=None,
        pre_train=False,
    ):
        inputs = data.transpose(1, 2).contiguous()
        time_steps = inputs.size(1)
        hidden_g = Variable(
            torch.zeros(inputs.size(0), inputs.size(2), self.msg_out_shape)
        )
        hidden_hg = Variable(
            torch.zeros(inputs.size(0), inputs.size(2), self.msg_out_shape)
        )
        if inputs.is_cuda:
            hidden_g = hidden_g.cuda()
            hidden_hg = hidden_hg.cuda()

        pred_all = []
        alpha_all = []
        mu_all = []
        sigma_all = []

        for step in range(output_steps):
            if step < burn_in_steps:
                ins = inputs[:, step, :, :]
            else:
                ins = pred_all[step - 1]

            pred, alpha, mu, sigma, hidden_g, hidden_hg = self.single_step_forward(
                ins,
                rel_rec_g,
                rel_send_g,
                rel_type_g,
                rel_rec_hg,
                rel_send_hg,
                rel_type_hg,
                hidden_g,
                hidden_hg,
                var,
                pre_train,
            )
            pred_all.append(pred)
            mu_all.append(mu)
            alpha_all.append(alpha)
            sigma_all.append(sigma)

        preds = torch.stack(pred_all, dim=1)
        alphas = torch.stack(alpha_all, dim=2)
        mus = torch.stack(mu_all, dim=2)
        sigmas = torch.stack(sigma_all, dim=2)

        return preds.transpose(1, 2).contiguous(), alphas, mus, sigmas