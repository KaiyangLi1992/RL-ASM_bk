import torch
import torch.nn as nn
from yacs.config import CfgNode as CN
from NSUBS.model.OurSGM.config import FLAGS
from config_loader import Config
from graphgps.encoder.laplace_pos_encoder import LapPENodeEncoder
def create_preencoder(d_in_raw, d_in):
    if FLAGS.dvn_config['preencoder']['type'] == 'concat+mlp':
        preencoder = PreEncoderConcatSelectedOneHotAndMLP(d_in_raw, d_in)
    elif FLAGS.dvn_config['preencoder']['type'] == 'mlp':
        preencoder = PreEncoderMLP(d_in_raw, d_in)
    else:
        assert False
    return preencoder

def get_one_hot_labelling(N, sel_nodes):
    cfg = CN.load_cfg(Config.get_config())
    sel_nodes_one_hot = torch.zeros(N, dtype=torch.float32, device=FLAGS.device)
    sel_nodes_one_hot[sel_nodes] = 1
    sel_nodes_one_hot = torch.stack((sel_nodes_one_hot, 1-sel_nodes_one_hot), dim=-1)
    return sel_nodes_one_hot

class PreEncoderConcatSelectedOneHotAndMLP(torch.nn.Module):
    def __init__(self, dim_in, dim_out):
        super(PreEncoderConcatSelectedOneHotAndMLP, self).__init__()

        
        cfg = CN.load_cfg(Config.get_config())
        self.lap_encoder_q = LapPENodeEncoder(cfg.gnn.dim_inner)
        self.lap_encoder_t  = LapPENodeEncoder(cfg.gnn.dim_inner)


        if dim_in + 2 != dim_out:
            if FLAGS.d_enc == 128:
                self.mlp_q = nn.Linear(dim_in + 2, dim_out-64)
                self.mlp_t = nn.Linear(dim_in + 2, dim_out-64)
                self.mlp_RWSEq = nn.Linear(20, 64)
                self.mlp_RWSEt = nn.Linear(20, 64)
            if FLAGS.d_enc == 32:
                self.mlp_q = nn.Linear(dim_in + 2, dim_out-16)
                self.mlp_t = nn.Linear(dim_in + 2, dim_out-16)
                # self.mlp_RWSEq = nn.Linear(20, 16)
                # self.mlp_RWSEt = nn.Linear(20, 16)
                

            if FLAGS.d_enc == 16:
                self.mlp_q = nn.Linear(dim_in + 2, 12)
                self.mlp_t = nn.Linear(dim_in + 2, 12)
                self.mlp_RWSEq = nn.Linear(20, 4)
                self.mlp_RWSEt = nn.Linear(20, 4)
        else:
            self.mlp_q = self.mlp_t = lambda x: x

    def forward(self, pyg_q_batch, pyg_t_batch, nn_map):
        Xq = pyg_q_batch.x
        Xt = pyg_t_batch.x
        
        selected_one_hot_q = get_one_hot_labelling(Xq.shape[0], list(nn_map.keys()))
        selected_one_hot_t = get_one_hot_labelling(Xt.shape[0], list(nn_map.values()))
        Xq = torch.cat((Xq, selected_one_hot_q), dim=1)
        Xt = torch.cat((Xt, selected_one_hot_t), dim=1)
        Xq = self.mlp_q(Xq)
        Xt = self.mlp_t(Xt)
        # RWSEq = self.mlp_RWSEq(RWSEq)
        # RWSEt = self.mlp_RWSEt(RWSEt)

        LapPE_q = self.lap_encoder_q(pyg_q_batch)
        LapPE_t = self.lap_encoder_t(pyg_t_batch)
        
        Xq = torch.cat((Xq, LapPE_q), dim=1)
        Xt = torch.cat((Xt, LapPE_t), dim=1)
        return Xq, Xt

class PreEncoderMLP(torch.nn.Module):
    def __init__(self, dim_in, dim_out):
        super(PreEncoderMLP, self).__init__()
        self.mlp_q = nn.Linear(dim_in, dim_out)
        self.mlp_t = nn.Linear(dim_in, dim_out)

    def forward(self, Xq, Xt, *args):
        Xq = self.mlp_q(Xq)
        Xt = self.mlp_t(Xt)
        return Xq, Xt