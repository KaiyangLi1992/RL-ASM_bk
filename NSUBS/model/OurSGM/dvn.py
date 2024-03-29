from NSUBS.src.utils import OurTimer
import torch
import torch.nn.functional as F
from torch_geometric.data import Data,Batch


from NSUBS.model.OurSGM.config import FLAGS

class DVN(torch.nn.Module):
    def __init__(self, pre_encoder, encoder, gq_egde_encoder,gt_egde_encoder,decoder_policy, decoder_value, norm_li):
        super(DVN, self).__init__()
        self.pre_encoder = pre_encoder
        self.encoder = encoder
        self.gt_egde_encoder =  gt_egde_encoder
        self.gq_egde_encoder =  gq_egde_encoder
        self.decoder_policy = decoder_policy
        self.decoder_value = decoder_value
        self.norm_li = norm_li

    def encoder_wrapper(self,  pyg_data_q,pyg_data_t, u, v_li, u2v_li, nn_map,
        node_mask, cache_target_embeddings):
        timer = None
        if FLAGS.time_analysis:
            timer = OurTimer()

        # Xq = pyg_data_q.x
        # edge_indexq = pyg_data_q.edge_index
        
        # Xt = pyg_data_t.x
        # edge_indext = pyg_data_t.edge_index

        pyg_q_batch = pyg_data_q.batch
        pyg_t_batch = pyg_data_t.batch


        norm_q, norm_t = None, None #self.create_norm_vec(gq, gt)

        if FLAGS.time_analysis:
            timer.time_and_clear(f'create_norm_vec')

        Xq, Xt = \
            self.pre_encoder(pyg_data_q,pyg_data_t, nn_map)
        
        # pyg_data_q = Batch(batch =pyg_q_batch,x=Xq,edge_index=edge_indexq)
        # pyg_data_t = Batch(batch =pyg_t_batch,x=Xt,edge_index=edge_indext)
        pyg_data_q.x = Xq
        pyg_data_t.x = Xt
        
        pyg_data_q = self.gq_egde_encoder(pyg_data_q)
        pyg_data_t = self.gt_egde_encoder(pyg_data_t)

        # Xq = self.norm_li[0](Xq)
        # Xt = self.norm_li[1](Xt)

        if FLAGS.time_analysis:
            timer.time_and_clear(f'pre_encoder')
 
        Xq, Xt = \
            self.encoder(
                pyg_data_q, pyg_data_t,
                nn_map, 
                norm_q, norm_t, u2v_li, node_mask,
                cache_target_embeddings,pyg_q_batch,pyg_t_batch
            )

        if FLAGS.time_analysis:
            timer.time_and_clear(f'encoder')
            # timer.print_durations_log()

        if FLAGS.apply_norm:
            Xq = self.norm_li[2](Xq)
            Xt = self.norm_li[3](Xt)

        if FLAGS.time_analysis:
            timer.time_and_clear(f'apply_norm')
            timer.print_durations_log()
        return Xq, Xt


    def forward(self, pyg_data_q,pyg_data_t, u,v_li,u2v_li,
                nn_map,node_mask,
                cache_embeddings, execute_action, query_tree):

        timer = None
        if FLAGS.time_analysis:
            timer = OurTimer()
  

        # start_time = datetime.now()
        Xq, Xt = \
            self.encoder_wrapper(
                pyg_data_q,pyg_data_t, u, v_li, u2v_li,nn_map,
                node_mask, cache_embeddings
            )
        # end_time = datetime.now()
        # elapsed_time = end_time - start_time
        # print(f"程序执行了 {elapsed_time} ")

        if FLAGS.time_analysis:
            timer.time_and_clear(f'encoder')
        pyg_data_q.x = Xq
        pyg_data_t.x = Xt


        # start_time = datetime.now()
        out_value, g_emb = self.decoder_value(pyg_data_q)
        # end_time = datetime.now()
        # elapsed_time = end_time - start_time
        # print(f"程序执行了 {elapsed_time} ")

        if FLAGS.time_analysis:
            timer.time_and_clear(f'decoder value')

        
        # start_time = datetime.now()
        out_policy, bilin_emb = self.decoder_policy(pyg_data_q, pyg_data_t, u, v_li, g_emb)
        # end_time = datetime.now()
        # elapsed_time = end_time - start_time
        # print(f"程序执行了 {elapsed_time} ")

        if FLAGS.time_analysis:
            timer.time_and_clear(f'decoder policy')
        # out_policy = [x.view(-1) for x in out_policy]
        # out_other = {
        #     'Xq':Xq,
        #     'Xt':Xt,
        #     'g_emb':g_emb,
        #     'bilin_emb':None
        # }

        if FLAGS.time_analysis:
            timer.print_durations_log()
        return out_policy, out_value

    def create_norm_vec(self, gq, gt):
        norm_q = torch.tensor([gq.degree(nid) + 1e-8 for nid in range(gq.number_of_nodes())], dtype=torch.float32, device=FLAGS.device).view(-1,1)
        norm_t = torch.tensor([gt.degree(nid) + 1e-8 for nid in range(gt.number_of_nodes())], dtype=torch.float32, device=FLAGS.device).view(-1,1)
        return norm_q, norm_t