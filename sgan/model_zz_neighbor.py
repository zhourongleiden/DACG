#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 21 08:52:51 2021

@author: zhouzhou
"""

# everything that has been commented out is useless !!!


import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_sequence
from torch.nn.utils.rnn import pad_packed_sequence


def make_mlp(dim_list, activation='relu', batch_norm=False, dropout=0):
    layers = []
    for dim_in, dim_out in zip(dim_list[:-1], dim_list[1:]):
        layers.append(nn.Linear(dim_in, dim_out))
        if batch_norm:
            layers.append(nn.BatchNorm1d(dim_out))
        if activation == 'relu':
            layers.append(nn.ReLU())
        elif activation == 'leakyrelu':
            layers.append(nn.LeakyReLU(negative_slope=0.2))
        if dropout > 0:
            layers.append(nn.Dropout(p=dropout))
    return nn.Sequential(*layers)


class BatchMultiHeadGraphAttention(nn.Module):
    def __init__(self, n_head, f_in, f_out, attn_dropout, bias=True):
        super(BatchMultiHeadGraphAttention, self).__init__()
        self.n_head = n_head
        self.f_in = f_in
        self.f_out = f_out
        self.w = nn.Parameter(torch.Tensor(n_head, f_in, f_out))
        self.a_src = nn.Parameter(torch.Tensor(n_head, f_out, 1))
        self.a_dst = nn.Parameter(torch.Tensor(n_head, f_out, 1))

        self.leaky_relu = nn.LeakyReLU(negative_slope=0.2)
        self.softmax = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(attn_dropout)
        if bias:
            self.bias = nn.Parameter(torch.Tensor(f_out))
            nn.init.constant_(self.bias, 0)
        else:
            self.register_parameter("bias", None)

        nn.init.xavier_uniform_(self.w, gain=1.414)
        nn.init.xavier_uniform_(self.a_src, gain=1.414)
        nn.init.xavier_uniform_(self.a_dst, gain=1.414)

    def forward(self, h, adj):
        bs, n = h.size()[:2]
        h_prime = torch.matmul(h.unsqueeze(1), self.w)
        attn_src = torch.matmul(h_prime, self.a_src)
        attn_dst = torch.matmul(h_prime, self.a_dst)
        attn = attn_src.expand(-1, -1, -1, n) + attn_dst.expand(-1, -1, -1, n).permute(
            0, 1, 3, 2
        )
        attn = self.leaky_relu(attn)
        mask = 1 - adj # bs x 1 x n x n
        attn.data.masked_fill_(mask.bool(), -1e12)        
        attn = self.softmax(attn)
        attn = self.dropout(attn)
        output = torch.matmul(attn, h_prime)
        if self.bias is not None:
            return output + self.bias, attn
        else:
            return output, attn

    def __repr__(self):
        return (
            self.__class__.__name__
            + " ("
            + str(self.n_head)
            + " -> "
            + str(self.f_in)
            + " -> "
            + str(self.f_out)
            + ")"
        )


class GAT(nn.Module):
    def __init__(self, n_units, n_heads, dropout, alpha):
        super(GAT, self).__init__()
        self.n_layer = len(n_units) - 1
        self.dropout = dropout
        self.layer_stack = nn.ModuleList()

        for i in range(self.n_layer):
            f_in = n_units[i] * n_heads[i - 1] if i else n_units[i]
            self.layer_stack.append(
                BatchMultiHeadGraphAttention(
                    n_heads[i], f_in=f_in, f_out=n_units[i + 1], attn_dropout=dropout
                )
            )
    
    def masked_instance_norm(self, x, mask, eps = 1e-5):
        """
        x of shape: [ts (N), batch (L), embedding dims(C)]
        mask of shape: [ts (N), batch (L)]
        每个时间步，所有行人，embedding每个维度作masked_norms
        """
        mask = mask.unsqueeze(-1)  # (N,L,1)
        mean = (torch.sum(x * mask, 1) / torch.where(torch.sum(mask, 1) > 0, torch.sum(mask, 1), 1))

        var_term = ((x - mean.unsqueeze(1).expand_as(x)) * mask)**2  # (N,L,C)
        var = (torch.sum(var_term, 1) / torch.where(torch.sum(mask, 1) > 0, torch.sum(mask, 1), 1))

        mean_reshaped = mean.unsqueeze(1).expand_as(x)  # (N, L, C)
        var_reshaped = var.unsqueeze(1).expand_as(x)    # (N, L, C)
        ins_norm = (x - mean_reshaped) / torch.sqrt(var_reshaped + eps)   # (N, L, C)
        
        return ins_norm

    def forward(self, x, adj, idex):
        bs, n = x.size()[:2]
        for i, gat_layer in enumerate(self.layer_stack):
            x = self.masked_instance_norm(x,idex)#self.norm_list[i](x.permute(0, 2, 1)).permute(0, 2, 1)
            x, attn = gat_layer(x, adj)
            if i + 1 == self.n_layer:
                x = x.squeeze(dim=1)
            else:
                x = F.elu(x.transpose(1, 2).contiguous().view(bs, n, -1))
                x = F.dropout(x, self.dropout, training=self.training)
        else:
            return x, attn.squeeze(1)


class GATEncoder(nn.Module):
    def __init__(self, n_units, n_heads, dropout, alpha):
        super(GATEncoder, self).__init__()
        self.gat_net = GAT(n_units, n_heads, dropout, alpha)

    def forward(self, obs_traj_embedding, seq_start_end, first_history_index):
        graph_embeded_data, attn = [], []
        for start, end in seq_start_end.data:
            obs_traj_embedding_per_scene = obs_traj_embedding[:, start:end, :]
            first_history_index_per_scene = first_history_index[start:end]
 
            # 若该场景有>=2个行人，计算GAT；若该场景只有一个行人（目标node），则直接将其obs_traj_embedding存储至graph_embedding_per_scene
            if end-start > 1:
                # 计算邻接矩阵
                adj = []
                idex_list = []

                for i in range(obs_traj_embedding_per_scene.size(0)):#time steps
                    idex = torch.where(first_history_index_per_scene <= i, 1, 0)
                    idex_list.append(idex.unsqueeze(0))
                    adj.append(torch.outer(idex,idex).unsqueeze(0))

                idex = torch.cat(idex_list, dim=0)
                adj = torch.cat(adj, dim=0).unsqueeze(1)
                graph_embedding_per_scene, attn_per_scene = self.gat_net(obs_traj_embedding_per_scene, adj, idex)
              
            else:
                graph_embedding_per_scene = obs_traj_embedding_per_scene    
            graph_embeded_data.append(graph_embedding_per_scene)        
            attn.append(attn_per_scene)
        graph_embeded_data = torch.cat(graph_embeded_data, dim=1)

        return graph_embeded_data.permute(1,0,2), attn
    
    
class GATEncoder_pred(nn.Module):
    """
    Inputs:
    - pred_lstm_hidden: Tensor of shape (batch*K, decoder_h_dim)
    - seq_start_end: A list of tuples which delimit sequences within batch.
    - first_future_index: Tensor of shape(batch, )
    - t: current timestamp
    Outputs:
    - neighbor_attn_data: Tensor of shape (batch*K, graph_network_out_dims)
    """   
    def __init__(self, n_units, n_heads, dropout, alpha):
        super(GATEncoder_pred, self).__init__()
        self.gat_net = GAT(n_units, n_heads, dropout, alpha)

    def forward(self, pred_lstm_hidden, seq_start_end, batch, K):       
        pred_lstm_hidden = pred_lstm_hidden.view(batch, K, -1).permute(1,0,2) # (K, batch, decoder_h_dim)
      
        neighbor_attn_data = []
        
        for start, end in seq_start_end.data:
            pred_lstm_hidden_per_scene = pred_lstm_hidden[:, start:end, :]
            
            # 若该场景有>=2个行人，计算GAT；若该场景只有一个行人（目标node），则直接将其pred_lstm_hidden_per_scene存储至neighbor_attn_per_scene
            if end-start > 1:
                # 计算邻接矩阵
                adj = []
                idex_list = []
                for i in range(pred_lstm_hidden_per_scene.size(0)): # K(将K视为timestamp)
                    idex = torch.ones(end-start).to(torch.int64).cuda()
                    idex_list.append(idex.unsqueeze(0))
                    adj.append(torch.outer(idex,idex).unsqueeze(0))
                idex = torch.cat(idex_list, dim=0)
                adj = torch.cat(adj, dim=0).unsqueeze(1)
                
                neighbor_attn_per_scene = self.gat_net(pred_lstm_hidden_per_scene, adj, idex)
              
            else:
                neighbor_attn_per_scene = pred_lstm_hidden_per_scene         
            neighbor_attn_data.append(neighbor_attn_per_scene)            
        neighbor_attn_data = torch.cat(neighbor_attn_data, dim=1) # (K, batch, graph_network_out_dims)
        neighbor_attn_data = neighbor_attn_data.permute(1,0,2).contiguous().view(batch*K, -1)

        return neighbor_attn_data    
    

class TrajectoryGenerator(nn.Module):
    def __init__(
        self, obs_len, pred_len, embedding_dim=16, encoder_h_dim=64,
        decoder_h_dim=64, mlp_dim=64, num_layers=1, noise_dim=(8, ),
        dropout=0.0, 
        activation='relu', batch_norm=False, 
        graph_lstm_hidden_size=64,
        graph_network_out_dims=64,
        n_units=0,
        n_heads=0,
        alpha=0,
        teacher_forcing_ratio=0,
        goal_embedding_dim=16,
        goal_h_dim=16,
        device='cpu',
        with_goal=True,
        with_cvae=True,
        neigh_min_his=7,
    ):
        super(TrajectoryGenerator, self).__init__()

        self.obs_len = obs_len
        self.pred_len = pred_len
        self.seq_len = obs_len + pred_len        
        self.mlp_dim = mlp_dim
        self.encoder_h_dim = encoder_h_dim
        self.decoder_h_dim = decoder_h_dim
        self.embedding_dim = embedding_dim
        self.noise_dim = noise_dim
        self.num_layers = num_layers
        self.noise_first_dim = noise_dim[0]
        self.graph_lstm_hidden_size = graph_lstm_hidden_size
        self.n_units = n_units
        self.n_heads = n_heads
        self.alpha = alpha
        self.graph_network_out_dims = graph_network_out_dims
        self.teacher_forcing_ratio = teacher_forcing_ratio
        self.device = device
        self.with_goal = with_goal
        self.with_cvae = with_cvae
        self.neigh_min_his=neigh_min_his
        
        #added
        self.goal_embedding_dim = goal_embedding_dim
        self.goal_h_dim = goal_h_dim
        
        #functions(main)
        self.spatial_embedding_X = nn.Sequential(nn.Linear(6, self.embedding_dim), nn.ReLU())
        self.spatial_embedding_Y = nn.Sequential(nn.Linear(2, self.embedding_dim), nn.ReLU())
        
        self.traj_lstm_model_X = nn.LSTM(self.embedding_dim, self.encoder_h_dim, batch_first = True)
        self.traj_lstm_model_Y = nn.LSTM(self.embedding_dim, self.encoder_h_dim, batch_first = True)
        
        if self.neigh_min_his != -1:
            self.graph_lstm_model = nn.LSTM(self.graph_network_out_dims, self.graph_lstm_hidden_size, batch_first = True)
            self.gatencoder = GATEncoder(self.n_units, self.n_heads, dropout, alpha)
        
        self.pred_lstm_model = nn.LSTMCell(self.embedding_dim, self.decoder_h_dim)
        
        if self.with_goal:
            self.pred_hidden2pos = nn.Sequential(nn.Linear(self.decoder_h_dim + self.goal_h_dim, 2))
        else:
            self.pred_hidden2pos = nn.Sequential(nn.Linear(self.decoder_h_dim, 2))

        if self.neigh_min_his != -1:
            self.input_dim = self.encoder_h_dim + self.graph_lstm_hidden_size
        else:
            self.input_dim = self.encoder_h_dim 
        mlp_decoder_context_dims = [self.input_dim, self.mlp_dim, self.decoder_h_dim - self.noise_first_dim]
        self.mlp_decoder_context = make_mlp(mlp_decoder_context_dims, activation, batch_norm, dropout)
        
        #functions(main_added) 
        self.goal_hidden_generator = nn.Sequential(nn.Linear(self.decoder_h_dim, self.goal_h_dim),
                                    nn.ReLU())          
        self.goal_input_generator = nn.Sequential(nn.Linear(self.decoder_h_dim, self.goal_embedding_dim),
                                    nn.ReLU())     
        
        self.goal_lstm_model = nn.LSTMCell(self.goal_embedding_dim, self.goal_h_dim)
        self.goal_hidden2input = nn.Sequential(nn.Linear(self.goal_h_dim, self.goal_embedding_dim), nn.ReLU())
        
        self.goal_input2pos = nn.Sequential(nn.Linear(self.goal_h_dim, 2)) 
        self.goal_pos2input = nn.Sequential(nn.Linear(2,self.goal_h_dim), nn.ReLU())    
        
        self.goal_attn = nn.Sequential(nn.Linear(self.goal_h_dim, 1), nn.ReLU())
        
        #if self.with_goal:
        #    self.decoder_X = nn.Sequential(nn.Linear(self.goal_h_dim, self.embedding_dim), nn.ReLU())
        #    self.decoder_Y = nn.Sequential(nn.Linear(2, self.embedding_dim), nn.ReLU())

        #else:
        self.decoder_X = nn.Sequential(nn.Linear(6, self.embedding_dim), nn.ReLU())
        self.decoder_Y = nn.Sequential(nn.Linear(2, self.embedding_dim), nn.ReLU())
        
        self.pred_lstm_hidden_generator = nn.Sequential(nn.Linear(self.goal_h_dim, self.decoder_h_dim), nn.ReLU())
                      
        #functions(noise)
        self.latent_encoder_X = nn.Sequential(nn.Linear(self.encoder_h_dim, 128),
                                    nn.ReLU(),
                                    nn.Linear(128, 64),
                                    nn.ReLU(),
                                    nn.Linear(64, self.noise_first_dim*2),
                                    nn.ReLU())
        
        self.latent_encoder_XY = nn.Sequential(nn.Linear(self.encoder_h_dim, 128),
                                    nn.ReLU(),
                                    nn.Linear(128, 64),
                                    nn.ReLU(),
                                    nn.Linear(64, self.noise_first_dim*2),
                                    nn.ReLU())     
    
   
    def add_noise(self, noise_input, lstm_h_t, lstm_h_t_Y, K):
        """
        Inputs:
        - noise_input: Tensor of shape (batch, decoder_h_dim - noise_first_dim)
        Outputs:
        - decoder_h: Tensor of shape (batch, decoder_h_dim)
        - KLD: Tensor of shape (batch,), with KLD loss
        """
        if self.with_cvae:
            # 1. sample z from piror
            z_mu_logvar_p = self.latent_encoder_X(lstm_h_t)
            z_mu_p = z_mu_logvar_p[:, :self.noise_first_dim]
            z_logvar_p = z_mu_logvar_p[:, self.noise_first_dim:]   
            
            # 2. sample z from posterior, for training only
            if self.training:
                z_mu_logvar_q = self.latent_encoder_XY(lstm_h_t_Y)
                z_mu_q = z_mu_logvar_q[:, :self.noise_first_dim]
                z_logvar_q = z_mu_logvar_q[:, self.noise_first_dim:]
                Z_mu = z_mu_q
                Z_logvar = z_logvar_q            
                
                # 3. compute KLD(q_z_xy||p_z_x)
                KLD = 0.5 * ((z_logvar_q.exp()/z_logvar_p.exp()) + \
                            (z_mu_p - z_mu_q).pow(2)/z_logvar_p.exp() - \
                            1 + \
                            (z_logvar_p - z_logvar_q))
                KLD = KLD.sum(dim=-1)
                KLD = torch.clamp(KLD, min=0.001)
            else:
                Z_mu = z_mu_p
                Z_logvar = z_logvar_p
                KLD = 0.0
            
            # 4. Draw sample from (Z_mu, Z_logvar)
            e_samples = torch.randn(noise_input.size(0), K, self.noise_first_dim).to(self.device)
            Z_std = torch.exp(0.5 * Z_logvar)
            Z = Z_mu.unsqueeze(1).repeat(1, K, 1) + e_samples * Z_std.unsqueeze(1).repeat(1, K, 1)
        else:
            Z = torch.randn(noise_input.size(0), K, self.noise_first_dim).to(self.device)
        # 拼接Z与_input得到pred_lstm_hidden
        noise_input = noise_input.unsqueeze(1).repeat(1, K, 1)
        decoder_h = torch.cat([noise_input, Z], dim=2)

        return decoder_h, KLD
    
    
    def forward(self, obs_traj_rel, pred_traj_gt_rel, seq_start_end, first_history_index, K):
        
        """
        Inputs:
        - obs_traj_rel: Tensor of shape (batch, obs_len, 6)
        - pred_traj_gt_rel: Tensor of shape (batch, pred_len, 2)        
        - seq_start_end: A list of tuples which delimit sequences within batch.
        - first_history_index: Tensor of shape (batch,), which indicates the start observation time of each pedestrian in obs_traj_rel
        Output:
        - pred_traj_fake_rel: Tensor of shape (batch, K, pred_len, 2)
        """
       
        batch = obs_traj_rel.size(0)
       
        ###################################################################################################################

        # 每个行人obs阶段t时刻的坐标，依次通过M-LSTM得到对应时刻隐状态
        pad_list = obs_traj_rel.view(-1,6)[~torch.any(obs_traj_rel.view(-1,6).isnan(),dim=1)] 
        length_per_batch = self.obs_len - first_history_index
        
        # embed and covert back to pad_list
        obs_traj_embedding = self.spatial_embedding_X(pad_list)
        pad_list = torch.split(obs_traj_embedding, length_per_batch.tolist())
               
        # 每个行人obs阶段t时刻的坐标，依次通过M-LSTM得到对应时刻隐状态
        packed_seqs = pack_sequence(pad_list, enforce_sorted=False) 
        packed_output, state = self.traj_lstm_model_X(packed_seqs)
        # pad zeros to the end so that the last non zero value 
        traj_lstm_h_t, _ = pad_packed_sequence(packed_output, batch_first=True, total_length=self.obs_len)      
        
        # 调整padding的位置
        for i in range(batch):
            temp = traj_lstm_h_t[i]
            if first_history_index[i] == 0:
                continue
            else:    
                temp = torch.cat((torch.zeros(first_history_index[i],self.encoder_h_dim).to(self.device),temp[0:(self.obs_len-first_history_index[i]),:]),dim=0)
            traj_lstm_h_t[i] = temp
            
        
        #traj_lstm_h_t = F.dropout(traj_lstm_h_t,
        #                    p=0.25,
        #                    training=self.training)
        
        ###################################################################################################################          
        if self.neigh_min_his != -1:
            #obs阶段每个行人t时刻隐状态，通过GAT网络得到每个时刻G-LSTM的输入
            graph_lstm_input, attn = self.gatencoder(traj_lstm_h_t.permute(1,0,2), seq_start_end, first_history_index)         
                     
            pad_list_graph_lstm = []
            for i in range(batch):
                pad_list_graph_lstm.append(graph_lstm_input[i,first_history_index[i]:self.obs_len])        
            
            # 每个行人t时刻的attention结果，依次通过G-LSTM得到对应时刻隐状态
            packed_seqs_graph_lstm = pack_sequence(pad_list_graph_lstm, enforce_sorted=False) 
            packed_output_graph_lstm, _ = self.graph_lstm_model(packed_seqs_graph_lstm)
            graph_lstm_h_t, _ = pad_packed_sequence(packed_output_graph_lstm, batch_first=True, total_length=self.obs_len)        
            
            # 调整padding的位置
            for i in range(batch):
                temp = graph_lstm_h_t[i]
                if first_history_index[i] == 0:
                    continue
                else:    
                    temp = torch.cat((torch.zeros(first_history_index[i],self.graph_lstm_hidden_size).to(self.device),temp[0:(self.obs_len-first_history_index[i]),:]),dim=0)#
                graph_lstm_h_t[i] = temp    
     
            ###################################################################################################################
            
            #M-LSTM输出和G-LSTM输出concat
            encoded_before_noise_hidden = torch.cat((traj_lstm_h_t[:,-1,:], graph_lstm_h_t[:,-1,:]), dim=1)
        else:
            attn = 0
            encoded_before_noise_hidden = traj_lstm_h_t[:,-1,:]
        
        ###################################################################################################################
    
        # 每个行人pred阶段t时刻的gt坐标，依次通过R-LSTM得到对应时刻隐状态
        pred_traj_gt_embedding = self.spatial_embedding_Y(pred_traj_gt_rel.view(-1, 2))
        pred_traj_gt_embedding = pred_traj_gt_embedding.view(batch, -1, self.embedding_dim)
        traj_lstm_h_t_Y, _ = self.traj_lstm_model_Y(pred_traj_gt_embedding,state) 
        
        #traj_lstm_h_t_Y = F.dropout(traj_lstm_h_t_Y,
        #                    p=0.25,
        #                    training=self.training)    
        
        ###################################################################################################################
      
        # 加入噪声        
        noise_input = self.mlp_decoder_context(encoded_before_noise_hidden)#batch, decoder_h_dim - noise_first_dim
        pred_lstm_hidden, KLD = self.add_noise(noise_input, traj_lstm_h_t[:,-1,:], traj_lstm_h_t_Y[:,-1,:], K)#batch, K, decoder_h_dim
        
        ###################################################################################################################
        if self.with_goal:         
            # stepwise goals init
            goals_lstm_h_t = self.goal_hidden_generator(pred_lstm_hidden.view(-1, self.decoder_h_dim))#batch*K, goal_h_dim
            goals_lstm_c_t = torch.zeros_like(goals_lstm_h_t)
            goals_input = self.goal_input_generator(pred_lstm_hidden.view(-1, self.decoder_h_dim))#batch*K, goal_embedding_dim
            
            # generate stepwise goals
            goals_embedded=[] 
            for i in range(self.pred_len): 
                goals_lstm_h_t, goals_lstm_c_t = self.goal_lstm_model(goals_input, (goals_lstm_h_t, goals_lstm_c_t))
                #goals_output = goals_lstm_h_t
                goals_input = self.goal_hidden2input(goals_lstm_h_t)
                goals_embedded += [goals_input]   #4 replace goals_lstm_h_t by goals_input
            goals_embedded = goals_embedded[::-1]
            goals_embedded = torch.stack(goals_embedded).permute(1,0,2)#batch*K,12,goal_h_dim
            
            # goal loss
            goal_loss = (pred_traj_gt_embedding[:,-1,:].unsqueeze(1).repeat(1, K, 1) - goals_embedded[:,-1,:].view(batch,K,self.goal_h_dim))**2
            goal_loss = goal_loss.sum(dim=2)
        else:
            goal_loss = torch.randn(batch,K).to(self.device)

        ###################################################################################################################           
        # 预测部分
        pred_traj_rel = [] 
        '''
        if self.with_goal: 
            #
            #output_embedding = self.decoder_X(goals_embedded[:,-1,:])                            
            #pred_lstm_h_t = self.pred_lstm_hidden_generator(goals_embedded[:,-1,:])    
            # 
            output_embedding = self.decoder_X(obs_traj_rel[:,-1,:]) 
            output_embedding = output_embedding.unsqueeze(1).repeat(1, K, 1).view(-1, self.embedding_dim)##batch*K, embedding_dim
            pred_lstm_h_t = pred_lstm_hidden.view(-1, self.decoder_h_dim)
        else:
        '''
        output_embedding = self.decoder_X(obs_traj_rel[:,-1,:]) 
        output_embedding = output_embedding.unsqueeze(1).repeat(1, K, 1).view(-1, self.embedding_dim)##batch*K, embedding_dim
        pred_lstm_h_t = pred_lstm_hidden.view(-1, self.decoder_h_dim)
        
        pred_lstm_c_t = torch.zeros_like(pred_lstm_h_t)                                      
                                                                                       
        for i in range(self.pred_len):                                                       
            pred_traj_embedding = output_embedding                                           
            #依次过LSTM，更新output
            pred_lstm_h_t, pred_lstm_c_t = self.pred_lstm_model(pred_traj_embedding, (pred_lstm_h_t, pred_lstm_c_t))           
            if self.with_goal: 
                #goals attention
                #goals_embedded_useful = goals_embedded[:,0:self.pred_len-i,:]
                goals_embedded_useful = goals_embedded[:,i:self.pred_len,:]
                goals_attn = self.goal_attn(torch.tanh(goals_embedded_useful)).squeeze(-1)
                goals_attn = F.softmax(goals_attn, dim =1).unsqueeze(1)
                goals_embedded_useful  = torch.bmm(goals_attn,goals_embedded_useful).squeeze(1)
                output = self.pred_hidden2pos(torch.cat((pred_lstm_h_t, goals_embedded_useful), dim = 1))
            else:
                output = self.pred_hidden2pos(pred_lstm_h_t)           
            pred_traj_rel += [output]
            #每个行人t时刻的相对坐标预测结果embedding
            output_embedding = self.decoder_Y(output)#batch*K,embedding_dim  
        #
        #if self.with_goal:
        #    pred_traj_rel = pred_traj_rel[::-1]
        pred_traj_fake_rel = torch.stack(pred_traj_rel).permute(1,0,2)#batch*K,12,2
        pred_traj_fake_rel = pred_traj_fake_rel.view(batch,K,self.pred_len,2)  
            
        return pred_traj_fake_rel, KLD, goal_loss, attn
    
    

class TrajectoryDiscriminator(nn.Module):
    def __init__(
        self, obs_len, pred_len, embedding_dim=16, h_dim=48, mlp_dim=64,
        activation='relu', batch_norm=False, dropout=0.0,
    ):
        super(TrajectoryDiscriminator, self).__init__()

        self.obs_len = obs_len
        self.pred_len = pred_len
        self.seq_len = obs_len + pred_len
        self.mlp_dim = mlp_dim
        self.encoder_h_dim_d = h_dim
        self.embedding_dim = embedding_dim

        #function
        self.spatial_embedding_X = nn.Sequential(nn.Linear(6, self.embedding_dim), nn.ReLU())
        self.spatial_embedding_Y = nn.Sequential(nn.Linear(2, self.embedding_dim), nn.ReLU())          
        self.lstm_model = nn.LSTM(self.embedding_dim, self.encoder_h_dim_d, batch_first = True)
        
        #real_classifier_dims = [self.encoder_h_dim_d+self.embedding_dim, mlp_dim, 1]        
        real_classifier_dims = [self.encoder_h_dim_d, mlp_dim, 1] 
        self.real_classifier = make_mlp(
            real_classifier_dims,
            activation='relu',
            batch_norm=batch_norm,
            dropout=0.0
        )


    def forward(self, obs_traj_rel, pred_traj_rel, pred_traj_rel_gt, first_history_index, K):
        """
        Inputs:
        - obs_traj_rel: Tensor of shape (batch, obs_len , 6)
        - pred_traj_rel: Tensor of shape (batch, K, pred_len , 2)
        - pred_traj_rel_gt: Tensor of shape (batch, pred_len , 2)
        - first_history_index: Tensor of shape(batch, )
        - first_future_index: Tensor of shape(batch, )
        Output:
        - scores: Tensor of shape (batch, K) with real/fake scores
        """

        batch = obs_traj_rel.size(0)
        
        pad_list = obs_traj_rel.view(-1,6)[~torch.any(obs_traj_rel.view(-1,6).isnan(),dim=1)] 
        length_per_batch = self.obs_len - first_history_index
        
        # embed and covert back to pad_list
        obs_traj_embedding = self.spatial_embedding_X(pad_list)
        pad_list = torch.split(obs_traj_embedding, length_per_batch.tolist())
               
        # 每个行人obs阶段t时刻的坐标，依次通过M-LSTM得到对应时刻隐状态
        packed_seqs = pack_sequence(pad_list, enforce_sorted=False) 
        packed_output, state = self.lstm_model(packed_seqs)
        
        state0 = state[0].squeeze(0).unsqueeze(1).repeat(1,K,1).view(-1,self.encoder_h_dim_d).unsqueeze(0)
        state1 = state[1].squeeze(0).unsqueeze(1).repeat(1,K,1).view(-1,self.encoder_h_dim_d).unsqueeze(0)
        state = (state0,state1)


        traj_embedding = self.spatial_embedding_Y(pred_traj_rel.contiguous().view(-1, 2))
        traj_embedding = traj_embedding.view(batch*K, -1, self.embedding_dim)
        output, _ = self.lstm_model(traj_embedding, state)
        
        #output = output.view(batch, K, self.pred_len, self.encoder_h_dim_d)
        #goal_embedding = self.spatial_embedding_Y(pred_traj_rel_gt[:,-1,:])

        #取最终LSTMcell输出结果
        #classifier_input = torch.cat((output[:,-1,:], goal_embedding.unsqueeze(1).repeat(1, K, 1).view(batch*K, -1)), dim=1)
        classifier_input = output[:,-1,:]
        #输出D打分值
        scores = self.real_classifier(classifier_input).view(batch, K)
        
        return scores
