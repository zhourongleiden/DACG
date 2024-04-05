#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec  9 16:34:45 2021

@author: zhouzhou
"""

##############################  load packages  ################################
import gc
import logging
import os
import sys
import numpy as np
import dill
import json
import random
from tqdm import tqdm
from collections import defaultdict
import torch
import torch.nn as nn
import torch.optim as optim
from torch import utils

from sgan.losses import gan_g_loss, gan_d_loss, l2_loss
from sgan.losses import displacement_error, final_displacement_error, collision_num
from sgan.model_zz_neighbor import TrajectoryGenerator, TrajectoryDiscriminator

sys.path.append('./trajectron') 

from argument_parser_neighbor import args
from model.dataset import EnvironmentDataset, collate

###########################  predifined functions  ############################
def init_weights(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        nn.init.kaiming_normal_(m.weight)


def restore(data):
    """
    In case we dilled some structures to share between multiple process this function will restore them.
    If the data input are not bytes we assume it was not dilled in the first place

    :param data: Possibly dilled data structure
    :return: Un-dilled data structure
    """
    if type(data) is bytes:
        return dill.loads(data)
    return data


#################################  MAIN  ######################################

def main(args,hyperparams_trajectron):     
    
#####################  part 1 trajectronpp data loader  #######################   
    train_scenes = []
    train_data_path = os.path.join(args.data_dir, args.train_data_dict)
    with open(train_data_path, 'rb') as f:
        train_env = dill.load(f, encoding='latin1')
    
    for attention_radius_override in args.override_attention_radius:
        node_type1, node_type2, attention_radius = attention_radius_override.split('')
        train_env.attention_radius[(node_type1, node_type2)] = float(attention_radius)
    
    if train_env.robot_type is None and hyperparams_trajectron['incl_robot_node']:
        train_env.robot_type = train_env.NodeType[0]
        for scene in train_env.scenes:
            scene.add_robot_from_nodes(train_env.robot_type)
   
    train_scenes = train_env.scenes
    #主要包含每个scene每个t包含的有效node
    train_dataset = EnvironmentDataset(train_env,
                                       hyperparams_trajectron['state'],
                                       hyperparams_trajectron['pred_state'],
                                       scene_freq_mult=hyperparams_trajectron['scene_freq_mult_train'],
                                       node_freq_mult=hyperparams_trajectron['node_freq_mult_train'],
                                       hyperparams=hyperparams_trajectron,
                                       min_history_timesteps=hyperparams_trajectron['minimum_history_length'],
                                       min_future_timesteps=hyperparams_trajectron['prediction_horizon'],
                                       return_robot= not hyperparams_trajectron['incl_robot_node'])
    train_data_loader = dict()
    for node_type_data_set in train_dataset:
        node_type_dataloader = utils.data.DataLoader(node_type_data_set,
                                                     collate_fn=collate,
                                                     pin_memory=False if args.device=='cpu' else True,
                                                     batch_size=args.batch_size,
                                                     shuffle=True,
                                                     num_workers=args.preprocess_workers)
        train_data_loader[node_type_data_set.node_type] = node_type_dataloader
    
    print(f"Loaded training data from {train_data_path}")
    
    #Offline Calculate Scene Graph
    if hyperparams_trajectron['offline_scene_graph'] == 'yes':
        print("Offline calculating scene graphs")
        for i, scene in enumerate(train_scenes):
            scene.calculate_scene_graph(train_env.attention_radius,
                                        hyperparams_trajectron['edge_addition_filter'],
                                        hyperparams_trajectron['edge_removal_filter'])
            print(f"Created Scene Graph for Training Scene {i}")

#####################  part 1.1 trajectronpp data loader  #######################   
    
    test_data_path = os.path.join(args.data_dir, args.test_data_dict)
    with open(test_data_path, 'rb') as f:
        test_env = dill.load(f, encoding='latin1') 
        
    if 'override_attention_radius' in hyperparams_trajectron:
        for attention_radius_override in hyperparams_trajectron['override_attention_radius']:
            node_type1, node_type2, attention_radius = attention_radius_override.split(' ')
            test_env.attention_radius[(node_type1, node_type2)] = float(attention_radius)
    
    test_scenes = test_env.scenes
    # 主要包含每个scene每个t包含的有效node
    test_dataset = EnvironmentDataset(test_env,
                                       hyperparams_trajectron['state'],
                                       hyperparams_trajectron['pred_state'],
                                       scene_freq_mult=hyperparams_trajectron['scene_freq_mult_train'],
                                       node_freq_mult=hyperparams_trajectron['node_freq_mult_train'],
                                       hyperparams=hyperparams_trajectron,
                                       min_history_timesteps= 7,
                                       min_future_timesteps=hyperparams_trajectron['prediction_horizon'],
                                       return_robot= not hyperparams_trajectron['incl_robot_node'])
    
    test_data_loader = dict()
    for node_type_data_set in test_dataset:
        node_type_dataloader = utils.data.DataLoader(node_type_data_set,
                                                     collate_fn=collate,
                                                     pin_memory=False if args.device=='cpu' else True,
                                                     batch_size=args.batch_size,
                                                     shuffle=False,
                                                     num_workers=args.preprocess_workers)
        test_data_loader[node_type_data_set.node_type] = node_type_dataloader
    
    print(f"Loaded training data from {test_data_path}")        
     
    #Offline Calculate Scene Graph
    if hyperparams_trajectron['offline_scene_graph'] == 'yes':
        print("Offline calculating scene graphs")
        for i, scene in enumerate(test_scenes):
            scene.calculate_scene_graph(test_env.attention_radius,
                                        hyperparams_trajectron['edge_addition_filter'],
                                        hyperparams_trajectron['edge_removal_filter'])
            print(f"Created Scene Graph for Training Scene {i}")       

#############################  part 2 gat-cvae-gan init ###############################   
    
    n_units = ([args.encoder_h_dim_g] + [int(x) for x in args.hidden_units.strip().split(",")] + [args.graph_network_out_dims])
    n_heads = [int(x) for x in args.heads.strip().split(",")]    

    generator = TrajectoryGenerator(
        obs_len=args.obs_len,
        pred_len=args.pred_len,
        embedding_dim=args.embedding_dim,
        encoder_h_dim=args.encoder_h_dim_g,
        decoder_h_dim=args.decoder_h_dim_g,
        mlp_dim=args.mlp_dim,
        num_layers=args.num_layers,
        noise_dim=args.noise_dim,
        dropout=args.dropout,
        batch_norm=args.batch_norm,
        graph_lstm_hidden_size=args.graph_lstm_hidden_size,
        graph_network_out_dims=args.graph_network_out_dims,
        n_units=n_units,
        n_heads=n_heads,
        alpha=args.alpha,
        teacher_forcing_ratio=args.teacher_forcing_ratio,
        goal_embedding_dim=args.goal_embedding_dim,
        goal_h_dim=args.goal_h_dim,
        device=args.device,
        with_goal=args.with_goal,
        with_cvae=args.with_cvae,
        neigh_min_his=args.neighbor_min_history
        )

    generator.apply(init_weights)
    generator.to(args.device).train()
    logger.info('Here is the generator:')
    logger.info(generator)

    discriminator = TrajectoryDiscriminator(
        obs_len=args.obs_len,
        pred_len=args.pred_len,
        embedding_dim=args.embedding_dim,
        h_dim=args.encoder_h_dim_d,
        mlp_dim=args.mlp_dim,
        dropout=args.dropout,
        batch_norm=args.batch_norm,
        )

    discriminator.apply(init_weights)
    discriminator.to(args.device).train()
    logger.info('Here is the discriminator:')
    logger.info(discriminator)

    g_loss_fn = gan_g_loss
    d_loss_fn = gan_d_loss

    optimizer_g = optim.Adam(generator.parameters(), lr=args.g_learning_rate)
    optimizer_d = optim.SGD(discriminator.parameters(), lr=args.d_learning_rate)
    lr_scheduler_g= optim.lr_scheduler.ExponentialLR(optimizer_g, gamma=0.99)
    lr_scheduler_d = optim.lr_scheduler.ExponentialLR(optimizer_d, gamma=0.99)      
    
    # Starting from scratch, so initialize checkpoint data structure
    checkpoint = {
        'args': args.__dict__,
        'hp': hyperparams_trajectron,
        'G_losses': defaultdict(list),
        'D_losses': defaultdict(list),
        'losses_ts': [],
        'g_state': None,
        'g_optim_state': None,
        'd_state': None,
        'd_optim_state': None,
    }
    

##############################  part 3 train  #################################      
        
    for t in range(0, args.train_epochs):
        gc.collect()
        d_steps_left = args.d_steps
        g_steps_left = args.g_steps
        train_dataset.augment = args.augment
        
        for node_type, data_loader in train_data_loader.items():
            pbar = tqdm(data_loader, ncols=100)
            
            for target_nodes in pbar:        
                # process data for batch
                first_history_index_nodes = target_nodes[0]
                target_nodes_x = target_nodes[3]
                target_nodes_y = target_nodes[4]
                first_history_index_neighbors = restore(target_nodes[5])
                first_future_index_neighbors = restore(target_nodes[6])
                target_nodes_neighbors_x = restore(target_nodes[7])
                target_nodes_neighbors_y = restore(target_nodes[8])
                
                seq_start = 0
                all_pedestrians_x = []
                all_pedestrians_y = []
                first_history_index = []
                seq_start_end = []
                
                total_neighbors = 0
                
                for idex in range(0,target_nodes_x.shape[0]):
                    first_history_index_node = first_history_index_nodes[idex]
                    target_node_x = target_nodes_x[idex].unsqueeze(0)
                    target_node_y = target_nodes_y[idex].unsqueeze(0)
                    
                    first_history_index_neighbor = first_history_index_neighbors[(node_type,node_type)][idex][0]
                    first_future_index_neighbor = first_future_index_neighbors[(node_type,node_type)][idex][0]
                    target_node_neighbors_x = target_nodes_neighbors_x[(node_type,node_type)][idex]
                    target_node_neighbors_y = target_nodes_neighbors_y[(node_type,node_type)][idex]
                    
                    #####delete neighbors that do not have at least 7-args.neighbor_min_history+1 steps ##########################################################################################
                    if len(target_node_neighbors_x) >= 1: 
                        mask_neighbors = (first_history_index_neighbor <= args.neighbor_min_history).nonzero().squeeze(-1) #neighbor_min_history : index
                        first_history_index_neighbor = first_history_index_neighbor[mask_neighbors]
                        first_future_index_neighbor = first_future_index_neighbor[mask_neighbors]
                        
                        if mask_neighbors.size(0) == 0:
                            target_node_neighbors_x = list()
                            target_node_neighbors_y = list()
                        else:                           
                            target_node_neighbors_x = torch.stack(target_node_neighbors_x) if len(target_node_neighbors_x)>=2 else target_node_neighbors_x[0].unsqueeze(0)
                            target_node_neighbors_y = torch.stack(target_node_neighbors_y) if len(target_node_neighbors_x)>=2 else target_node_neighbors_y[0].unsqueeze(0)
                            
                            target_node_neighbors_x = target_node_neighbors_x[mask_neighbors]
                            target_node_neighbors_y = target_node_neighbors_y[mask_neighbors]
                            
                            target_node_neighbors_x = list(target_node_neighbors_x.chunk(target_node_neighbors_x.size(0)))            
                            target_node_neighbors_y = list(target_node_neighbors_y.chunk(target_node_neighbors_y.size(0)))

                    ######################################################################################################################################################
                    
                    total_neighbors += len(target_node_neighbors_x)
                    
                    target_node_x_with_neighbors = [target_node_x] + target_node_neighbors_x
                    target_node_x_with_neighbors = torch.cat(target_node_x_with_neighbors, dim=0)
                    
                    target_node_y_with_neighbors = [target_node_y] + target_node_neighbors_y
                    target_node_y_with_neighbors = torch.cat(target_node_y_with_neighbors, dim=0)
                    
                    all_pedestrians_x.append(target_node_x_with_neighbors)
                    all_pedestrians_y.append(target_node_y_with_neighbors)
                    
                    seq_start_end.append((seq_start, seq_start + target_node_x_with_neighbors.size(0)))
                    seq_start = seq_start + target_node_x_with_neighbors.size(0)
                    
                    first_history_index.append(first_history_index_node.view(1))
                    if len(target_node_neighbors_x)>=1:
                        first_history_index.append(first_history_index_neighbor)
                        
                all_pedestrians_x = torch.cat(all_pedestrians_x).to(args.device)
                all_pedestrians_y = torch.cat(all_pedestrians_y).to(args.device)
                seq_start_end = torch.Tensor(seq_start_end).to(torch.int64).to(args.device)
                first_history_index = torch.cat(first_history_index).to(torch.int64).to(args.device)
                
                
                #build batch
                batch = (all_pedestrians_x, all_pedestrians_y, seq_start_end, first_history_index)                         
                                                
                #start training
                if d_steps_left > 0:
                    losses_d = discriminator_step(args, batch, generator, discriminator, d_loss_fn, optimizer_d)
                    d_steps_left -= 1
                elif g_steps_left > 0:
                    losses_g = generator_step(args, batch, generator, discriminator, g_loss_fn, optimizer_g)
                    g_steps_left -= 1       
    
                # Skip the rest if we are not at the end of an iteration
                if d_steps_left > 0 or g_steps_left > 0:
                    continue
   
                d_steps_left = args.d_steps
                g_steps_left = args.g_steps
                
            logger.info('\n')
            logger.info('lr={}'.format(optimizer_g.param_groups[0]['lr']))    
            for k, v in sorted(losses_d.items()):
                logger.info('[D] {}: {:.3f}'.format(k, v))
            for k, v in sorted(losses_g.items()):
                logger.info('[G] {}: {:.3f}'.format(k, v))
            
            lr_scheduler_d.step()
            lr_scheduler_g.step()
                
            #test
            ##################################################################
            ade_all,fde_all,c_n_pred,c_n_real,total_ped=[],[],[],[],[]
            for i in range(args.test_num):
                ade_all_i, fde_all_i, c_n_pred_i, c_n_real_i, total_ped_i = do_test(args, generator, test_dataset, test_data_loader, t)
                ade_all.append(ade_all_i)
                fde_all.append(fde_all_i)
                c_n_pred.append(c_n_pred_i)
                c_n_real.append(c_n_real_i)
                total_ped.append(total_ped_i)
            ade_all = np.array(ade_all)
            fde_all = np.array(fde_all)
            c_n_pred = np.array(c_n_pred)
            c_n_real = np.array(c_n_real)
            total_ped = np.array(total_ped)
            result_idex_all = np.where(ade_all == ade_all.min())[0]
            FDE_min = 1e10
            for item in result_idex_all:
                if fde_all[item] < FDE_min:
                    FDE_min = fde_all[item]
                    result_index = item
           
            # Maybe save loss and checkpoint           
            if t % args.save_every == 0:

                for k, v in sorted(losses_d.items()):
                    checkpoint['D_losses'][k].append(v)
                for k, v in sorted(losses_g.items()):
                    checkpoint['G_losses'][k].append(v)

                checkpoint['g_state'] = generator.state_dict()
                #checkpoint['g_optim_state'] = optimizer_g.state_dict()
                checkpoint['d_state'] = discriminator.state_dict()
                #checkpoint['d_optim_state'] = optimizer_d.state_dict()
                checkpoint_path = os.path.join(args.output_dir, '%s_with_model_%d_%.2f_%.2f_%d_%d_%d.pt' 
                                               % (args.checkpoint_name,t,ade_all[result_index],fde_all[result_index],c_n_pred[result_index],c_n_real[result_index],total_ped[result_index]))
                logger.info('Saving checkpoint to {}'.format(checkpoint_path))
                torch.save(checkpoint, checkpoint_path)
                logger.info('Done.')

            generator.train()                
                

def evaluate_helper(error):
    best_idx = torch.argmin(error, dim=1)
    error_total = error[range(len(best_idx)), best_idx].sum()   
    return error_total,best_idx


def discriminator_step(args, batch, generator, discriminator, d_loss_fn, optimizer_d):
    #batch = [tensor.to(args.device) for tensor in batch]
    (obs_traj_rel, pred_traj_gt_rel, seq_start_end, first_history_index) = batch
    losses = {}
    loss = torch.zeros(1).to(args.device)
    
    K = args.best_k
    
    pred_traj_fake_rel, _, _, _ = generator(obs_traj_rel, pred_traj_gt_rel, seq_start_end, first_history_index, K)

    scores_fake = discriminator(obs_traj_rel, pred_traj_fake_rel, pred_traj_gt_rel, first_history_index, K).detach()
    scores_real = discriminator(obs_traj_rel, pred_traj_gt_rel.unsqueeze(1).repeat(1, K, 1, 1), pred_traj_gt_rel, first_history_index, K)
    ###########################################################################
    # Compute loss with optional gradient penalty
    scores_real_target_nodes = []
    scores_fake_target_nodes = []
    for start, _ in seq_start_end.data:
        scores_real_target_nodes.append(scores_real[start])
        scores_fake_target_nodes.append(scores_fake[start])
    scores_real_target_nodes = torch.stack(scores_real_target_nodes)
    scores_fake_target_nodes = torch.stack(scores_fake_target_nodes)
    
    ###########################################################################
    
    # Compute loss with optional gradient penalty    
    data_loss_real, data_loss_fake = d_loss_fn(scores_real_target_nodes, scores_fake_target_nodes)
    loss += data_loss_real.mean() + data_loss_fake.mean()
    losses['D_loss_real'] = data_loss_real.mean().item()
    losses['D_loss_fake'] = data_loss_fake.mean().item()

    optimizer_d.zero_grad()
    loss.backward()
    if args.clipping_threshold_d > 0:
        nn.utils.clip_grad_norm_(discriminator.parameters(), args.clipping_threshold_d)
    optimizer_d.step()

    return losses


def generator_step(args, batch, generator, discriminator, g_loss_fn, optimizer_g):
    #batch = [tensor.to(args.device) for tensor in batch]
    (obs_traj_rel, pred_traj_gt_rel, seq_start_end, first_history_index) = batch
    losses = {}
    loss = torch.zeros(1).to(args.device)
    KLD_sum = []
    
    K = args.best_k
    #predict
    pred_traj_fake_rel, KLD, GOAL_sum, _ = generator(obs_traj_rel, pred_traj_gt_rel, seq_start_end, first_history_index, K)
    KLD_sum = KLD.unsqueeze(1).repeat(1, K)
    g_l2_loss_rel = args.l2_loss_weight * l2_loss(pred_traj_fake_rel, pred_traj_gt_rel.unsqueeze(1).repeat(1, K, 1, 1))
    
    g_l2_KLD_GOAL = torch.zeros(1).to(args.device)
    #为了最终能够分别显示l2_loss,KLD_loss,goal_loss
    g_l2 = torch.zeros(1).to(args.device)
    g_KLD = torch.zeros(1).to(args.device)
    g_GOAL = torch.zeros(1).to(args.device)   
    
    for start, _ in seq_start_end.data:          
        #每一场景下，目标行人的l2_loss
        _g_l2_loss_rel = g_l2_loss_rel[start]
        #每一场景下，目标行人的KLD_loss
        _KLD_sum = KLD_sum[start]
        #每一场景下，目标行人的GOAL_loss
        _GOAL_sum = GOAL_sum[start]
        #l2_loss和KLD_loss和GOAL_sum相加
        _g_l2_KLD_GOAL = _g_l2_loss_rel + args.kld_loss_weight*_KLD_sum + args.goal_loss_weight*_GOAL_sum
        #每一场景下取best_k次最小值, 计算所有场景下的L2及KLD_loss及GOAL_loss
        g_l2_KLD_GOAL += torch.min(_g_l2_KLD_GOAL)        
        #为了最终能够分别显示l2_loss和KLD_loss和GOAL_loss
        idex = torch.argmin(_g_l2_KLD_GOAL)
        g_l2 += _g_l2_loss_rel[idex] 
        g_KLD += _KLD_sum[idex]
        g_GOAL += _GOAL_sum[idex]
        
    g_l2_KLD_GOAL /= seq_start_end.size(0)
    g_l2 /= seq_start_end.size(0)   
    g_KLD /= seq_start_end.size(0)
    g_GOAL /= seq_start_end.size(0)
    ###########################################################################    
    
    loss += g_l2_KLD_GOAL    
    losses['G_l2_loss_rel'] = g_l2.item()
    losses['G_KLD_loss_rel'] = g_KLD.item()      
    losses['G_GOAL_loss'] = g_GOAL.item()   
    
    #G-discriminator_loss部分
    scores_fake = discriminator(obs_traj_rel, pred_traj_fake_rel, pred_traj_gt_rel, first_history_index, K)
    
    ###########################################################################    
    scores_fake_target_nodes = []
    for start, _ in seq_start_end.data:
        scores_fake_target_nodes.append(scores_fake[start])
    scores_fake_target_nodes = torch.stack(scores_fake_target_nodes)
    ###########################################################################    
    
    discriminator_loss = g_loss_fn(scores_fake_target_nodes)
    loss += args.gan_loss_weight*discriminator_loss.min()
    losses['G_discriminator_loss'] = discriminator_loss.min().item()
    losses['G_total_loss'] = loss.item()
    
    optimizer_g.zero_grad()
    loss.backward()
    if args.clipping_threshold_g > 0:
        nn.utils.clip_grad_norm_(
            generator.parameters(), args.clipping_threshold_g
        )
    optimizer_g.step()

    return losses

def do_test(args, generator, test_dataset, test_data_loader, t):
    generator.eval() 
    ade_outer, fde_outer = [], [] 
    total_target_nodes = 0
    c_n_pred = 0
    c_n_real = 0
    total_number_of_pedestrian = 0
    with torch.no_grad():
        test_dataset.augment = False
        for node_type, data_loader in test_data_loader.items():
            pbar = tqdm(data_loader, ncols=100)
            for target_nodes in pbar:        
                # process data for batch
                first_history_index_nodes = target_nodes[0]
                target_nodes_x_original = target_nodes[1]
                target_nodes_y_original = target_nodes[2]               
                target_nodes_x = target_nodes[3]
                target_nodes_y = target_nodes[4]
                first_history_index_neighbors = restore(target_nodes[5]) 
                first_future_index_neighbors = restore(target_nodes[6])  
                target_nodes_neighbors_x = restore(target_nodes[7])
                target_nodes_neighbors_y = restore(target_nodes[8])
                target_nodes_neighbors_x_original = restore(target_nodes[9])
                target_nodes_neighbors_y_original = restore(target_nodes[10])
                
                seq_start = 0
                all_pedestrians_x = []
                all_pedestrians_y = []
                first_history_index = []
                first_future_index = []
                seq_start_end = []
                
                total_neighbors = 0
                
                for idex in range(0,target_nodes_x.shape[0]):
                    first_history_index_node = first_history_index_nodes[idex]
                    target_node_x = target_nodes_x[idex].unsqueeze(0)
                    target_node_y = target_nodes_y[idex].unsqueeze(0)
                    
                    first_history_index_neighbor = first_history_index_neighbors[(node_type,node_type)][idex][0]
                    first_future_index_neighbor = first_future_index_neighbors[(node_type,node_type)][idex][0]
                    target_node_neighbors_x = target_nodes_neighbors_x[(node_type,node_type)][idex]
                    target_node_neighbors_y = target_nodes_neighbors_y[(node_type,node_type)][idex]
                    
                    #####delete neighbors that do not have 7-args.neighbor_min_history+1 steps##########################################################################################
                    if len(target_node_neighbors_x)>=1: 
                        mask_neighbors = (first_history_index_neighbor <= args.neighbor_min_history).nonzero().squeeze(-1)
                        first_history_index_neighbor = first_history_index_neighbor[mask_neighbors]
                        first_future_index_neighbor = first_future_index_neighbor[mask_neighbors]
                        
                        if mask_neighbors.size(0) == 0:
                            target_node_neighbors_x = list()
                            target_node_neighbors_y = list()
                        else:                           
                            target_node_neighbors_x = torch.stack(target_node_neighbors_x) if len(target_node_neighbors_x)>=2 else target_node_neighbors_x[0].unsqueeze(0)
                            target_node_neighbors_y = torch.stack(target_node_neighbors_y) if len(target_node_neighbors_x)>=2 else target_node_neighbors_y[0].unsqueeze(0)
                            
                            target_node_neighbors_x = target_node_neighbors_x[mask_neighbors]
                            target_node_neighbors_y = target_node_neighbors_y[mask_neighbors]
                            
                            target_node_neighbors_x = list(target_node_neighbors_x.chunk(target_node_neighbors_x.size(0)))            
                            target_node_neighbors_y = list(target_node_neighbors_y.chunk(target_node_neighbors_y.size(0)))

                    ######################################################################################################################################################      
                    
                    total_neighbors += len(target_node_neighbors_x)
                    
                    target_node_x_with_neighbors = [target_node_x] + target_node_neighbors_x
                    target_node_x_with_neighbors = torch.cat(target_node_x_with_neighbors, dim=0)
                    
                    target_node_y_with_neighbors = [target_node_y] + target_node_neighbors_y
                    target_node_y_with_neighbors = torch.cat(target_node_y_with_neighbors, dim=0)
                    
                    all_pedestrians_x.append(target_node_x_with_neighbors)
                    all_pedestrians_y.append(target_node_y_with_neighbors)
                    
                    seq_start_end.append((seq_start, seq_start + target_node_x_with_neighbors.size(0)))
                    seq_start = seq_start + target_node_x_with_neighbors.size(0)
                    
                    first_history_index.append(first_history_index_node.view(1))
                    first_future_index.append(torch.tensor([11]))
                    if len(target_node_neighbors_x)>=1:
                        first_history_index.append(first_history_index_neighbor)
                        first_future_index.append(first_future_index_neighbor)
                        
                all_pedestrians_x = torch.cat(all_pedestrians_x).to(args.device)
                all_pedestrians_y = torch.cat(all_pedestrians_y).to(args.device)
                seq_start_end = torch.Tensor(seq_start_end).to(torch.int64).to(args.device)
                first_history_index = torch.cat(first_history_index).to(torch.int64).to(args.device)
                first_future_index = torch.cat(first_future_index).to(torch.int64).to(args.device)
                
                # predict
                ade, fde = [], []
                total_target_nodes += seq_start_end.size(0)            
                
                pred_traj_fake_rel, _, _, _ = generator(all_pedestrians_x, all_pedestrians_y, seq_start_end, first_history_index, K=args.best_k)
                
                pred_traj_fake_rel_target = []
                for (start, _) in seq_start_end:
                    pred_traj_fake_rel_target.append(pred_traj_fake_rel[start])
                pred_traj_fake_rel_target = torch.stack(pred_traj_fake_rel_target).cpu()
                
                for i in range(pred_traj_fake_rel_target.size(0)):
                    pred_traj_fake_rel_target[i] += target_nodes_x_original[i,-1,0:2].unsqueeze(0).repeat(12,1).unsqueeze(0).repeat(args.best_k,1,1)                            
               
                # evaluate
                ade = displacement_error(pred_traj_fake_rel_target, target_nodes_y_original.unsqueeze(1).repeat(1,args.best_k,1,1))
                fde = final_displacement_error(pred_traj_fake_rel_target, target_nodes_y_original.unsqueeze(1).repeat(1,args.best_k,1,1))
                
                ade_sum, best_idx = evaluate_helper(ade)
                fde_sum, _ = evaluate_helper(fde)
                ade_outer.append(ade_sum)
                fde_outer.append(fde_sum)
                
                #collision
                target_nodes_neighbors_y_original = target_nodes_neighbors_y_original[(node_type,node_type)]
                first_future_index_neighbors = first_future_index_neighbors[(node_type,node_type)]
                c_n_pred += collision_num(best_idx, pred_traj_fake_rel_target, target_nodes_neighbors_y_original, first_future_index_neighbors, realorpred=0, radius=0.1)
                c_n_real += collision_num(best_idx, target_nodes_y_original, target_nodes_neighbors_y_original, first_future_index_neighbors, realorpred=1, radius=0.1)
                
                # calculate total number of pedestrians in a batch
                total_number_of_pedestrian += target_nodes_y_original.size(0)
                for i in range(len(target_nodes_neighbors_y_original)):    
                    total_number_of_pedestrian += len(target_nodes_neighbors_y_original[i])
                
            ade_all = sum(ade_outer) / (total_target_nodes * args.pred_len) 
            fde_all = sum(fde_outer) / (total_target_nodes)               
            
            print('Dataset: {}, Epoch: {}, Pred Len: {}, ADE: {:.2f}, FDE: {:.2f}, c_n_pred: {}, c_n_real: {}, total_number_of_pedestrian: {},'
                  .format(args.test_data_dict, t, args.pred_len, ade_all, fde_all, c_n_pred, c_n_real, total_number_of_pedestrian))      
            
    return ade_all, fde_all, c_n_pred, c_n_real, total_number_of_pedestrian


if __name__ == '__main__':    
    FORMAT = '[%(levelname)s: %(filename)s: %(lineno)4d]: %(message)s'
    logging.basicConfig(level=logging.INFO, format=FORMAT, stream=sys.stdout)
    logger = logging.getLogger(__name__)  
    # Load args
    args = args()
    # Load hyperparameters from json
    if not os.path.exists(args.conf):
        print('Config json not found!')
    with open(args.conf, 'r', encoding='utf-8') as conf_json:
        hyperparams_trajectron = json.load(conf_json)
    hyperparams_trajectron['minimum_history_length'] = 1 #TODO
    hyperparams_trajectron['offline_scene_graph'] = 'no'
    # random seed
    if args.seed is not None:
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
    # create dir
    if os.path.exists(args.output_dir) == False: 
        os.makedirs(args.output_dir)
    # run main
    main(args,hyperparams_trajectron)
