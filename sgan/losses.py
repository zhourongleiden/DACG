import torch
import random


def bce_loss(result, target):
    """
    Numerically stable version of the binary cross-entropy loss function.
    As per https://github.com/pytorch/pytorch/issues/751
    See the TensorFlow docs for a derivation of this formula:
    https://www.tensorflow.org/api_docs/python/tf/nn/sigmoid_cross_entropy_with_logits
    Input:
    - input: PyTorch Tensor of shape (N, ) giving scores.
    - target: PyTorch Tensor of shape (N,) containing 0 and 1 giving targets.

    Output:
    - A PyTorch Tensor containing the mean BCE loss over the minibatch of
      input data.
    """
    neg_abs = -result.abs()
    loss = result.clamp(min=0) - result * target + (1 + neg_abs.exp()).log()
    return loss.mean(dim=0)


def gan_g_loss(scores_fake):
    """
    Input:
    - scores_fake: Tensor of shape (N,) containing scores for fake samples

    Output:
    - loss: Tensor of shape (,) giving GAN generator loss
    """
    y_fake = torch.ones_like(scores_fake) #* random.uniform(0.9, 1.1)
    return bce_loss(scores_fake, y_fake)


def gan_d_loss(scores_real, scores_fake):
    """
    Input:
    - scores_real: Tensor of shape (N,) giving scores for real samples
    - scores_fake: Tensor of shape (N,) giving scores for fake samples

    Output:
    - loss: Tensor of shape (,) giving GAN discriminator loss
    """
    y_real = torch.ones_like(scores_real) #* random.uniform(0.9, 1.1)
    y_fake = torch.zeros_like(scores_fake) #* random.uniform(0, 0.2)
    loss_real = bce_loss(scores_real, y_real)
    loss_fake = bce_loss(scores_fake, y_fake)
    return loss_real, loss_fake


def l2_loss(pred_traj, pred_traj_gt, first_future_index=None):
    """
    Input:
    - pred_traj: Tensor of shape (batch, K, seq_len, 2). Predicted trajectory.
    - pred_traj_gt: Tensor of shape (batch, K, seq_len, 2). Groud truth
    - first_future_index:  Tensor of shape (batch, )
    Output:
    - loss: l2 loss 
    """
    batch, K, pred_len, pred_dim = pred_traj.size()  
    if first_future_index != None:
        first_future_index = first_future_index + 1
        mask = torch.zeros_like(pred_traj)
        for i in range(batch):
            mask[i,:,0:(first_future_index[i])] = 1
    else:
        mask = 1
    loss = ((pred_traj_gt - pred_traj)**2)
    loss = loss*mask
    
    return (loss.sum(dim=3).sum(dim=2))/(first_future_index.unsqueeze(1).repeat(1,K)) if first_future_index != None else (loss.sum(dim=3).sum(dim=2))/pred_traj.size(2)

def goal_loss(pred_traj, pred_traj_gt):
    """
    Input:
    - pred_traj: Tensor of shape (batch, K, 2). Predicted trajectory.
    - pred_traj_gt: Tensor of shape (batch, K,  2). Groud truth

    Output:
    - loss: l2 loss 
    """

    loss = (pred_traj_gt - pred_traj)**2

    
    return loss.sum(dim=2)

def displacement_error(pred_traj, pred_traj_gt):
    """
    Input:
    - pred_traj: Tensor of shape (batch, K, seq_len, 2). Predicted trajectory.
    - pred_traj_gt: Tensor of shape (batch, K, seq_len, 2). Ground truth
    predictions.
    Output:
    - loss: gives the eculidian displacement error
    """
    
    loss = pred_traj_gt - pred_traj
    loss = loss**2
    loss = torch.sqrt(loss.sum(dim=3)).sum(dim=2)
    
    return loss


def final_displacement_error(pred_pos, pred_pos_gt):
    """
    Input:
    - pred_pos: Tensor of shape (batch, K, seq_len, 2). 
    - pred_pos_gt: Tensor of shape (batch, K, seq_len, 2). 
    Output:
    - loss: gives the eculidian displacement error
    """
    loss = pred_pos_gt[:,:,-1,:] - pred_pos[:,:,-1,:]#Predicted last pos-Groud truth last pos
    loss = loss**2
    loss = torch.sqrt(loss.sum(dim=2))
    
    return loss



def collision_num(best_idx, traj, target_nodes_neighbors_y_original, first_future_index_neighbors, realorpred, radius):
    
    if realorpred == 0 :
        pred_best = []
        for i in range(best_idx.size(0)):
            pred_best.append(traj[i,best_idx[i],:,:])
        pred_best = torch.stack(pred_best)
    else:
        pred_best = traj
    
    c_num = 0
    for i in range(best_idx.size(0)):
        neighbors = target_nodes_neighbors_y_original[i]
        neighbors_future_idx = first_future_index_neighbors[i][0].to(torch.int64)
        if len(neighbors) >0 :
            neighbors_mask = torch.zeros(len(neighbors),12)
            for j, item in enumerate(neighbors_future_idx):
                neighbors_mask[j,0:item+1] = 1
            neighbors = torch.stack(neighbors)
            target_pred = pred_best[i].unsqueeze(0).repeat(len(neighbors),1,1)
            dist = neighbors_mask * torch.sqrt(torch.sum((target_pred-neighbors)**2, dim=2))
            dist.data.masked_fill_(~neighbors_mask.bool(), 1e12) 
            c_num += torch.lt(dist,radius).nonzero().size(0)
    
    return c_num
        

def collision_num_od(best_idx, traj, realorpred, radius): #original dataloader
    
    if realorpred == 0 :
        pred_best = []
        for i in range(best_idx.size(0)):
            pred_best.append(traj[i,best_idx[i],:,:])
        pred_best = torch.stack(pred_best)
    else:
        pred_best = traj
   
    tmp1 = pred_best.unsqueeze(1).repeat(1,best_idx.size(0),1,1)
    tmp2 = pred_best.unsqueeze(0).repeat(best_idx.size(0),1,1,1)
    dist = torch.sqrt(torch.sum((tmp1-tmp2)**2, dim=3))
    
    mask = torch.eye(best_idx.size(0))
    mask = mask.unsqueeze(2).repeat(1,1,pred_best.size(1))
    
    dist.data.masked_fill_(mask.bool(), 1e12) 
    
    c_num = torch.lt(dist,radius).nonzero().size(0)
    
    return c_num
                    

