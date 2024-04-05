import argparse
import torch
from sgan.utils import int_tuple, bool_flag

def args():
    parser = argparse.ArgumentParser()
    # Generator Options
    parser.add_argument('--obs_len', default=8, type=int)
    parser.add_argument('--pred_len', default=12, type=int)
    parser.add_argument('--embedding_dim', default=256, type=int)# equal to goal_h_dim
    parser.add_argument('--encoder_h_dim_g', default=256, type=int)# equal to graph_network_out_dims
    parser.add_argument('--decoder_h_dim_g', default=256, type=int)
    parser.add_argument('--mlp_dim', default=128, type=int)
    parser.add_argument('--num_layers', default=1, type=int)
    parser.add_argument('--noise_dim', default=(32, ), type=int_tuple)
    parser.add_argument('--dropout', default=0, type=float)
    parser.add_argument('--batch_norm', default=0, type=bool_flag)
    parser.add_argument('--graph_lstm_hidden_size', default=256, type=int)
    parser.add_argument('--graph_network_out_dims', default=256, type=int)
    parser.add_argument("--heads", type=str, default="4,1", help="Heads in each layer, splitted with comma")
    parser.add_argument("--hidden_units", type=str, default="128", help="Hidden units in each hidden layer, splitted with comma")
    parser.add_argument("--alpha", type=float, default=0.2, help="Alpha for the leaky_relu.")
    parser.add_argument("--teacher_forcing_ratio", type=float, default=0)
    parser.add_argument("--goal_embedding_dim", type=int, default=256)
    parser.add_argument("--goal_h_dim", type=int, default=256)
    parser.add_argument("--neighbor_min_history", type=int, default=7)#TODO
    parser.add_argument('--clipping_threshold_g', default=1.5, type=float)
    parser.add_argument('--g_learning_rate', default=1e-4, type=float)
    parser.add_argument('--g_steps', default=1, type=int)
    parser.add_argument('--with_goal', default=True, type=str)
    parser.add_argument('--with_cvae', default=True, type=str)
    
    # Discriminator Options
    parser.add_argument('--encoder_h_dim_d', default=256, type=int)
    parser.add_argument('--d_learning_rate', default=1e-4, type=float)
    parser.add_argument('--d_steps', default=1, type=int)
    parser.add_argument('--clipping_threshold_d', default=0, type=float)
    
    # Dataset options
    parser.add_argument('--dataset_name', default='eth', type=str)
    parser.add_argument('--delim', default='\t')
    parser.add_argument('--skip', default=1, type=int)
    parser.add_argument('--batch_size', default=64, type=int)
    parser.add_argument("--data_dir", default='./trajectron/experiments/processed', type=str)
    parser.add_argument("--train_data_dict", default='eth_train.pkl', type=str)
    parser.add_argument("--val_data_dict", default='eth_val.pkl', type=str)
    parser.add_argument("--test_data_dict", default='eth_test.pkl', type=str)
    
    # Optimization
    parser.add_argument('--l2_loss_weight', default=1, type=float)
    parser.add_argument('--kld_loss_weight', default=10, type=float)
    parser.add_argument('--goal_loss_weight', default=0.01, type=float)
    parser.add_argument('--gan_loss_weight', default=1, type=float)
    parser.add_argument('--test_num', default=1, type=int)
    parser.add_argument('--best_k', default=20, type=int)
    parser.add_argument('--train_epochs', default=100, type=int)
    parser.add_argument('--augment', help="Whether to augment the scene during training", default=True)
    parser.add_argument('--override_attention_radius', action='append', help='Specify one attention radius to override. E.g. "PEDESTRIAN VEHICLE 10.0"', default=[])
    parser.add_argument("--offline_scene_graph", default='yes', type=str)
    parser.add_argument('--seed', default=7, type=int)
    
    # Output
    parser.add_argument('--output_dir', default='')
    parser.add_argument('--save_every', default=1, type=int)
    parser.add_argument('--checkpoint_name', default='checkpoint')
    parser.add_argument('--log_dir', default='')
    parser.add_argument("--log_tag", help="tag for the log folder", default='', type=str)
    
    # Misc
    parser.add_argument('--gpu', default=-1, type=int)
    parser.add_argument('--device', default='cpu', type=str)
    parser.add_argument("--preprocess_workers", default=0, type=int)
    
    # Hyperprameters for dataloader
    parser.add_argument("--conf", default='./trajectron/experiments/models/eth_attention_radius_3/config.json')
    
    ##############################################################################
    args = parser.parse_args()
    args.device = torch.device('cuda:{}'.format(args.gpu) if args.gpu != -1 else 'cpu')
    if args.with_goal == False:
        args.goal_loss_weight = 0
    args.output_dir = './results/'+args.dataset_name
    args.train_data_dict = args.dataset_name+'_train.pkl'
    args.val_data_dict = args.dataset_name+'_val.pkl'
    args.test_data_dict = args.dataset_name+'_test.pkl'
    args.conf = './trajectron/experiments/models/'+args.dataset_name+'_attention_radius_3/config.json'
    
    
    return args

if __name__ == "__main__":  
    arguments = args()
