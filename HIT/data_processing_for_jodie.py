"""Unified interface to all dynamic graph model experiments"""
from tqdm import tqdm
import pandas as pd
from log import set_up_logger
from parser import *
from eval import *
from utils import *
# from train import *
#import numba

# from module import TGAN
# from graph import NeighborFinder
# import resource

args, sys_argv = get_args()

# BATCH_SIZE = args.bs
# NUM_NEIGHBORS = args.n_degree
# NUM_EPOCH = args.n_epoch
# ATTN_NUM_HEADS = args.attn_n_head
# DROP_OUT = args.drop_out
# GPU = args.gpu
# UNIFORM = args.uniform
# USE_TIME = args.time
# ATTN_AGG_METHOD = args.attn_agg_method
# ATTN_MODE = args.attn_mode
# SEQ_LEN = NUM_NEIGHBORS
DATA = args.data
# NUM_LAYER = args.n_layer
# LEARNING_RATE = args.lr
# POS_DIM = args.pos_dim
# WALK_N_HEAD = args.walk_n_head
# WALK_RECENT_BIAS = args.walk_recent_bias if not UNIFORM else 1.0  #TODO: understand the mechanisms of parameter control
# WALK_MUTUAL = args.walk_mutual
# TOLERANCE = args.tolerance
# CPU_CORES = args.cpu_cores
# NGH_CACHE = args.ngh_cache
# VERBOSITY = args.verbosity
AGG = args.agg
SEED = args.seed
# assert(CPU_CORES >= -1)
# set_random_seed(SEED)
logger, get_checkpoint_path, best_model_path = set_up_logger(args, sys_argv)

# Load data and sanity check
g_df = pd.read_csv('./processed/ml_{}.csv'.format(DATA))
e_feat = np.load('./processed/ml_{}.npy'.format(DATA))
e_feat_dim = len(e_feat[0])
n_feat = np.load('./processed/ml_{}_node.npy'.format(DATA))
src_l = g_df.u.values
dst_l = g_df.i.values
e_idx_l = g_df.idx.values
label_l = g_df.label.values
ts_l = g_df.ts.values
max_idx = max(src_l.max(), dst_l.max())
assert(np.unique(np.stack([src_l, dst_l])).shape[0] == max_idx)  # all nodes except node 0 should appear and be compactly indexed
assert(n_feat.shape[0] == max_idx + 1)  # the nodes need to map one-to-one to the node feat matrix

# split and pack the data by generating valid train/val/test flag according to the "mode"
# val_time, test_time = list(np.quantile(g_df.ts, [0.70, 0.85]))
val_time = (max(ts_l) - min(ts_l)) * 0.75 + min(ts_l)
test_time = (max(ts_l) - min(ts_l)) * 0.825 + min(ts_l)
end_time = (max(ts_l) - min(ts_l)) * 0.9 + min(ts_l)
if args.mode == 't':
    logger.info('Transductive training...')
    valid_train_flag = (ts_l <= val_time)
    valid_val_flag = (ts_l <= test_time) * (ts_l > val_time)
    valid_test_flag = (ts_l <= end_time) * (ts_l > test_time)

else:
    assert(args.mode == 'i')
    logger.info('Inductive training...')
    # pick some nodes to mask (i.e. reserved for testing) for inductive setting
    total_node_set = set(np.unique(np.hstack([g_df.u.values, g_df.i.values])))
    num_total_unique_nodes = len(total_node_set)
    mask_node_set = set(random.sample(set(src_l[ts_l > val_time]).union(set(dst_l[ts_l > val_time])), int(0.1 * num_total_unique_nodes)))
    mask_src_flag = g_df.u.map(lambda x: x in mask_node_set).values
    mask_dst_flag = g_df.i.map(lambda x: x in mask_node_set).values
    none_mask_node_flag = (1 - mask_src_flag) * (1 - mask_dst_flag)
    valid_train_flag = (ts_l <= val_time) * (none_mask_node_flag > 0.5)
    valid_val_flag = (ts_l <= test_time) * (ts_l > val_time) * (none_mask_node_flag > 0.5)  # both train and val edges can not contain any masked nodes
    valid_test_flag = (ts_l > test_time) * (none_mask_node_flag < 0.5)  # test edges must contain at least one masked node
    logger.info('Sampled {} nodes (10 %%) which are masked in triaining and reserved for testing...'.format(len(mask_node_set)))

train_src_l, train_dst_l, train_ts_l, train_e_idx_l, train_label_l = src_l[valid_train_flag], dst_l[valid_train_flag], ts_l[valid_train_flag], e_idx_l[valid_train_flag], label_l[valid_train_flag]
val_src_l, val_dst_l, val_ts_l, val_e_idx_l, val_label_l = src_l[valid_val_flag], dst_l[valid_val_flag], ts_l[valid_val_flag], e_idx_l[valid_val_flag], label_l[valid_val_flag]
test_src_l, test_dst_l, test_ts_l, test_e_idx_l, test_label_l = src_l[valid_test_flag], dst_l[valid_test_flag], ts_l[valid_test_flag], e_idx_l[valid_test_flag], label_l[valid_test_flag]
train_data = train_src_l, train_dst_l, train_ts_l, train_e_idx_l, train_label_l
val_data = val_src_l, val_dst_l, val_ts_l, val_e_idx_l, val_label_l
train_val_data = (train_data, val_data)

# data process for jodie
# jodie
# .csv file
# The network should be in the following format:
# - One line per interaction/edge.
# - Each line should be: *user, item, timestamp, state label, comma-separated array of features*.
# - First line is the network format.
# - *User* and *item* fields can be alphanumeric.
# - *Timestamp* should be in cardinal format (not in datetime).
# - *State label* should be 1 whenever the user state changes, 0 otherwise. If there are no state labels, use 0 for all interactions.
# - *Feature list* can be as long as desired. It should be atleast 1 dimensional. If there are no features, use 0 for all interactions.
#
# For example, the first few lines of a dataset can be:
# ```
# user,item,timestamp,state_label,comma_separated_list_of_features
# 0,0,0.0,0,0.1,0.3,10.7
# 2,1,6.0,0,0.2,0.4,0.6
# 5,0,41.0,0,0.1,15.0,0.6
# 3,2,49.0,1,100.7,0.8,0.9

fout = open('./jodie_data/ml_{}_{}.csv'.format(DATA, args.mode), 'w')
fout.write('user,item,timestamp,state_label,comma_separated_list_of_features\n')

def print_data(src_l, dst_l, ts_l, e_idx_l):
    for i in tqdm(range(len(src_l))):
        fout.write('%s,%s,%s,%s' %(src_l[i], dst_l[i], ts_l[i], 0))
        for l in e_feat[e_idx_l[i]]:
            fout.write(',%s' %l)

        # fout.write(',%s' %e_feat[e_idx_l[i]])
        fout.write('\n')

#
print('start train\n')
print_data(train_src_l, train_dst_l, train_ts_l, train_e_idx_l)
# val
print('start val\n')
print_data(val_src_l, val_dst_l, val_ts_l, val_e_idx_l)
# train
print('start test\n')
print_data(test_src_l, test_dst_l, test_ts_l, test_e_idx_l)

fout.close()

fout = open('./jodie_data/ml_{}_{}_split.csv'.format(DATA, args.mode), 'w')
fout.write('%s,%s,%s\n' %(len(train_src_l), len(train_src_l) + len(val_src_l), len(train_src_l) + len(val_src_l) + len(test_src_l)))