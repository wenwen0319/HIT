import pandas as pd
from log import set_up_logger
from parser import *
from eval_time_prediction import *
from utils_time_prediction import *
from train_time_prediction import *
#import numba
import os
from module_intrepertation import CAWN
# from module import CAWN
from graph import NeighborFinder
import resource
from sklearn.preprocessing import scale
from histogram import plot_hist

# python main.py -d tags-math-sx --gpu 0 --bias 1e-7 --pos_enc lp --time_prediction --time_prediction_type 1
# python main.py -d tags-math-sx --gpu 0 --bias 1e-7 --interpretation --interpretation_type 1 --walk_linear_out --test_path ./saved_checkpoints/1610336081.2772572-tags-ask-ubuntu-t-walk-2-64k2-108-lp-inter-1/checkpoint-epoch-2-inter.pth > ./interpretation_output/tags-ask-ubuntu_1.txt


args, sys_argv = get_args()

BATCH_SIZE = args.bs
NUM_NEIGHBORS = args.n_degree
NUM_EPOCH = args.n_epoch
ATTN_NUM_HEADS = args.attn_n_head
DROP_OUT = args.drop_out
GPU = args.gpu
USE_TIME = args.time
ATTN_AGG_METHOD = args.attn_agg_method
ATTN_MODE = args.attn_mode
DATA = args.data
NUM_LAYER = args.n_layer
if args.time_prediction_type > 0:
    LEARNING_RATE = 1e-3
else:
    LEARNING_RATE = args.lr

POS_ENC = args.pos_enc
POS_DIM = args.pos_dim
WALK_N_HEAD = args.walk_n_head
WALK_MUTUAL = args.walk_mutual
TOLERANCE = args.tolerance
CPU_CORES = args.cpu_cores
NGH_CACHE = args.ngh_cache
VERBOSITY = args.verbosity
interpretation = args.interpretation
interpretation_type = args.interpretation_type
time_prediction = args.time_prediction
time_prediction_type = args.time_prediction_type
WALK_POOL = args.walk_pool
walk_linear_out = args.walk_linear_out
test_path = args.test_path
# print("test", test_path, test_path is None)
# Nega = args.negative
AGG = args.agg
SEED = args.seed
assert(CPU_CORES >= -1)
set_random_seed(SEED)
logger, get_checkpoint_path, best_model_path = set_up_logger(args, sys_argv)
if args.debug == False:
    if interpretation:
        print("Interpretation used spd, walk Linear out")
        POS_ENC = 'spd'
        walk_linear_out = True
        best_model_path = './interpretation_output/{}_{}.pth'.format(DATA, interpretation_type)
        fout = open('./interpretation_output/{}_{}.txt'.format(DATA, interpretation_type), 'w')
        sys.stdout = fout
    elif time_prediction:
        print("Time prediction used lp")
        POS_ENC = 'lp'
        best_model_path = './time_prediction_output/{}_{}.pth'.format(DATA, time_prediction_type)
        fout = open('./time_prediction_output/{}_{}.txt'.format(DATA, time_prediction_type), 'w')
        sys.stdout = fout
# Load data and sanity check

g_df = pd.read_csv('./processed/ml_{}.csv'.format(DATA))
e_feat = np.load('./processed/ml_{}.npy'.format(DATA))
n_feat = np.load('./processed/ml_{}_node.npy'.format(DATA))
src_l = g_df.u.values
dst_l = g_df.i.values
e_idx_l = g_df.idx.values
label_l = g_df.label.values
ts_l = g_df.ts.values
# plot_hist(ts_l, 100, ts_l.min(), ts_l.max(), 'ts_list', DATA, density=True, CDF=True)
# _time = 1
# while _time < max(ts_l):
#     _time = _time * 10
# ts_l = ts_l * 1.0 / (_time * 1e-7)
# congress-bills dont have to do the scale
if DATA == 'congress-bills':
    pass
# # elif DATA == 'tags-math-sx':
elif DATA == 'DAWN':
    pass
# elif DATA == 'NDC-substances':
#     pass
else:
    pass
    _time = 1
    while _time < max(ts_l):
        _time = _time * 10
    if time_prediction:
        ts_l = ts_l * 1.0 / (_time * 1e-7)    
    else:
        ts_l = ts_l * 1.0 / (_time * 1e-7)
    
    print(_time)
print(max(ts_l), min(ts_l))
ts_l = ts_l - min(ts_l) # + 10000
# ts_l = scale(ts_l)
print(max(ts_l), min(ts_l))
max_idx = max(src_l.max(), dst_l.max())
print(n_feat.shape[0], max_idx, np.unique(np.stack([src_l, dst_l])).shape[0])
assert(np.unique(np.stack([src_l, dst_l])).shape[0] == max_idx)  # all nodes except node 0 should appear and be compactly indexed
assert(n_feat.shape[0] == max_idx + 1)  # the nodes need to map one-to-one to the node feat matrix



# find possitive and negative triplet
if DATA == 'tags-math-sx':
    time_window_factor, time_start_factor = 0.10, 0.4
elif DATA == 'email-Eu':
    time_window_factor, time_start_factor = 0.10, 0.05
else:
    time_window_factor, time_start_factor = 0.10, 0.4
file_path = './saved_triplets/'+DATA+'/'+DATA+'_'+str(time_start_factor)+'_'+str(time_window_factor)
test = 0
if os.path.exists(file_path) and (test==0):
    # load cls_tri, opn_tri, wedge, nega, set_all_nodes
    # flag_save = 1 # save it later
    with open(file_path+'/triplets.npy', 'rb') as f:
        x = np.load(f, allow_pickle=True)
        cls_tri, opn_tri, wedge, nega, set_all_nodes = x[0], x[1], x[2], x[3], x[4]
        print("close tri", len(cls_tri[0]))
        print("open tri", len(opn_tri[0]))
        print("wedge", len(wedge[0]))
        print("nega", len(nega[0]))
else:
    cls_tri, opn_tri, wedge, nega, set_all_nodes = preprocess_dataset(ts_list=ts_l, src_list=src_l, dst_list=dst_l, node_max=max_idx, edge_idx_list=e_idx_l, 
                                                                      label_list=label_l, time_window_factor=time_window_factor, time_start_factor=time_start_factor)
    if not(os.path.exists(file_path)):
        os.makedirs(file_path)
    with open(file_path+'/triplets.npy', 'wb') as f:
        x = np.array([cls_tri, opn_tri, wedge, nega, set_all_nodes])
        np.save(f, x)




# triangle closure
# randomly choose 70% as training, 15% as validating, 15% as testing
ts1 = time_start_factor + 0.7 * (1 - time_start_factor - time_window_factor)
ts2 = time_start_factor + 0.85 * (1 - time_start_factor - time_window_factor)
ts_start = (ts_l.max() - ts_l.min()) * time_start_factor + ts_l.min()
ts_end = ts_l.max() - (ts_l.max() - ts_l.min()) * time_window_factor
ts_train = (ts_end - ts_start) * 0.7 + ts_start
ts_val = (ts_end - ts_start) * 0.85 + ts_start
# ts_train = np.quantile(ts_l, ts1)
# ts_val = np.quantile(ts_l, ts2)
# ts_end = np.quantile(ts_l, 1 - time_start_factor)
# DataLoader = TripletSampler(cls_tri, opn_tri, wedge, nega, ts_train, ts_val)

# create two neighbor finders to handle graph extraction.
# for transductive mode all phases use full_ngh_finder, for inductive node train/val phases use the partial one
# while test phase still always uses the full one
partial_adj_list = [[] for _ in range(max_idx + 1)]
full_adj_list = [[] for _ in range(max_idx + 1)]

full_adj_list = [[] for _ in range(max_idx + 1)]
for src, dst, eidx, ts in zip(src_l, dst_l, e_idx_l, ts_l):
    full_adj_list[src].append((dst, eidx, ts))
    full_adj_list[dst].append((src, eidx, ts))
full_ngh_finder = NeighborFinder(full_adj_list, bias=args.bias, use_cache=NGH_CACHE, sample_method=args.pos_sample)
partial_adj_list = [[] for _ in range(max_idx + 1)]
idx_partial = ts_l <= ts_val
# print(idx_partial)
for src, dst, eidx, ts in zip(src_l[idx_partial], dst_l[idx_partial], e_idx_l[idx_partial], ts_l[idx_partial]):
    partial_adj_list[src].append((dst, eidx, ts))
    partial_adj_list[dst].append((src, eidx, ts))
    # if (src == 415) and (eidx == 1265020):
    #     print()
partial_ngh_finder = NeighborFinder(partial_adj_list, bias=args.bias, use_cache=NGH_CACHE, sample_method=args.pos_sample)
print("Finish build CAWN")
ngh_finders = partial_ngh_finder, full_ngh_finder

# Yanbang's code
# # create random samplers to generate train/val/test instances
# train_rand_sampler = RandEdgeSampler((train_src_l, ), (train_dst_l, ))
# val_rand_sampler = RandEdgeSampler((train_src_l, val_src_l), (train_dst_l, val_dst_l))
# test_rand_sampler = RandEdgeSampler((train_src_l, val_src_l, test_src_l), (train_dst_l, val_dst_l, test_dst_l))
# rand_samplers = train_rand_sampler, val_rand_sampler




# multiprocessing memory setting
rlimit = resource.getrlimit(resource.RLIMIT_NOFILE)
resource.setrlimit(resource.RLIMIT_NOFILE, (200*args.bs, rlimit[1]))

# model initialization
device = torch.device('cuda:{}'.format(GPU))
cawn = CAWN(n_feat, e_feat, agg=AGG,
            num_layers=NUM_LAYER, use_time=USE_TIME, attn_agg_method=ATTN_AGG_METHOD, attn_mode=ATTN_MODE,
            n_head=ATTN_NUM_HEADS, drop_out=DROP_OUT, pos_dim=POS_DIM, pos_enc=POS_ENC, walk_pool=WALK_POOL, 
            num_neighbors=NUM_NEIGHBORS, walk_n_head=WALK_N_HEAD, walk_mutual=WALK_MUTUAL, walk_linear_out=walk_linear_out,
            cpu_cores=CPU_CORES, verbosity=VERBOSITY, get_checkpoint_path=get_checkpoint_path, interpretation=interpretation, interpretation_type=interpretation_type, time_prediction=time_prediction)
cawn.to(device)

# # Yunyu Triangle closure
dataset = TripletSampler(cls_tri, opn_tri, wedge, nega, ts_start, ts_train, ts_val, ts_end, set_all_nodes, DATA, interpretation_type, time_prediction_type=time_prediction_type)

logger.info(interpretation and (not os.path.exists(best_model_path)))

# if ((not interpretation) and test_path is None) or (interpretation and (not os.path.exists(best_model_path))): # change only for interpretation
if (not os.path.exists(best_model_path)):
    optimizer = torch.optim.Adam(cawn.parameters(), lr=LEARNING_RATE)
    # criterion = torch.nn.BCELoss()
    criterion = torch.nn.CrossEntropyLoss() # cls_tri, opn_tri, wedge, nega
    # criterion = torch.nn.MSELoss()
    early_stopper = EarlyStopMonitor(tolerance=TOLERANCE)

    # start train and val phases
    # train_val(train_val_data, cawn, args.mode, BATCH_SIZE, NUM_EPOCH, criterion, optimizer, early_stopper, ngh_finders, rand_samplers, logger)
    # Yunyu's triangle closure
    train_val(dataset, cawn, args.mode, BATCH_SIZE, NUM_EPOCH, criterion, optimizer, early_stopper, ngh_finders, logger, interpretation=interpretation, time_prediction=time_prediction)
else:
    # print("load modal")
    logger.info("load model")
    dataset.set_batch_size(BATCH_SIZE)
    cawn.load_state_dict(torch.load(test_path))
    # if interpretation:
    #     # cawn.update_ngh_finder(partial_ngh_finder)
    #     cawn.update_ngh_finder(full_ngh_finder)
    #     # test_acc, test_ap, test_f1, test_auc, cm = eval_one_epoch('train for {} nodes'.format(args.mode), cawn, dataset, val_flag='train', interpretation=interpretation)
    #     test_acc, test_ap, test_f1, test_auc, cm = eval_one_epoch('train for {} nodes'.format(args.mode), cawn, dataset, val_flag='test', interpretation=interpretation)
    #     # return 
    #     sys.exit()
dic1 = {}
dic2 = {}
# final testing
cawn.update_ngh_finder(full_ngh_finder)  # remember that testing phase should always use the full neighbor finder
# test_acc, test_ap, test_f1, test_auc, dic1, dic2 = eval_one_epoch('test for {} nodes'.format(args.mode), cawn, test_rand_sampler, test_src_l, test_dst_l, test_ts_l, test_label_l, test_e_idx_l, './jodie_data/ml_{}_{}'.format(DATA, args.mode)+'.mat', dic1, dic2)


"""
comment for interpretation
"""
# if time_prediction:
#     NLL_total, num_test_instance, time_predicted_total, time_gt_total = eval_one_epoch('test for {} nodes'.format(args.mode), cawn, dataset, val_flag='test', interpretation=interpretation, time_prediction=time_prediction)    
#     print('Testing NLL: ', NLL_total, 'num instances: ', num_test_instance)
#     file_addr = './Histogram/'+DATA+'/'
#     print("time_predicted_total", time_predicted_total)
#     print("time_gt_total", time_gt_total)
#     if not os.path.exists(file_addr):
#             os.makedirs(file_addr)
#     histogram.plot_hist_multi([time_predicted_total, time_gt_total], bins=50, figure_title='time_prediction_histogram', file_addr=file_addr, label=['Ours', 'Groundtruth'])
# else:
#     test_acc, test_ap, test_f1, test_auc, cm = eval_one_epoch('test for {} nodes'.format(args.mode), cawn, dataset, val_flag='test', interpretation=interpretation, time_prediction=time_prediction)
#     print('Test statistics: {} all nodes -- acc: {}, auc: {}, ap: {}'.format(args.mode, test_acc, test_auc, test_ap))
#     # print("Confusion matrix\n", cm)
#     logger.info(', '.join(str(r) for r in cm.reshape(1,-1)))
#     logger.info('Test statistics: {} all nodes -- acc: {}, auc: {}, ap: {}'.format(args.mode, test_acc, test_auc, test_ap))

# if args.mode == 'i':
#     test_new_new_acc, test_new_new_ap, test_new_new_f1, test_new_new_auc, _, _ = eval_one_epoch('test for {} nodes'.format(args.mode), cawn, test_rand_sampler, test_src_new_new_l, test_dst_new_new_l, test_ts_new_new_l, test_label_new_new_l, test_e_idx_new_new_l, './jodie_data/ml_{}_{}'.format(DATA, args.mode)+'_new_new.mat', dic1, dic2)
#     logger.info('Test statistics: {} new-new nodes -- acc: {}, auc: {}, ap: {}'.format(args.mode, test_new_new_acc, test_new_new_ap, test_new_new_auc ))
#     test_new_old_acc, test_new_old_ap, test_new_old_f1, test_new_old_auc, _, _ = eval_one_epoch('test for {} nodes'.format(args.mode), cawn, test_rand_sampler, test_src_new_old_l, test_dst_new_old_l, test_ts_new_old_l, test_label_new_old_l, test_e_idx_new_old_l, './jodie_data/ml_{}_{}'.format(DATA, args.mode)+'_new_old.mat', dic1, dic2)
#     logger.info('Test statistics: {} new-old nodes -- acc: {}, auc: {}, ap: {}'.format(args.mode, test_new_old_acc, test_new_old_ap, test_new_old_auc))

# save model
if test is None:
    logger.info('Saving cawn model')
    torch.save(cawn.state_dict(), best_model_path)
    logger.info('cawn models saved')
# # motif_checking
# sio.savemat('./motif_checking/{}_testing_mode_{}_pos.mat'.format(args.data, args.mode), dic1)
# sio.savemat('./motif_checking/{}_testing_mode_{}_neg.mat'.format(args.data,  args.mode), dic2)

# dic3 = {}
# dic4 = {}
# # final testing
# tgan.update_ngh_finder(full_ngh_finder)  # remember that testing phase should always use the full neighbor finder
# test_acc, test_ap, test_f1, test_auc, dic3, dic4 = eval_one_epoch('test for {} nodes'.format(args.mode), tgan, train_rand_sampler, train_src_l, train_dst_l, train_ts_l, train_label_l, train_e_idx_l, './jodie_data/ml_{}_{}'.format(DATA, args.mode)+'.mat', dic3, dic4)
# logger.info('Train statistics: {} all nodes -- acc: {}, auc: {}, ap: {}'.format(args.mode, test_acc, test_auc, test_ap))

# sio.savemat('./motif_checking/{}_traing_mode_{}_pos.mat'.format(args.data, args.mode), dic3)
# sio.savemat('./motif_checking/{}_training_mode_{}_neg.mat'.format(args.data , args.mode), dic4)

#TODO: use alpha to tune sampling strategy, check the logic again and then commit: fix time enocder for walk, add transductive inductive switch, fix label leak in inductive setting