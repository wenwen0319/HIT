import math
import torch
import numpy as np
from sklearn.metrics import average_precision_score
from sklearn.metrics import f1_score
from sklearn.metrics import roc_auc_score
import scipy.io as sio
import motif_checking

# def eval_one_epoch(hint, tgan, sampler, src, dst, ts, label, val_e_idx_l=None, file_name='', dic1={}, dic2={}):
def eval_one_epoch(hint, tgan, sampler, src_1, src_2, dst, ts, val_e_idx_l=None):
    val_acc, val_ap, val_f1, val_auc = [], [], [], []
    src_1_fake, src_2_fake, dst_fake = [], [], []
    with torch.no_grad():
        tgan = tgan.eval()
        TEST_BATCH_SIZE = 30
        num_test_instance = len(src_1)
        num_test_batch = math.ceil(num_test_instance / TEST_BATCH_SIZE)
        for k in range(num_test_batch):
            # percent = 100 * k / num_test_batch
            # if k % int(0.2 * num_test_batch) == 0:
            #     logger.info('{0} progress: {1:10.4f}'.format(hint, percent))
            s_idx = k * TEST_BATCH_SIZE
            e_idx = min(num_test_instance, s_idx + TEST_BATCH_SIZE)
            if s_idx == e_idx:
                continue
            src_1_l_cut = src_1[s_idx:e_idx]
            src_2_l_cut = src_2[s_idx:e_idx]
            dst_l_cut = dst[s_idx:e_idx]
            ts_l_cut = ts[s_idx:e_idx]
            e_l_cut = val_e_idx_l[s_idx:e_idx] if (val_e_idx_l is not None) else None
            # label_l_cut = label[s_idx:e_idx]

            size = len(src_1_l_cut)
            src_1_l_neg, src_2_l_neg, dst_l_neg, ts_l_neg, e_idx_l_neg = sampler.sample(size)
            # if dst_fake == []:
            #     dst_fake = np.copy(dst_l_fake)
            # else:
            # dst_fake = np.concatenate([dst_fake, dst_l_fake])

            # pos_pattern, neg_pattern, pos_prob, neg_prob = tgan.contrast(src_1_l_cut, src_2_l_cut, dst_l_cut, dst_l_fake, ts_l_cut, e_l_cut)
            pos_prob, neg_prob = tgan.contrast(src_1_l_cut, src_2_l_cut, dst_l_cut, ts_l_cut, None, src_1_l_neg, src_2_l_neg, dst_l_neg, ts_l_neg, None)   # the core training code
            # print('begin motif checking')
            # dic1, dic2 = motif_checking.get_pattern_score(dic1, dic2, pos_pattern, neg_pattern)
            # print('end motif checking')
            # sio.savemat('./motif_checking/epoch{}_training mode {}'.format(epoch, mode))
            # print(dic1, dic2)
            pred_score = np.concatenate([(pos_prob).cpu().numpy(), (neg_prob).cpu().numpy()])
            pred_label = pred_score > 0.5
            true_label = np.concatenate([np.ones(size), np.zeros(size)])

            val_acc.append((pred_label == true_label).mean())
            val_ap.append(average_precision_score(true_label, pred_score))
            # val_f1.append(f1_score(true_label, pred_label))
            val_auc.append(roc_auc_score(true_label, pred_score))
    # data ={}
    # data['src'] = src
    # data['dst'] = dst
    # data['dst_fake'] = dst_fake
    # print(len(src), len(dst_fake))
    # sio.savemat(file_name, data)
    return np.mean(val_acc), np.mean(val_ap), None, np.mean(val_auc)