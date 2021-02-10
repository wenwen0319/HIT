import math
import torch
import numpy as np
from sklearn.metrics import average_precision_score
from sklearn.metrics import f1_score
# from sklearn.metrics import roc_auc_score
from sklearn.metrics import confusion_matrix
import scipy.io as sio
import motif_checking
from interpretation import process_pattern
from tqdm import tqdm
import utils

# def eval_one_epoch(hint, tgan, sampler, src, dst, ts, label, val_e_idx_l=None, file_name='', dic1={}, dic2={}):
def eval_one_epoch(hint, model, dataset, val_flag='val', interpretation=False):
    val_acc, val_ap, val_f1, val_auc = [], [], [], []
    y_true, y_pred, y_score, y_one_hot_np = None, None, None, None
    dataset.reset()
    model.test = True
    if interpretation:
        roc_auc_score = utils.roc_auc_score_single
    else:
        roc_auc_score = utils.roc_auc_score_multi
    if val_flag == 'train':
        num_test_instance = dataset.get_size()
        get_sample = dataset.train_samples
        dataset.initialize()
    elif val_flag == 'val':
        num_test_instance = dataset.get_val_size()
        get_sample = dataset.val_samples
        dataset.initialize_val()
    elif val_flag == 'test':
        num_test_instance = dataset.get_test_size()
        get_sample = dataset.test_samples
        dataset.initialize_test()
    with torch.no_grad():
        model = model.eval()
        TEST_BATCH_SIZE = dataset.bs
        # num_test_instance = len(src_1)
        num_test_batch = math.ceil(num_test_instance / TEST_BATCH_SIZE)
        num_class = dataset.num_class
        walk_pattern = None
        walk_score = None
        walk_pattern_total = None
        walk_score_total = None
        for k in tqdm(range(num_test_batch)):
        # for k in tqdm(range(1)):
            # percent = 100 * k / num_test_batch
            # if k % int(0.2 * num_test_batch) == 0:
            #     logger.info('{0} progress: {1:10.4f}'.format(hint, percent))
            # s_idx = k * TEST_BATCH_SIZE
            # e_idx = min(num_test_instance, s_idx + TEST_BATCH_SIZE)
            # if s_idx == e_idx:
            #     continue
            
            src_1_l_cut, src_2_l_cut, dst_l_cut, ts_l_cut, e_l_cut, true_label = get_sample()
            y_one_hot = torch.nn.functional.one_hot(torch.from_numpy(true_label).long(), num_classes=model.num_class).float().cpu().numpy()
            # true_label[true_label == 1] = 0
            # true_label[true_label == 2] = 3
            # print(true_label)
            # true_label = torch.from_numpy(true_label).long().to(device)
            # if dst_fake == []:
            #     dst_fake = np.copy(dst_l_fake)
            # else:
            # dst_fake = np.concatenate([dst_fake, dst_l_fake])

            # pos_pattern, neg_pattern, pos_prob, neg_prob = tgan.contrast(src_1_l_cut, src_2_l_cut, dst_l_cut, dst_l_fake, ts_l_cut, e_l_cut)
            # try:
            if interpretation:
                pred_score, pattern_score = model.contrast(src_1_l_cut, src_2_l_cut, dst_l_cut, ts_l_cut, e_l_cut)
                # interpretation
                src_1_walks_score, src_2_walks_score, tgt_walks_score, src_1_walk_pattern, src_2_walk_pattern, tgt_walk_pattern = pattern_score
                # deal with scores and pattern
                # reshape, cpu, numpy scores
                src_1_walks_score = src_1_walks_score.detach().cpu().numpy()
                src_2_walks_score = src_2_walks_score.detach().cpu().numpy()
                tgt_walks_score = tgt_walks_score.detach().cpu().numpy()
                walk_pattern = np.concatenate([src_1_walk_pattern.reshape(-1), src_2_walk_pattern.reshape(-1),tgt_walk_pattern.reshape(-1)])
                if walk_pattern_total is None:
                    walk_pattern_total = walk_pattern
                else:
                    walk_pattern_total = np.concatenate([walk_pattern_total, walk_pattern])
                walk_score = np.concatenate([src_1_walks_score, src_2_walks_score, tgt_walks_score])
                if walk_score_total is None:
                    walk_score_total = walk_score
                else:
                    walk_score_total = np.concatenate([walk_score_total, walk_score])
                # print(model.position_encoder.pattern)
                # _, _, result = process_pattern(walk_pattern, walk_score)
                # print(result)
                # _, _, _, result1 = process_pattern(, src_1_walks_score.reshape(-1))
                # _, _, _, result2 = process_pattern(, src_2_walks_score.reshape(-1))
                # _, _, _, result3 = process_pattern(, tgt_walks_score.reshape(-1))
            else:
                pred_score, _ = model.contrast(src_1_l_cut, src_2_l_cut, dst_l_cut, ts_l_cut, e_l_cut)
            # except:
            #     continue
            # print('begin motif checking')
            # dic1, dic2 = motif_checking.get_pattern_score(dic1, dic2, pos_pattern, neg_pattern)
            # print('end motif checking')
            # sio.savemat('./motif_checking/epoch{}_training mode {}'.format(epoch, mode))
            # print(dic1, dic2)
            pred_label = torch.argmax(pred_score, dim=1).cpu().detach().numpy()
            pred_score = torch.nn.functional.softmax(pred_score, dim=1).cpu().numpy()
            if y_pred is None:
                y_pred = np.copy(pred_label)
                y_true = np.copy(true_label)
                y_score = np.copy(pred_score)
                y_one_hot_np = y_one_hot
            else:
                y_pred = np.concatenate((y_pred, pred_label))
                y_true = np.concatenate((y_true, true_label))
                y_score = np.concatenate((y_score, pred_score))
                y_one_hot_np = np.concatenate((y_one_hot_np, y_one_hot))
            
            
            val_acc.append((pred_label == true_label).mean())
            val_ap.append(1)
            # val_auc.append(1)
            # val_ap.append(average_precision_score(true_label, pred_score))
            
    # data ={}
    # data['src'] = src
    # data['dst'] = dst
    # data['dst_fake'] = dst_fake
    # print(len(src), len(dst_fake))
    # sio.savemat(file_name, data)
    print(val_flag)
    # print(confusion_matrix(y_true, y_pred))

    cm = confusion_matrix(y_true, y_pred)
    print('Confusion Matrix')
    print(cm)
    if interpretation:
        _, _, result = process_pattern(walk_pattern_total, walk_score_total, pattern_dict=model.position_encoder.pattern, non_idx=model.num_layers*2)
        print('result')
        print(result)
        print('walk pattern')
        print(model.position_encoder.pattern)
        print('walk pattern number:', len(np.unique(walk_pattern_total)))
    # logger.info('confusion matrix: ')
    # logger.info(', '.join(str(r) for r in cm.reshape(1,-1)))
    # y_one_hot = torch.nn.functional.one_hot(torch.from_numpy(true_label).long(), num_classes=4).float().cpu().numpy()
    # true_label
    # val_auc = roc_auc_score(y_true, y_score, multi_class='ovo')
    # print(y_true, y_score)
    # print(y_true.size, y_score.size)
    val_auc = roc_auc_score(y_one_hot_np, y_score)

    return np.mean(val_acc), np.mean(val_ap), None, val_auc, cm