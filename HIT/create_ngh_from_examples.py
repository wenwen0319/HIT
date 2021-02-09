idx1 = e_idx_l_pos_1[train_pos_idx]
idx2 = e_idx_l_pos_2[train_pos_idx]
idx3 = e_idx_l_pos_3[train_pos_idx]
ts_l_1 = ts_l[idx1]
ts_l_2 = ts_l[idx2]
ts_l_3 = ts_l[idx3]

# process train samples
for x, y, z, eidx_1, eidx_2, eidx_3, ts1, ts2, ts3 in zip(train_src_1_l_pos, train_src_2_l_pos, train_dst_l_pos, idx1, idx2, idx3, ts_l_1, ts_l_2, ts_l_3):
    partial_adj_list[x].append((y, eidx_1, ts1))
    partial_adj_list[y].append((x, eidx_1, ts1))

    partial_adj_list[y].append((z, eidx_2, ts2))
    partial_adj_list[z].append((y, eidx_2, ts2))

    if mode=='ffw':
        # for src, dst, eidx, ts in zip(src_1_l_pos, dst_l, e_idx_l_pos_3, ts_l_3):
        partial_adj_list[x].append((z, eidx_3, ts3))
        partial_adj_list[z].append((x, eidx_3, ts3))
    elif mode=='cycle':
        # for src, dst, eidx, ts in zip(dst_l, src_1_l_pos, e_idx_l_pos_3, ts_l_3):
        partial_adj_list[z].append((x, eidx_3, ts3))
        partial_adj_list[x].append((z, eidx_3, ts3))

if args.negative == 'NegaWedge':
    # for src, dst, eidx, ts in zip(train_src_1_l_neg, train_src_2_l_neg, train_e_idx_l_neg, train_ts_l_neg):
    #     partial_adj_list[src].append((dst, eidx, ts))
    #     partial_adj_list[dst].append((src, eidx, ts))
    # for src, dst, eidx, ts in zip(train_src_2_l_neg, train_dst_l_neg, train_e_idx_l_neg, train_ts_l_neg):
    #     partial_adj_list[src].append((dst, eidx, ts))
    #     partial_adj_list[dst].append((src, eidx, ts))
    idx1 = e_idx_l_neg_1[train_neg_idx]
    idx2 = e_idx_l_neg_2[train_neg_idx]
    ts_l_1 = ts_l[idx1]
    ts_l_2 = ts_l[idx2]
    for x, y, z, eidx_1, eidx_2, ts1, ts2 in zip(train_src_1_l_neg, train_src_2_l_neg, train_dst_l_neg, idx1, idx2, ts_l_1, ts_l_2):
        partial_adj_list[x].append((y, eidx_1, ts1))
        partial_adj_list[y].append((x, eidx_1, ts1))

        partial_adj_list[y].append((z, eidx_2, ts2))
        partial_adj_list[z].append((y, eidx_2, ts2))

elif args.negative == 'Nega20':
    idx1 = e_idx_l_neg_1[train_neg_idx]
    idx2 = e_idx_l_neg_2[train_neg_idx]
    idx3 = e_idx_l_neg_3[train_neg_idx]
    ts_l_1 = ts_l[idx1]
    ts_l_2 = ts_l[idx2]
    ts_l_3 = ts_l[idx3]
    for x, y, z, eidx_1, eidx_2, eidx_3, ts1, ts2, ts3 in zip(train_src_1_l_neg, train_src_2_l_neg, train_dst_l_neg, idx1, idx2, idx3, ts_l_1, ts_l_2, ts_l_3):
        partial_adj_list[x].append((y, eidx_1, ts1))
        partial_adj_list[y].append((x, eidx_1, ts1))

        partial_adj_list[y].append((z, eidx_2, ts2))
        partial_adj_list[z].append((y, eidx_2, ts2))

        if mode=='ffw':
            # for src, dst, eidx, ts in zip(src_1_l_pos, dst_l, e_idx_l_pos_3, ts_l_3):
            partial_adj_list[x].append((z, eidx_3, ts3))
            partial_adj_list[z].append((x, eidx_3, ts3))
        elif mode=='cycle':
            # for src, dst, eidx, ts in zip(dst_l, src_1_l_pos, e_idx_l_pos_3, ts_l_3):
            partial_adj_list[z].append((x, eidx_3, ts3))
            partial_adj_list[x].append((z, eidx_3, ts3))
    
elif args.negative == 'rand':
    pass

idx1 = e_idx_l_pos_1[val_pos_idx]
idx2 = e_idx_l_pos_2[val_pos_idx]
idx3 = e_idx_l_pos_3[val_pos_idx]
ts_l_1 = ts_l[idx1]
ts_l_2 = ts_l[idx2]
ts_l_3 = ts_l[idx3]
# process val samples
for x, y, z, eidx_1, eidx_2, eidx_3, ts1, ts2, ts3 in zip(val_src_1_l_pos, val_src_2_l_pos, val_dst_l_pos, idx1, idx2, idx3, ts_l_1, ts_l_2, ts_l_3):
    partial_adj_list[x].append((y, eidx_1, ts1))
    partial_adj_list[y].append((x, eidx_1, ts1))

    partial_adj_list[y].append((z, eidx_2, ts2))
    partial_adj_list[z].append((y, eidx_2, ts2))

    if mode=='ffw':
        # for src, dst, eidx, ts in zip(src_1_l_pos, dst_l, e_idx_l_pos_3, ts_l_3):
        partial_adj_list[x].append((z, eidx_3, ts3))
        partial_adj_list[z].append((x, eidx_3, ts3))
    elif mode=='cycle':
        # for src, dst, eidx, ts in zip(dst_l, src_1_l_pos, e_idx_l_pos_3, ts_l_3):
        partial_adj_list[z].append((x, eidx_3, ts3))
        partial_adj_list[x].append((z, eidx_3, ts3))
# for src, dst, eidx, ts in zip(val_src_1_l_neg, val_src_2_l_neg, val_e_idx_l_neg, val_ts_l_neg):
#     partial_adj_list[src].append((dst, eidx, ts))
#     partial_adj_list[dst].append((src, eidx, ts))

if args.negative == 'NegaWedge':
    # for src, dst, eidx, ts in zip(train_src_1_l_neg, train_src_2_l_neg, train_e_idx_l_neg, train_ts_l_neg):
    #     partial_adj_list[src].append((dst, eidx, ts))
    #     partial_adj_list[dst].append((src, eidx, ts))
    # for src, dst, eidx, ts in zip(train_src_2_l_neg, train_dst_l_neg, train_e_idx_l_neg, train_ts_l_neg):
    #     partial_adj_list[src].append((dst, eidx, ts))
    #     partial_adj_list[dst].append((src, eidx, ts))
    idx1 = e_idx_l_neg_1[val_neg_idx]
    idx2 = e_idx_l_neg_2[val_neg_idx]
    ts_l_1 = ts_l[idx1]
    ts_l_2 = ts_l[idx2]
    for x, y, z, eidx_1, eidx_2, ts1, ts2 in zip(val_src_1_l_neg, val_src_2_l_neg, val_dst_l_neg, idx1, idx2, ts_l_1, ts_l_2):
        partial_adj_list[x].append((y, eidx_1, ts1))
        partial_adj_list[y].append((x, eidx_1, ts1))

        partial_adj_list[y].append((z, eidx_2, ts2))
        partial_adj_list[z].append((y, eidx_2, ts2))

elif args.negative == 'Nega20':
    idx1 = e_idx_l_neg_1[val_neg_idx]
    idx2 = e_idx_l_neg_2[val_neg_idx]
    idx3 = e_idx_l_neg_3[val_neg_idx]
    ts_l_1 = ts_l[idx1]
    ts_l_2 = ts_l[idx2]
    ts_l_3 = ts_l[idx3]
    for x, y, z, eidx_1, eidx_2, eidx_3, ts1, ts2, ts3 in zip(val_src_1_l_neg, val_src_2_l_neg, val_dst_l_neg, idx1, idx2, idx3, ts_l_1, ts_l_2, ts_l_3):
        partial_adj_list[x].append((y, eidx_1, ts1))
        partial_adj_list[y].append((x, eidx_1, ts1))

        partial_adj_list[y].append((z, eidx_2, ts2))
        partial_adj_list[z].append((y, eidx_2, ts2))

        if mode=='ffw':
            # for src, dst, eidx, ts in zip(src_1_l_pos, dst_l, e_idx_l_pos_3, ts_l_3):
            partial_adj_list[x].append((z, eidx_3, ts3))
            partial_adj_list[z].append((x, eidx_3, ts3))
        elif mode=='cycle':
            # for src, dst, eidx, ts in zip(dst_l, src_1_l_pos, e_idx_l_pos_3, ts_l_3):
            partial_adj_list[z].append((x, eidx_3, ts3))
            partial_adj_list[x].append((z, eidx_3, ts3))
    
elif args.negative == 'rand':
    pass

partial_ngh_finder = NeighborFinder(partial_adj_list, bias=args.bias, use_cache=NGH_CACHE)

ts_l_1 = ts_l[e_idx_l_pos_1]
ts_l_2 = ts_l[e_idx_l_pos_2]
ts_l_3 = ts_l[e_idx_l_pos_3]
for x, y, z, eidx_1, eidx_2, eidx_3, ts1, ts2, ts3 in zip(src_1_l_pos, src_2_l_pos, dst_l_pos, e_idx_l_pos_1, e_idx_l_pos_2, e_idx_l_pos_3, ts_l_1, ts_l_2, ts_l_3):
    full_adj_list[x].append((y, eidx_1, ts1))
    full_adj_list[y].append((x, eidx_1, ts1))

    full_adj_list[y].append((z, eidx_2, ts2))
    full_adj_list[z].append((y, eidx_2, ts2))

    if mode=='ffw':
        # for src, dst, eidx, ts in zip(src_1_l_pos, dst_l, e_idx_l_pos_3, ts_l_3):
        full_adj_list[x].append((z, eidx_3, ts3))
        full_adj_list[z].append((x, eidx_3, ts3))
    elif mode=='cycle':
        # for src, dst, eidx, ts in zip(dst_l, src_1_l_pos, e_idx_l_pos_3, ts_l_3):
        full_adj_list[z].append((x, eidx_3, ts3))
        full_adj_list[x].append((z, eidx_3, ts3))
# for src, dst, eidx, ts in zip(src_1_l_neg, src_2_l_neg, e_idx_l_neg_1, ts_l_neg):
#     full_adj_list[src].append((dst, eidx, ts))
#     full_adj_list[dst].append((src, eidx, ts))
if args.negative == 'NegaWedge':
    # for src, dst, eidx, ts in zip(train_src_1_l_neg, train_src_2_l_neg, train_e_idx_l_neg, train_ts_l_neg):
    #     partial_adj_list[src].append((dst, eidx, ts))
    #     partial_adj_list[dst].append((src, eidx, ts))
    # for src, dst, eidx, ts in zip(train_src_2_l_neg, train_dst_l_neg, train_e_idx_l_neg, train_ts_l_neg):
    #     partial_adj_list[src].append((dst, eidx, ts))
    #     partial_adj_list[dst].append((src, eidx, ts))
    ts_l_1 = ts_l[e_idx_l_neg_1]
    ts_l_2 = ts_l[e_idx_l_neg_2]
    for x, y, z, eidx_1, eidx_2, ts1, ts2 in zip(src_1_l_neg, src_2_l_neg, dst_l_neg, e_idx_l_neg_1, e_idx_l_neg_2, ts_l_1, ts_l_2):
        partial_adj_list[x].append((y, eidx_1, ts1))
        partial_adj_list[y].append((x, eidx_1, ts1))

        partial_adj_list[y].append((z, eidx_2, ts2))
        partial_adj_list[z].append((y, eidx_2, ts2))

elif args.negative == 'Nega20':
    ts_l_1 = ts_l[e_idx_l_neg_1]
    ts_l_2 = ts_l[e_idx_l_neg_2]
    ts_l_3 = ts_l[e_idx_l_neg_3]
    for x, y, z, eidx_1, eidx_2, eidx_3, ts1, ts2, ts3 in zip(src_1_l_neg, src_2_l_neg, dst_l_neg, e_idx_l_neg_1, e_idx_l_neg_2, e_idx_l_neg_3, ts_l_1, ts_l_2, ts_l_3):
        partial_adj_list[x].append((y, eidx_1, ts1))
        partial_adj_list[y].append((x, eidx_1, ts1))

        partial_adj_list[y].append((z, eidx_2, ts2))
        partial_adj_list[z].append((y, eidx_2, ts2))

        if mode=='ffw':
            # for src, dst, eidx, ts in zip(src_1_l_pos, dst_l, e_idx_l_pos_3, ts_l_3):
            partial_adj_list[x].append((z, eidx_3, ts3))
            partial_adj_list[z].append((x, eidx_3, ts3))
        elif mode=='cycle':
            # for src, dst, eidx, ts in zip(dst_l, src_1_l_pos, e_idx_l_pos_3, ts_l_3):
            partial_adj_list[z].append((x, eidx_3, ts3))
            partial_adj_list[x].append((z, eidx_3, ts3))
    
elif args.negative == 'rand':
    pass
full_ngh_finder = NeighborFinder(full_adj_list, bias=args.bias, use_cache=NGH_CACHE)
