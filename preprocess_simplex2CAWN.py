# change the simplex version to the CAWN version
import numpy as np

file_name = 'NDC-substances'
fout = open('./processed/ml_' + file_name + '.csv', 'w')
fout.write(',u,i,ts,label,idx\n')
file_addr = './' + file_name + '/' + file_name

fin_nverts = open(file_addr + '-nverts.txt', 'r')
fin_simplices = open(file_addr + '-simplices.txt', 'r')
fin_times = open(file_addr + '-times.txt', 'r')

nverts = []
simplices = []
times = []
node2idx = {}
for i in fin_nverts:
    nverts.append(int(i))
count = 1
for i in fin_simplices:
    simplices.append(int(i))
    # if not(int(i) in node2idx):
    #     node2idx[int(i)] = count
    #     count += 1
last_time = -1
idx = -1
for i in fin_times:
    idx += 1
    if int(i) >= last_time:
        last_time = int(i)
    else:
        print("Time not monotune", last_time, int(i), nverts[idx])
    times.append(int(i))

print("First")

times = np.array(times)
y = np.argsort(times) 
print(y)

simplices_i = 0
edge_idx = 0
exist_solo = len(node2idx.keys())
node_bool = {}
node_list_total = []
for idx_nverts, nverts_i in enumerate(nverts):
    node_list_total.append(simplices[simplices_i: simplices_i + nverts_i])
    if nverts_i == 1: # there may be 1 simplex, which means doesn't have edge with other nodes, so remove them
        simplices_i += 1
        continue
    
    for i in simplices[simplices_i: simplices_i + nverts_i]:
        if not(i in node2idx):
            node2idx[i] = count
            count += 1
    
    simplices_i += nverts_i

for idx_y in y:
    node_list = node_list_total[idx_y]
    if len(node_list) == 1:
        continue
    for idx_st, st in enumerate(node_list[:-1]):
        for ed in node_list[idx_st+1:]:
            node_bool[node2idx[st]] = 1
            node_bool[node2idx[ed]] = 1
            fout.write("%s,%s,%s,%s,%s,%s\n" %(edge_idx, node2idx[st], node2idx[ed], times[idx_y], 0, edge_idx + 1))
            # if ( node2idx[st] == 1629) or (node2idx[ed] == 1629):
            #     print(1629, node2idx[1629])
            #     print("%s,%s,%s,%s,%s,%s\n" %(edge_idx, node2idx[st], node2idx[ed], times[idx_nverts], 0, edge_idx + 1))
            edge_idx += 1


print(len(node2idx.keys()), min(node2idx.values()), max(node2idx.values()))

print('solo', len(node_bool.keys()))
# for i in range(1, len(node2idx.keys())+1):
#     if i not in node2idx:

print(edge_idx)
fout.close()
fin_times.close()
fin_simplices.close()
fin_nverts.close()

rand_feat = np.zeros((count, 172))
np.save('./processed/ml_'+ file_name + '_node.npy', rand_feat)
rand_feat = np.zeros((edge_idx, 172))
np.save('./processed/ml_'+ file_name + '.npy', rand_feat)