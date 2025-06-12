import pandas as pd
import numpy as np
import h5py
import torch
from torch_geometric.data import Data
from torch_geometric.utils import dense_to_sparse

res_to_id = {
    "K": 1,
    "A": 2,
    "R": 3,
    "N": 4,
    "D": 5,
    "C": 6,
    "Q": 7,
    "E": 8,
    "G": 9,
    "H": 10,
    "I": 11,
    "L": 12,
    "M": 13,
    "F": 14,
    "P": 15,
    "S": 16,
    "T": 17,
    "W": 18,
    "Y": 19,
    "V": 20
}
# 获取预训练的数据，即h5的emb数据
h5_protein_file = "data/portT5/OGly/per_residue_embeddings.h5"

emb_file = h5py.File(h5_protein_file, "r")

uniprot_id = []
labels = []
dataset = []
for key in emb_file.keys():
    print(key)
    # print(key[-1])
    uniprot_id.append(key)
    labels.append(key[-1])
    emb_data = np.array(emb_file[key])
    dataset.append(emb_data)
    break

# 构建图的节点特征数据
emb = np.array(dataset)


# print(labels)
# print(emb.shape)


def run(df):
    train_ls = []
    test_ls = []
    val_ls = []
    for _, record in df.iterrows():
        seq = record['seq']  # len(seq) = 51

        unique_id = record['unique_id'].split(';')
        uniprot = unique_id[0]
        label = int(unique_id[1])

        n_seq = int((len(seq) - 1) / 2)
        emb = torch.tensor(emb_file[record['unique_id']], dtype=torch.float32)

        x = [res_to_id[res] for res in seq]
        x = torch.tensor(x, dtype=torch.int32).unsqueeze(1)
        # 构建图的边特征数据
        n = len(seq)
        edge_index1 = dense_to_sparse(torch.ones((n, n)))[0]
        a = torch.zeros((n, n))
        a[range(n), np.arange(n)] = 1
        a[range(n - 1), np.arange(n - 1) + 1] = 1
        a[np.arange(n - 1) + 1, np.arange(n - 1)] = 1
        idx = int(n / 2)
        a[[idx] * n, range(n)] = 1
        edge_index2 = dense_to_sparse(a)[0]

        data = Data(x=x,
                    edge_index1=edge_index1,
                    edge_index2=edge_index2,
                    emb=emb,
                    seq=seq,
                    uniprot=uniprot,
                    unique_id=';'.join(unique_id),
                    y=torch.tensor(label, dtype=torch.float32)
                    )
        group = record['set']
        if group == 'train':
            train_ls.append(data)
        elif group == 'val':
            val_ls.append(data)
        elif group == 'test':
            test_ls.append(data)
        else:
            raise Exception('Unknown data group')
    return train_ls, val_ls, test_ls


# 生成构建好的数据文件 pt文件
df = pd.read_csv('data/ogly/csv/O_Gly_Data_cross.csv')
save_path = f'data/ogly/input/seq_data_cross.pt'
data_ls = run(df)
torch.save(data_ls, save_path)
print('Success!')
# train_ls, test_ls = run(df)
# print('-------train_ls:-------\n', train_ls)
# print('-------test_ls:--------\n', test_ls)
"""
- per_protein_embeddings:
Data(x=[51, 1], 
     y=1.0, 
     edge_index1=[2, 2601], 
     edge_index2=[2, 199], 
     emb=[1024], 
     seq='LLRGVYAYGFERPSAIQQRAIMPVIKGHDVIAQAQSGTGKTATFSISVLQK', 
     uniprot='KlaPids0', 
     unique_id='KlaPids0;1'
     )

- per_residue_embeddings:
Data(x=[51, 1],
     y=1.0, 
     edge_index1=[2, 2601], 
     edge_index2=[2, 199], 
     emb=[51, 1024], 
     seq='LLRGVYAYGFERPSAIQQRAIMPVIKGHDVIAQAQSGTGKTATFSISVLQK', 
     uniprot='KlaPids0', 
     unique_id='KlaPids0;1'
     )
"""
