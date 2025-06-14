import torch
from types import SimpleNamespace
import numpy as np
from utils import TrainProcessor
import random
from torch_geometric.loader import DataLoader
# from model import GNNTrans
from new_model import GNNTrans
import os

random.seed(42)
np.random.seed(42)

if __name__ == '__main__':
    # args
    args = {
        'epochs': 100,
        'batch_size': 64,
        'device': 'cuda',
        'opt': 'adam',
        'opt_scheduler': 'step',
        'opt_decay_step': 20,
        'opt_decay_rate': 0.92,
        'weight_decay': 1e-4,
        'lr': 3e-5,
        'es_patience': 10,
        'save': True
    }
    args = SimpleNamespace(**args)
    print(args)
    train_ls, val_ls, test_ls = torch.load('./data/ogly/input/seq_data_cross.pt')
    train_data_loader = DataLoader(train_ls, batch_size=args.batch_size)
    val_data_loader = DataLoader(val_ls, batch_size=args.batch_size)
    test_data_loader = DataLoader(test_ls, batch_size=args.batch_size)
    for i in range(5):
        model = GNNTrans(input_dim=1024, hidden_dim=256,
                         num_layers=3)  # hidden_dim:[64, 128, 256, 512], num_layers:[2,3]
        model.to(args.device)
        print(model)

        train_val = TrainProcessor(
            model=model,
            loaders=[train_data_loader, val_data_loader, test_data_loader],
            args=args
        )
        best_model, test_metrics = train_val.train()
        # best_model, test_metrics = train_val.train_cross()
        # print('test loss: {:5f}; test acc: {:4f}; test auroc: {:4f}; test auprc: {:.4f}'.format(
        #     test_metrics.loss, test_metrics.acc, test_metrics.auroc, test_metrics.auprc))
        print(
            'test loss: {:5f}; test acc: {:4f}; test auroc: {:4f}; test auprc: {:.4f}; '
            'test f1: {:4f}; test mcc: {:4f};test sn: {:4f};test sp: {:4f}'.format(
                test_metrics.loss, test_metrics.acc, test_metrics.auroc, test_metrics.auprc,
                test_metrics.f1, test_metrics.mcc, test_metrics.sn, test_metrics.sp))

        if args.save:
            save_dir = f'result/O_seq_data_val'
            if not os.path.exists(save_dir):
                os.mkdir(save_dir)

            save_path = os.path.join(save_dir,
                                     '{}_acc{:.4f}_roc{:.4f}_prc{:.4f}'.format(i,
                                                                               test_metrics.acc,
                                                                               test_metrics.auroc,
                                                                               test_metrics.auprc,
                                                                               ))
            torch.save(best_model.state_dict(), save_path)
