import torch
import numpy as np


def manual_seed(seed):
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    # print(torch.backends.cudnn.deterministic)
    # print(torch.backends.cudnn.benchmark)
    np.random.seed(seed)        # checked, still work


# def net2gpu(net, device):
#     net.atom_embed.to(device)
#     if net.gnn is not None:
#         net.gnn.to(device)
#     if net.attn is not None:
#         net.attn.to(device)
#     net.readout.to(device)
#     net.mlp.to(device)
#     if net.head_embed is not None:
#         net.head_embed.to(device)
#     if net.encoder is not None:
#         net.encoder.to(device)

def net2gpu(net, device):
    net.atom_embed.to(device)
    net.bond_embed.to(device)
    net.backbone.to(device)
    net.readout.to(device)
    net.pred.to(device)
    # if net.gnn is not None:
    #     net.gnn.to(device)
    # if net.attn is not None:
    #     net.attn.to(device)
    # net.readout.to(device)
    # net.mlp.to(device)
    # if net.head_embed is not None:
    #     net.head_embed.to(device)
    # if net.encoder is not None:
    #     net.encoder.to(device)


def normalize_bond_mat(bond_mat, mask):
    bond_mat *= torch.matmul(mask[..., None], mask[:, None])
    # print(bond_mat[0, 0, :])
    # print(bond_mat[0, -1, :])
    bond_sum = torch.sum(bond_mat, dim=1)[:, None]
    bond_sum[bond_sum == 0] = 1
    bond_mat /= bond_sum
    # bond_mat /= torch.sum(bond_sum) / torch.sum(mask)
    # print(torch.sum(bond_mat, dim=1)[0, ...])
    # print(torch.sum(bond_mat, dim=2)[0, ...])
    # quit()
    return bond_mat
