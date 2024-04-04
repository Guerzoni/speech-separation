import torch
from itertools import permutations
import torch.nn as nn

def mse(x, s, eps=1e-8):

    def l2norm(mat, keepdim=False):
        return torch.norm(mat, dim=-1, keepdim=keepdim)

    if x.shape != s.shape:
        raise RuntimeError(
            "Dimention mismatch when calculate mse, {} vs {}".format(
                x.shape, s.shape))

    t = torch.sum((x - s) ** 2) / len(x)
    #print(t)
    return t

def mse_loss(ests, egs):
    refs = egs["ref"]
    num_spks = len(refs)

    def mse_loss_n(permute):
        return sum(
            [mse(ests[s], refs[t]) for s,t in enumerate(permute)]) / len(permute)

    N = egs["mix"].size(0)
    mse_mat = torch.stack(
        [mse_loss_n(p) for p in permutations(range(num_spks))])
    min_permutt, _ = torch.max(mse_mat, dim=0)
    return torch.sum(min_permutt) / N

if __name__ == "__main__":
    a_t = torch.tensor([4, 3, 3], dtype=torch.float32)
    b_t = torch.tensor([5, 4, 6], dtype=torch.float32)
    mse(a_t,b_t)
