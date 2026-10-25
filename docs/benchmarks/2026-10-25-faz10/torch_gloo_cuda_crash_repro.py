import sys
import torch
import torch.distributed as dist
import torch.multiprocessing as mp


def w(rank, n):
    dist.init_process_group("gloo", init_method="tcp://127.0.0.1:29761", rank=rank, world_size=n)
    x = torch.ones(1024, device="cuda")
    print(rank, "cuda all_reduce basliyor", flush=True)
    dist.all_reduce(x)
    print(rank, "cuda all_reduce tamam", x[0].item(), flush=True)
    dist.destroy_process_group()


if __name__ == "__main__":
    mp.spawn(w, args=(2,), nprocs=2, join=True)
