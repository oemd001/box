import torch.distributed as dist

def main(rank, world_size):
    dist.init_process_group(
        "gloo",
        rank=rank,
        world_size=world_size,
        init_method='tcp://0.0.0.0:6436' # Listening on all addresses, port 6436
    )

if __name__ == "__main__":
    world_size = 2 
    torch.multiprocessing.spawn(main, args=(world_size,), nprocs=world_size, join=True)
