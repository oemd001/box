import torch
import torch.distributed as dist

def main(rank, world_size):
    print(f"Initializing process group for rank {rank} out of {world_size} total processes.")
    dist.init_process_group(
        "gloo",
        rank=rank,
        world_size=world_size,
        init_method='tcp://0.0.0.0:6436' 
    )
    print("Process group initialized. Ready to accept connections.")

if __name__ == "__main__":
    world_size = 2 
    print("Spawning subprocesses...")
    torch.multiprocessing.spawn(main, args=(world_size,), nprocs=world_size, join=True)
    print("All subprocesses have been spawned.")
    
    import time
    time.sleep(60)  # Keep the script running for 60 seconds
