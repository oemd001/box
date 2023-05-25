import torch
import torch.distributed as dist
import signal
import sys

stop = False

def signal_handler(sig, frame):
    print('You pressed Ctrl+C!')
    global stop
    stop = True

signal.signal(signal.SIGINT, signal_handler)

def main(rank, world_size):
    print(f"Initializing process group for rank {rank} out of {world_size} total processes.")
    dist.init_process_group(
        "gloo",
        rank=rank,
        world_size=world_size,
        init_method='tcp://127.0.0.1:6435'
    )
    print("Process group initialized. Ready to accept connections.")

if __name__ == "__main__":
    world_size = 2
    print("Spawning subprocesses...")
    torch.multiprocessing.spawn(main, args=(world_size,), nprocs=world_size, join=True)
    print("All subprocesses have been spawned.")
    
    while not stop:
        pass  # keep the script running

