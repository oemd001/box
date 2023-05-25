import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.datasets import load_iris
import socket

class SimpleModel(nn.Module):
    def __init__(self):
        super(SimpleModel, self).__init__()
        self.fc = nn.Linear(4, 3)

    def forward(self, x):
        return self.fc(x)

def get_ip_address():
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        # doesn't even have to be reachable
        s.connect(('10.255.255.255', 1))
        IP = s.getsockname()[0]
    except Exception:
        IP = '127.0.0.1'
    finally:
        s.close()
    return IP

def main(rank, world_size):
    print(f"Initializing process group for rank {rank} out of {world_size} total processes.")
    dist.init_process_group(
        "gloo",
        rank=rank,
        world_size=world_size,
        init_method='tcp://ec2-13-56-161-92.us-west-1.compute.amazonaws.com:6436' 
    )
    print("Process group initialized. Connected to master node.")
    # rest of your code

if __name__ == "__main__":
    world_size = 2 
    print("Spawning subprocesses...")
    torch.multiprocessing.spawn(main, args=(world_size,), nprocs=world_size, join=True)
    print("All subprocesses have been spawned.")
