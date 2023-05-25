import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.datasets import load_iris

class SimpleModel(nn.Module):
    def __init__(self):
        super(SimpleModel, self).__init__()
        self.fc = nn.Linear(4, 3)

    def forward(self, x):
        return self.fc(x)

def main(rank, world_size):
    dist.init_process_group(
        "gloo",
        rank=rank,
        world_size=world_size,
        init_method='tcp://ec2-13-56-161-92.us-west-1.compute.amazonaws.com:6436'
    )

    # Load the iris dataset
    iris = load_iris()
    X = iris['data']
    y = iris['target']

    # Create a tensor dataset and dataloader
    tensor_x = torch.Tensor(X) 
    tensor_y = torch.Tensor(y)
    dataset = TensorDataset(tensor_x,tensor_y)
    dataloader = DataLoader(dataset, batch_size=30)

    model = SimpleModel().to(rank)
    model = DistributedDataParallel(model, device_ids=[rank])

    optimizer = optim.SGD(model.parameters(), lr=0.01)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(100): # training loop
        for data, target in dataloader:
            data = data.to(rank)
            target = target.to(rank).long()
            optimizer.zero_grad()
            outputs = model(data)
            loss = criterion(outputs, target)
            loss.backward()
            optimizer.step()

if __name__ == "__main__":
    world_size = 2 
    torch.multiprocessing.spawn(main, args=(world_size,), nprocs=world_size, join=True)
