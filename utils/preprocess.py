import torch
from tqdm import tqdm
from torch_geometric.utils import degree


def compute_mean(dataloader, property="y"):
    """Mean"""
    s = 0
    N = 0
    for graph in tqdm(dataloader):
        x = graph[property]
        if len(x.shape) == 2:
            x = torch.linalg.vector_norm(x, dim=-1)

        s += x.sum()
        N += x.size(0)

    return (s / N).item()


def compute_mad(dataloader, mean, property="y"):
    """Mean Absolute Deviation"""
    s = 0
    N = 0
    for graph in tqdm(dataloader):
        x = graph[property]
        if len(x.shape) == 2:
            x = torch.linalg.vector_norm(x, dim=-1)
        s += (x - mean).abs().sum()
        N += x.size(0)

    return (s / N).item()


def compute_rms(dataloader, property="y"):
    """Root Mean Square"""
    s = 0
    N = 0
    for graph in tqdm(dataloader):
        x = graph[property]
        if len(x.shape) == 2:
            x = torch.linalg.vector_norm(x, dim=-1)

        s += x.pow(2).sum()
        N += x.size(0)
    return (s / N).sqrt().item()


def compute_rmsd(dataloader, mean, property="y"):
    """Root Mean Square Deviation"""
    s = 0
    N = 0
    for graph in tqdm(dataloader):
        x = graph[property]
        if len(x.shape) == 2:
            x = torch.linalg.vector_norm(x, dim=-1)
        s += (x - mean).pow(2).sum()
        N += x.size(0)

    return (s / N).sqrt().item()


def compute_avg_num_neighbours(dataloader, direction="incoming"):
    """Average number of neighbours"""
    s = 0
    N = 0
    for graph in tqdm(dataloader):
        if direction == "incoming":
            s += degree(graph.edge_index[1], graph.num_nodes).sum()
        elif direction == "outgoing":
            s += degree(graph.edge_index[0], graph.num_nodes).sum()
        N += graph.num_nodes

    return (s / N).item()


def compute_one_hot_dict(train_dataset, property="z"):
    """Compute a one-hot dictionary for a property"""
    uniques = set()
    print("Creating one-hot dictionary")
    for item in tqdm(train_dataset):
        uniques.update(set(item[property].numpy()))

    return {z: i for z, i in zip(uniques, range(len(uniques)))}
