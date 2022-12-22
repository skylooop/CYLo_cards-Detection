import torch
from tqdm import tqdm

from oml.datasets.base import DatasetWithLabels
from oml.losses.triplet import TripletLossWithMiner
from oml.miners.inbatch_all_tri import AllTripletsMiner
from oml.models.vit.vit import ViTExtractor
from oml.samplers.balance import BalanceSampler
from oml.utils.download_mock_dataset import download_mock_dataset

import pyrallis
from dataclasses import field, dataclass


@dataclass
class CFG:
    path_to_ds: str = field('../datasets/labeled/train')
    path_to_csv: str = field('../datasets/labeled/labeled.csv')
    
def main(args):
    model = ViTExtractor("vits16_dino", arch="vits16",
                        normalise_features=False).train()

    optimizer = torch.optim.SGD(model.parameters(), lr=1e-6)

    train_dataset = DatasetWithLabels(args.path_to_csv, dataset_root=args.path_to_ds)
    criterion = TripletLossWithMiner(margin=0.1, miner=AllTripletsMiner())
    
    sampler = BalanceSampler(train_dataset.get_labels(), n_labels=2, n_instances=2)
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_sampler=sampler)

    for batch in tqdm(train_loader):
        embeddings = model(batch["input_tensors"])
        loss = criterion(embeddings, batch["labels"])
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()


if __name__ == "__main__":
    args = pyrallis.parse(CFG)
    main(args)