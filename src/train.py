import argparse
import logging
import os
import sys

import torch
import torch.distributed as dist
import torch.nn as nn
import torchvision
import torchvision.models as models
import torchvision.transforms as transforms
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.utils.tensorboard import SummaryWriter

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
logger.addHandler(logging.StreamHandler(sys.stdout))

TRAIN_TRANSFORM = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.ColorJitter(
        brightness=0.1,
        contrast=0.1,
        saturation=0.1
    ),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    ),
])

TEST_TRANSFORM = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    ),
])


def net(version, local_rank, device):
    if version == 'resnet18':
        model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    elif version == 'resnet34':
        model = models.resnet34(weights=models.ResNet34_Weights.DEFAULT)
    elif version == 'resnet50':
        model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
    elif version == 'resnet101':
        model = models.resnet101(weights=models.ResNet101_Weights.DEFAULT)
    elif version == 'resnet152':
        model = models.resnet152(weights=models.ResNet152_Weights.DEFAULT)
    else:
        raise ValueError('Unknown version')
    dist.barrier()

    # Freeze all pretrained parameters
    for param in model.parameters():
        param.requires_grad = False

    # Unfreeze parameters of layer4
    for param in model.layer4.parameters():
        param.requires_grad = True

    # Replace the final fully connected layer
    num_features = model.fc.in_features
    model.fc = nn.Linear(num_features, 1)

    return DistributedDataParallel(model.to(device), device_ids=[local_rank])


class RegressionFolderValueDataset(torchvision.datasets.ImageFolder):
    def __init__(self, root, transform):
        super().__init__(root, transform=transform)
        self.values = [float(c) for c in self.classes]

    def __getitem__(self, i):
        img, cls_idx = super().__getitem__(i)
        return img, torch.tensor(self.values[cls_idx], dtype=torch.float32)


def create_data_loaders(train_data_path,
                        validation_data_path,
                        batch_size,
                        rank,
                        world_size,
                        loading_workers):
    train_dataset = RegressionFolderValueDataset(root=train_data_path, transform=TRAIN_TRANSFORM)
    validation_dataset = RegressionFolderValueDataset(root=validation_data_path, transform=TEST_TRANSFORM)

    train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank, shuffle=True)
    validation_sampler = DistributedSampler(validation_dataset, num_replicas=world_size, rank=rank, shuffle=False)

    return (
        DataLoader(
            train_dataset,
            batch_size=batch_size,
            sampler=train_sampler,
            num_workers=loading_workers,
            pin_memory=True,
            drop_last=True,
            persistent_workers=loading_workers > 0
        ),
        DataLoader(
            validation_dataset,
            batch_size=batch_size,
            sampler=validation_sampler,
            num_workers=loading_workers,
            pin_memory=True,
            persistent_workers=loading_workers > 0
        )
    )


def reduce_mean(total, count, device):
    t = torch.tensor([total, count], dtype=torch.float64, device=device)
    dist.all_reduce(t, op=dist.ReduceOp.SUM)
    return (t[0] / t[1]).item()


def train(model,
          train_loader,
          criterion,
          optimizer):
    model.train()
    device = next(model.parameters()).device

    total_loss = .0
    count = 0
    for idx, (inputs, target_value) in enumerate(train_loader):
        inputs, target_value = inputs.to(device, non_blocking=True), target_value.float().to(device, non_blocking=True)

        outputs = model(inputs)
        optimizer.zero_grad()
        loss = criterion(outputs.squeeze(1), target_value)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * inputs.size(0)
        count += inputs.size(0)

    return reduce_mean(total_loss, count, device)


@torch.no_grad()
def test(model, test_loader, criterion):
    model.eval()
    device = next(model.parameters()).device

    total_loss = .0
    count = 0
    for idx, (inputs, target_value) in enumerate(test_loader):
        inputs, target_value = inputs.to(device, non_blocking=True), target_value.long().to(device, non_blocking=True)

        outputs = model(inputs)
        outputs = round_to_closest_non_negative_integer(outputs)
        loss = criterion(outputs.squeeze(1), target_value)
        total_loss += loss.item() * inputs.size(0)
        count += inputs.size(0)

    return reduce_mean(total_loss, count, device)


def round_to_closest_non_negative_integer(batch):
    return batch.round().clamp(min=0).long()


def main(args):
    # Setup distributed training
    dist.init_process_group(backend="nccl")
    rank, world_size = dist.get_rank(), dist.get_world_size()
    local_rank = int(os.environ["LOCAL_RANK"])

    is_leader = rank == 0
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)

    logger.info(f'Started training. World size:{world_size}, Local rank:{local_rank}, World rank:{rank}')

    # Initiate the model
    model = net(args.resnet_version, local_rank, device)

    # Prepare the data loaders
    train_loader, validation_loader = create_data_loaders(
        args.train_data_dir,
        args.validation_data_dir,
        args.batch_size,
        rank,
        world_size,
        args.loading_workers
    )

    # Define loss function
    loss_criterion = nn.L1Loss()

    # Define optimizer
    optimizer = torch.optim.Adam((p for p in model.parameters() if p.requires_grad), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)

    logger.info(
        "Starting training with the following parameters: "
        f"Batch size: {args.batch_size} "
        f"Epochs: {args.epochs} "
        f"Learning rate: {args.lr}"
    )

    if is_leader:
        tensorboard_writer = SummaryWriter(log_dir=os.path.join(args.output_dir, 'tensorboard', 'metrics'))
    else:
        tensorboard_writer = None
    dist.barrier()

    no_improvement_epochs = 0
    best_loss = float('inf')
    best_model_index = None
    # Training loop
    for epoch in range(args.epochs):
        logger.info(f"Epoch {epoch + 1}/{args.epochs}")

        train_loader.sampler.set_epoch(epoch)

        train_avg_loss = train(model, train_loader, loss_criterion, optimizer)
        logger.info(f'Epoch [{epoch + 1}] Train Avg Loss: {train_avg_loss:.8f}')
        if is_leader:
            tensorboard_writer.add_scalar("Train/Loss", train_avg_loss, epoch + 1)
        dist.barrier()

        val_avg_loss = test(model, validation_loader, loss_criterion)
        logger.info(f'Epoch [{epoch + 1}] Validation Avg Loss: {val_avg_loss:.8f}')
        if is_leader:
            tensorboard_writer.add_scalar("Validation/Loss", val_avg_loss, epoch + 1)
        dist.barrier()

        if val_avg_loss < best_loss:
            best_loss = val_avg_loss
            no_improvement_epochs = 0
            best_model_index = epoch
        else:
            no_improvement_epochs += 1

        scheduler.step()

        # Save the model at current epoch
        if is_leader:
            torch.save(model.module.state_dict(), os.path.join(args.model_dir, f"model_{epoch}.pth"))
        dist.barrier()

        if no_improvement_epochs >= args.no_improvement_threshold:
            logger.info(f"Early stopping at epoch {epoch + 1}")
            break

    if is_leader:
        # Keep only the best nodel
        saved_models = os.listdir(args.model_dir)
        for model_file in saved_models:
            if model_file == f'model_{best_model_index}.pth':
                os.rename(
                    os.path.join(args.model_dir, model_file),
                    os.path.join(args.model_dir, "model.pth")
                )
            else:
                os.remove(os.path.join(args.model_dir, model_file))

        tensorboard_writer.flush()
        tensorboard_writer.close()
    dist.barrier()

    dist.destroy_process_group()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--batch-size",
        type=int,
        default=128,
        help="input batch size for training (default: 128)",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=10,
        help="number of epochs to train (default: 10)",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=0.001,
        help="learning rate (default: 0.001)"
    )
    parser.add_argument(
        "--resnet-version",
        type=str,
        default='resnet18',
        help="Resnet version (default: resnet18)"
    )
    parser.add_argument(
        "--loading-workers",
        type=int,
        default=3,
        help="Number of loading workers (default: 3)"
    )
    parser.add_argument(
        "--no-improvement-threshold",
        type=int,
        default=3,
        help="Number of epoch with no improvement when training should be stopped (default: 3)"
    )

    parser.add_argument(
        "--model-dir",
        type=str,
        default=os.environ["SM_MODEL_DIR"]
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=os.environ["SM_OUTPUT_DATA_DIR"]
    )
    parser.add_argument(
        "--train-data-dir",
        type=str,
        default=os.environ["SM_CHANNEL_TRAIN"]
    )
    parser.add_argument(
        "--validation-data-dir",
        type=str,
        default=os.environ["SM_CHANNEL_VALIDATION"]
    )

    args = parser.parse_args()

    main(args)
