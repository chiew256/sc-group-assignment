import torch
from torch import nn, optim
from torch.utils.data import DataLoader

from trainer import Trainer
from dataset import HandwritingDataset, PrepForDataLoader

from models import get_model

dataset = HandwritingDataset()
train_dataset, test_dataset = dataset.stratified_split()
train_dataset, test_dataset = PrepForDataLoader(train_dataset), PrepForDataLoader(
    test_dataset
)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=True)

device = torch.device("cuda") if torch.cuda.is_available() else "cpu"

lr = 3e-4
epochs = 1
n_evals = 1

model_type = "vit"
model_config = {
    "num_classes": 4,
    "num_channels": 1,
    "embed_dim": 256,
    "hidden_dim": 512,
    "num_heads": 8,
    "num_layers": 6,
    "patch_size": 4,
    "num_patches": 64,
    "dropout": 0.2,
}
model = get_model(model_type, **model_config)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=0.0001)
# scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.1, patience=5)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10, eta_min=1e-4)

trainer = Trainer(
    {
        "model": model,
        "model_type": model_type,
        "criterion": criterion,
        "optimizer": optimizer,
        "scheduler": scheduler,
        "train_loader": train_loader,
        "test_loader": test_loader,
        "n_evals": n_evals,
        "epochs": 1,
        "device": device,
    }
)

trainer.train()
