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
epochs = 100

model_type = "resnet50"
model = get_model(model_type, num_classes=4, channels=1)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=0.0001)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.1, patience=5)

trainer = Trainer(
    {
        "model": model,
        "model_type": model_type,
        "criterion": criterion,
        "optimizer": optimizer,
        "scheduler": scheduler,
        "train_loader": train_loader,
        "test_loader": test_loader,
        "epochs": epochs,
        "device": device,
    }
)

trainer.train()
