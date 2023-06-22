import os
import time
import torch
import matplotlib.pyplot as plt


class Trainer:
    def __init__(
        self,
        config={
            "model": None,
            "model_type": None,
            "criterion": None,
            "optimizer": None,
            "scheduler": None,
            "train_loader": None,
            "test_loader": None,
            "epochs": 100,
            "device": None,
        },
    ):
        self.model = config["model"]
        self.model_type = config["model_type"]
        self.config = config

    def train(self):
        config = self.config
        criterion, optimizer, scheduler = (
            config["criterion"],
            config["optimizer"],
            config["scheduler"],
        )
        train_loader, test_loader, epochs = (
            config["train_loader"],
            config["test_loader"],
            config["epochs"],
        )
        device = self.config["device"]

        total_train_data = len(train_loader.dataset)
        total_test_data = len(test_loader.dataset)

        self.model.train()

        total_time = 0
        history = {
            "train_losses": list(),
            "test_losses": list(),
            "train_accuracy": list(),
            "test_accuracy": list(),
        }

        for i in range(epochs):
            train_losses = list()
            test_losses = list()
            total_train_correct = 0
            total_test_correct = 0

            t1 = time.time()

            for x, y in train_loader:
                x, y = x.to(device), y.to(device)

                logits = self.model(x)
                loss = criterion(logits, y)
                pred = logits.argmax(dim=-1)
                total_train_correct += (pred == y).sum().item()

                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
                scheduler.step(loss)

                train_losses.append(loss.item())

            for x, y in test_loader:
                x, y = x.to(device), y.to(device)

                logits = self.model(x)
                loss = criterion(logits, y)
                pred = logits.argmax(dim=-1)
                total_test_correct += (pred == y).sum().item()

                test_losses.append(loss.item())

            train_accuracy = total_train_correct / total_train_data
            test_accuracy = total_test_correct / total_test_data

            train_loss = torch.tensor(train_losses).mean().item()
            test_loss = torch.tensor(test_losses).mean().item()

            history["train_accuracy"].append(train_accuracy)
            history["test_accuracy"].append(test_accuracy)
            history["train_losses"].append(train_loss)
            history["test_losses"].append(test_loss)

            t2 = time.time()
            elapsed = t2 - t1
            total_time += elapsed

            print(
                f"Epoch: {i + 1:2d} | train loss: {train_loss:6f} | train accuracy: {train_accuracy:6f} | test loss: {test_loss:6f} | test accuracy: {test_accuracy:6f} | time: {elapsed:6f}"
            )

        current_time = time.time()
        save_path = os.path.join("checkpoints", self.model_type, current_time)
        os.makedirs(save_path, exist_ok=True)
        torch.save(self.model.state_dict(), save_path)

        plt.figure(figsize=(10, 5))
        plt.plot(history["train_accuracy"], label="train")
        plt.plot(history["test_accuracy"], label="test")
        plt.xlabel("Epoch")
        plt.ylabel("Accuracy")
        plt.legend()
        plt.savefig(os.path.join(save_path, "accuracy.png"))

        plt.figure(figsize=(10, 5))
        plt.plot(history["train_losses"], label="train")
        plt.plot(history["test_losses"], label="test")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.legend()
        plt.savefig(os.path.join(save_path, "loss.png"))

        print(f"Total time elapsed: {total_time}")
        return history
