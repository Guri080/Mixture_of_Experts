import torchvision
from torchmetrics import Accuracy, Precision, Recall, F1Score
import torch
import torch.nn as nn
from torch.optim import Adam
import torch.nn.functional as F

import csv

from dataloader import MedMNIST_2D_Datasets

import config


class GatingFunc(nn.Module):
    def __init__(self, input_dims, num_experts, k=2):
        super().__init__()
        self.fc = nn.Linear(input_dims, num_experts)
        self.k = k
    
    def forward(self, x):
        # get the score of each expert
        logits = self.fc(x)
        # find the top k experts with the highest score
        topk_vals, topk_indicies = torch.topk(logits, self.k, dim=-1)
        # apply softmax to the scores of the topk experts to get weights
        gate_weights = F.softmax(topk_vals, dim=-1)
        # Create a sparse tensor with weights only for the top-k experts
        sparse_gate_weights = torch.zeros_like(logits).scatter(-1, topk_indicies, gate_weights)
        
        return sparse_gate_weights

class SwinMoe(nn.Module):
    def __init__(self, swin_experts, expert_num_classes, final_num_classes, k=2):
        super().__init__()

        self.experts = nn.ModuleList(swin_experts)
        num_experts = len(swin_experts)

        feature_dim = swin_experts[0].num_features

        # freeze all layers except the classification
        for expert in self.experts:
            for params in expert.parameters():
                params.requires_grad = False
            for params in expert.head.parameters():
                params.requires_grad = True
        
        self.gating = GatingFunc(feature_dim, num_experts)

        # self.layer_norm = nn.LayerNorm(num_classes)

        self.projections = nn.ModuleList([
            nn.Linear(expert_num_classes[i], final_num_classes)
            for i in range(num_experts)
        ])

    def forward(self, x):
        # get features
        expert_outputs = []

        with torch.no_grad():
            features = self.experts[0].features(x)
        
        features = F.adaptive_avg_pool2d(features, (1, 1)).flatten(1)
        gate_weights = self.gating(features)

        for i, expert in enumerate(self.experts):
            output = expert(x)
            projected = self.projections[i](output)
            expert_outputs.append(projected)

        expert_outputs = torch.stack(expert_outputs, dim=1)
        moe_output = (gate_weights.unsqueeze(-1) * expert_outputs).sum(dim=1)

        #load balancing
        D_i = (gate_weights > 0).float().mean(dim=0)
        load_balancing_loss = (D_i * torch.log(D_i + 1e-8)).sum()

        return moe_output, load_balancing_loss

class Trainer:
    def __init__(self, model, train_loader, val_loader, criterion, optimizer, epochs, device='cpu', save_path='main.csv', alpha=0.001):
        self.model = model.to(device)  # Ensure model is on the specified device
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.criterion = criterion
        self.optimizer = optimizer
        self.epochs = epochs
        self.device = device
        self.save_path = save_path
        self.alpha = alpha
        self.accuracy = Accuracy(task="multilabel", num_classes=config.NUM_CLASSES).to(device)
        self.precision = Precision(task="multilabel", num_classes=config.NUM_CLASSES, average='macro').to(device)
        self.recall = Recall(task="multilabel", num_classes=config.NUM_CLASSES, average='macro').to(device)
        self.f1 = F1Score(task="multilabel", num_classes=config.NUM_CLASSES, average='macro').to(device)
        self.metrics_history = []

    def learn(self):
        num_ds = len(self.train_loader)
        for epoch in range(self.epochs):
            self.model.train()
            running_loss = [0.0] * num_ds
            total_acc = [torch.tensor(0.0).to(self.device) for _ in range(num_ds)]
            total_f1 = [torch.tensor(0.0).to(self.device) for _ in range(num_ds)]
            epoch_loss, epoch_acc, epoch_f1 = [0.0] * num_ds, [0.0] * num_ds, [0.0] * num_ds
            for i, loader in enumerate(self.train_loader):
                for images, labels in loader:
                    images = images.to(self.device)
                    labels = labels.to(self.device)
                    self.optimizer.zero_grad()
                    outputs, load_balancing_loss = self.model(images)
                    loss = self.criterion(outputs, labels) + self.alpha * load_balancing_loss
                    loss.backward()
                    self.optimizer.step()
                    running_loss[i] += loss.item()
                    total_acc[i] += self.accuracy(outputs, labels)
                    # total_prec += self.precision(outputs, labels)
                    # total_recall += self.recall(outputs, labels)
                    total_f1[i] += self.f1(outputs, labels)

                # Compute epoch metrics
                epoch_loss[i] = running_loss[i] / len(loader)
                epoch_acc[i] = total_acc[i] / len(loader)
                epoch_f1[i] = total_f1[i] / len(loader)

            # Validation phase
            self.model.eval()
            with torch.no_grad():
                val_loss = [0.0] * num_ds
                for i, loader in enumerate(self.val_loader):
                    for images, labels in loader:
                        images = images.to(self.device)
                        labels = labels.to(self.device)
                        outputs, _ = self.model(images)
                        loss = self.criterion(outputs, labels)
                        val_loss[i] += loss.item()

                    val_loss[i] /= len(loader)

            # Append metrics for this epoch
            for i in range(num_ds):
                self.metrics_history.append({
                    'dataset': i,
                    'epoch': epoch + 1,
                    'train_loss': epoch_loss[i],
                    'val_loss': val_loss[i],
                    'accuracy': epoch_acc[i].item() if hasattr(epoch_acc[i], 'item') else epoch_acc[i],
                    'f1_score': epoch_f1[i].item() if hasattr(epoch_f1[i], 'item') else epoch_f1[i]
                })
                print(f"Dataset {i}, Epoch [{epoch+1}/{self.epochs}], Train Loss: {epoch_loss[i]:.4f}, Val Loss: {val_loss[i]:.4f}, "
                    f"Accuracy: {epoch_acc[i]:.4f}, F1 Score: {epoch_f1[i]:.4f}")

        print("Training complete.")
        self.save_metrics()

    def save_metrics(self):
        with open(self.save_path, mode='w', newline='') as file:
            writer = csv.DictWriter(file, fieldnames=self.metrics_history[0].keys())
            writer.writeheader()
            writer.writerows(self.metrics_history)
        print(f"Metrics saved to {self.save_path}.")

if __name__ == "__main__":
    chest_path = '/home/gssodhi/comp_vis/experiments/MedMNIST2D/myModels/chestmnist/best_model_chestmnist.pth'
    retina_path = '/home/gssodhi/comp_vis/experiments/MedMNIST2D/myModels/organsmnist/best_model_organsmnist.pth'
    organs_path = '/home/gssodhi/comp_vis/experiments/MedMNIST2D/myModels/organsmnist/best_model_organsmnist.pth'

    expert1 = torch.load(chest_path)
    expert2 = torch.load(retina_path)
    expert3 = torch.load(organs_path)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = SwinMoe(
        swin_experts=[expert1, expert2, expert3],
        expert_num_classes=[config.CHEST_CLASSES, 
                            config.RETINA_CLASSES, 
                            config.ORGANS_CLASSES],
        final_num_classes= config.NUM_CLASSES,
        k=2
    )

    chest_train_loader = MedMNIST_2D_Datasets('chestmnist', split='train')
    retina_train_loader = MedMNIST_2D_Datasets('retinamnist', split='train')
    organs_train_loader = MedMNIST_2D_Datasets('organsmnist', split='train')

    chest_val_loader = MedMNIST_2D_Datasets('chestmnist', split='val')
    retina_val_loader = MedMNIST_2D_Datasets('retinamnist', split='val')
    organs_val_loader = MedMNIST_2D_Datasets('organsmnist', split='val')

    criterion = torch.nn.BCEWithLogitsLoss()

    optimizer = Adam(model.parameters(), lr=config.lr)
    # (self, model, train_loader, val_loader, criterion, optimizer, epochs, device='cpu', save_path='main.csv', alpha=0.001):
    machine = Trainer(model=model,
                        train_loader=[chest_train_loader, retina_train_loader, organs_train_loader],
                        val_loader=[chest_val_loader, retina_val_loader, organs_val_loader],
                        criterion=criterion,
                        optimizer=optimizer,
                        epochs=config.epochs,
                        device=device,
                        save_path=config.save_path
    )

    machine.learn() 