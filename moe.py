import torchvision
import torchvision.nn as nn


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
        sparse_gate_weights = torch.zero_like(logits).scatter(-1, topk_indicies, gate_weights)
        
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

        self.layer_norm = nn.LayerNorm(num_classes)

        self.projections = nn.ModuleList([
            nn.Linear(expert_num_classes[i], final_num_classes)
            for i in range(len(num_experts))
        ])

    def forward(self, x):
        # get features
        expert_features = []
        expert_outputs = []

        
        for i, expert in enumerate(self.experts):
            with torch.no_grad():
                features = expert.features(x)
            expert_features.append(features)

            output = expert.head(features)
            projected = self.projections[i](output)
            expert_outputs.append(projected)
        
        avg_features = torch.stack(expert_features, dim=0).mean(dim=0)
        gate_weights = self.gating(avg_features)
        
        expert_outputs = torch.stack(expert_outputs, dim=1)
        moe_output = (gate_weights.unsqueeze(-1) * expert_outputs).sum(dim=1)


        #load balancing
        D_i = gate_weights.mean(dim=0)
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
        self.accuracy = Accuracy(task="multiclass", num_classes=config.NUM_CLASSES).to(device)
        self.precision = Precision(task="multiclass", num_classes=config.NUM_CLASSES, average='macro').to(device)
        self.recall = Recall(task="multiclass", num_classes=config.NUM_CLASSES, average='macro').to(device)
        self.f1 = F1Score(task="multiclass", num_classes=config.NUM_CLASSES, average='macro').to(device)
        self.metrics_history = []

    def train(self):
        for epoch in range(self.epochs):
            self.model.train()
            running_loss = 0.0
            total_acc, total_prec, total_recall, total_f1 = 0.0, 0.0, 0.0, 0.0
            for images, labels in self.train_loader:
                images = images.to(self.device)
                labels = labels.to(self.device)
                self.optimizer.zero_grad()
                outputs, load_balancing_loss = self.model(images)
                loss = self.criterion(outputs, labels) + self.alpha * load_balancing_loss
                loss.backward()
                self.optimizer.step()
                running_loss += loss.item()
                total_acc += self.accuracy(outputs, labels)
                total_prec += self.precision(outputs, labels)
                total_recall += self.recall(outputs, labels)
                total_f1 += self.f1(outputs, labels)

            # Compute epoch metrics
            epoch_loss = running_loss / len(self.train_loader)
            epoch_acc = total_acc / len(self.train_loader)
            epoch_prec = total_prec / len(self.train_loader)
            epoch_recall = total_recall / len(self.train_loader)
            epoch_f1 = total_f1 / len(self.train_loader)

            # Validation phase
            val_loss = 0.0
            self.model.eval()
            with torch.no_grad():
                for images, labels in self.val_loader:
                    images = images.to(self.device)
                    labels = labels.to(self.device)
                    outputs, _ = self.model(images)
                    loss = self.criterion(outputs, labels)
                    val_loss += loss.item()

            val_loss /= len(self.val_loader)

            # Append metrics for this epoch
            self.metrics_history.append({
                'epoch': epoch + 1,
                'train_loss': epoch_loss,
                'val_loss': val_loss,
                'accuracy': epoch_acc.item(),
                'precision': epoch_prec.item(),
                'recall': epoch_recall.item(),
                'f1_score': epoch_f1.item()
            })

            print(f"Epoch [{epoch+1}/{self.epochs}], Train Loss: {epoch_loss:.4f}, Val Loss: {val_loss:.4f}, "
                  f"Accuracy: {epoch_acc:.4f}, Precision: {epoch_prec:.4f}, "
                  f"Recall: {epoch_recall:.4f}, F1 Score: {epoch_f1:.4f}")

        print("Training complete.")
        self.save_metrics()

    def save_metrics(self):
        with open(self.save_path, mode='w', newline='') as file:
            writer = csv.DictWriter(file, fieldnames=self.metrics_history[0].keys())
            writer.writeheader()
            writer.writerows(self.metrics_history)
        print(f"Metrics saved to {self.save_path}.")

if __name__ == "__main__":
    chest_path = ''
    derma_path = ''
    retina_path = ''

    expert1 = torch.load(chest_path)
    expert2 = torch.load(derma_path)
    expert3 = torch.load(retina_path)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = SwinMoe(
        swin_experts=[expert1, expert2, expert3],
        expert_num_classes=[],
        final_num_classes=2,
        k=2
    )

    


        


