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


class TransformerBlockWithMoe(nn.Module):
    def __init__(self, input_dim, num_experts, expert_list, hidden_dim, output_dim, k=2):
        super().__init__()
        self.experts = nn.ModuleList([expert for expert in expert_list])
        self.gating_network = GatingFunc(input_dim, num_experts, k)
        self.layer_norm = nn.LayerNorm(output_dim)
    
    def forward(self):
        # get the weights from the gating network
        gate_weights = self.gating_network(x)
        # pass the input to all experts
        expert_output = torch.stack([expert(x) for expert in self.experts], dim=1)
        # Weight the expert outputs and sum them up
        moe_output = (gate_weights.unsqueeze(-1) * expert_output).sum(dim=1)
        gate_weights = self.gating_network(x)
        # calculate the avaerage dispatch probability for each expert
        D_i = gate_weights.mean(dim=0)
        # calcualte loss that penalizes uneven distribution
        load_balancing_loss = (D_i * torch.log(D_i + 1e-8)).sum()
        # add a residual connection and a weight normalization
        return self.layer_norm(moe_output + x), load_balancing_loss


class CNNFeatureExtractor(nn.Module):
    def __init__(self):
        super().__init__()

class CNNMoeTransformer(nn.Module):
    def __init__(self, input_dim, num_experts, hidden_dim, output_dim, k=2):
        super().__init__()
        # CNN to extract a feature vector from the image
        self.feature_extractor = CNNFeatureExtractor()
        # MoE block to process the feature vector
        self.transformer_block = TransformerBlockWithMoe(input_dim, 
                                                        num_experts, 
                                                        hidden_dim,
                                                        output_dim, 
                                                        k=2)

    def forward(self, x):
        x = self.feature_extractor(x)
        x, load_balancing_loss = self.transformer_block(x)
        return self.fc_out(x), load_balancing_loss