import torch
import torch.nn as nn
import torch.nn.functional as F

class RipeNet(nn.Module):
   
    def __init__(self, backbone, n_fruits: int, n_states: int, fruit_emb_dim: int = 128, use_probs: bool = False):
        super().__init__()
        self.backbone = backbone
        
        if not hasattr(self.backbone, "fc"):
            raise ValueError("Backbone deve expor atributo 'fc' com in_features (ex: torchvision resnet).")
        feat_dim = self.backbone.fc.in_features
        
        self.backbone.fc = nn.Identity()

        self.feat_dim = feat_dim
        self.n_fruits = n_fruits
        self.n_states = n_states
        self.fruit_emb_dim = fruit_emb_dim
        self.use_probs = use_probs

        
        self.head_fruit = nn.Linear(feat_dim, n_fruits)


        self.fruit_to_state_proj = nn.Sequential(
            nn.Linear(n_fruits, fruit_emb_dim),
            nn.ReLU(),
            nn.Linear(fruit_emb_dim, fruit_emb_dim),
            nn.ReLU()
        )

       
        self.head_state = nn.Linear(feat_dim + fruit_emb_dim, n_states)

       
        nn.init.normal_(self.head_fruit.weight, 0, 0.01)
        nn.init.constant_(self.head_fruit.bias, 0)
        nn.init.normal_(self.head_state.weight, 0, 0.01)
        nn.init.constant_(self.head_state.bias, 0)

    def forward(self, x):
       
        features = self.backbone(x) 
        fruit_logits = self.head_fruit(features) 

        if self.use_probs:
            fruit_input = F.softmax(fruit_logits, dim=1)
        else:
            fruit_input = fruit_logits

        fruit_emb = self.fruit_to_state_proj(fruit_input)

        state_input = torch.cat([features, fruit_emb], dim=1) 
        state_logits = self.head_state(state_input) 

        return fruit_logits, state_logits
