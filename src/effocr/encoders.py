from transformers import ViTModel
import torch
import time


class VitEncoder(torch.nn.Module):

    def __init__(self, hub_url='facebook/dino-vitb16'):
        super().__init__()
        net = ViTModel.from_pretrained(hub_url)
        net.to("cuda" if torch.cuda.is_available() else "cpu")
        self.net = net

    def forward(self, x):
        start_time = time.time()
        x = self.net(x)
        x = x.last_hidden_state[:,0,:]
        return x

    @classmethod
    def load(cls, checkpoint):
        ptnet = cls()
        ptnet.load_state_dict(torch.load(checkpoint, map_location=torch.device('cpu')))
        ptnet.to("cuda" if torch.cuda.is_available() else "cpu")
        return ptnet
