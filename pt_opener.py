import torch

loaded_data = torch.load("file.pt")
model = YourModelClass()
model.load_state_dict(loaded_data)
model.eval()

