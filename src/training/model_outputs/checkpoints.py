import torch

def save_checkpoint(model, optimizer, path):
    torch.save({'model': model.state_dict(), 'opt': optimizer.state_dict()}, path)

def load_checkpoint(model, optimizer, path, device):
    checkpoint = torch.load(path, map_location=device)
    model.load_state_dict(checkpoint['model'])
    optimizer.load_state_dict(checkpoint['opt'])