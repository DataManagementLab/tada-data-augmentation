import torch


class DownstreamModel:
    def __init__(self):
        pass

    def save_model(self, optimizer, path):
        print("=> Saving checkpoint")
        checkpoint = {
            "state_dict": self.state_dict(),
            "optimizer": optimizer.state_dict(),
        }

        torch.save(checkpoint, path)

    def load_model(self, optimizer, lr, path):
        print("=> Loading checkpoint")
        checkpoint = torch.load(path)
        self.load_state_dict(checkpoint["state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer"])

        for param_group in optimizer.param_groups:
            param_group["lr"] = lr

    def train_epoch(self, epoch_idx, dataloader, optimizer, device="cpu"):
        raise NotImplementedError

    def eval_model(self, dataloader, device="cpu", cat_label=False):
        raise NotImplementedError
