from pathlib import Path
from typing import Tuple
import torch
import torch.nn as nn
from modules import SimCLR_Loss, init_optim
import modules.projection_head as projection_head
from tqdm import tqdm
from torch.cuda.amp import GradScaler, autocast
from torch.utils.tensorboard import SummaryWriter


def load_optimizer(args, model) -> Tuple[torch.optim.Optimizer, torch.optim.lr_scheduler.CosineAnnealingLR]:
    # optimized using LARS with linear learning rate scaling
    # (i.e. LearningRate = 0.3 × BatchSize/256) and weight decay of 10−6.
    optimizer = init_optim(model=model, args=args)

    # "decay the learning rate with the cosine decay schedule without restarts"
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, args['steps'], eta_min=0, last_epoch=-1
    )

    return optimizer, scheduler


class SimCLRContrastiveLearning(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        
        self.model: nn.Module = projection_head.__dict__[self.args['model_type']](
            self.args["in_features"], 
            self.args["hidden_features"],
            self.args["out_features"],
            self.args['dropout']
        )
        self.model.to(self.args["device"])

        self.criterion = SimCLR_Loss(self.args["batch_size"], self.args["temperature"])
        self.optimizer, self.scheduler = load_optimizer(args, self.model)

        self.writer = SummaryWriter(log_dir=Path(args["exp_path"]) / Path("log_dir"))

    def save_model(self, epoch):
        out_path = Path(self.args["exp_path"]) / Path(f"checkpoint_{epoch}.tar")
        torch.save(self.model.state_dict(), out_path)
    
    def load_model(self, ckpt_path, device):
        self.model.to(device)
        ckpt = torch.load(ckpt_path, map_location=device)
        self.model.load_state_dict(ckpt)

    def train(self, train_loader, val_loader):
        # Transform loaders into iterators - we don't need them anymore!
        train_loader = iter(train_loader)
        val_loader = iter(val_loader)
        scaler = GradScaler()

        train_loss = 0.
        
        steps = tqdm(range(self.args['steps']))
        training_info = {}
        for step in steps:
            step += 1  # Useful for divions and stuff
            # Find LR
            lr = self.optimizer.param_groups[0]["lr"]

            ############
            # TRAINING #
            ############

            # Fetch data
            items = next(train_loader)
            items = [item.to(self.args["device"], non_blocking=True) for item in items]

            # Actual training
            self.model.train()
            self.optimizer.zero_grad()
            
            with autocast():
                z_i, z_j = self.model(items)
                loss = self.criterion(z_i, z_j)

            scaler.scale(loss).backward()
            scaler.step(self.optimizer)
            scaler.update()

            # Add loss to train loss
            train_loss += loss.item()

            # Print some info
            if step % self.args['train_steps'] == 0:
                # Save infos to writer
                steps.set_description(f"TrainLoss: {train_loss / self.args['train_steps']:.4f}  LR: {lr:.6f}", refresh=True)
                training_info["Train"] = train_loss / self.args['train_steps']
                train_loss = 0.

            # Update scheduler
            self.scheduler.step()

            ##############
            # VALIDATION #
            ##############

            # Evaluate if it's time
            if step % self.args["val_steps"] == 0:
                val_loss = 0.
                # Put model in eval mode
                self.model.eval()
                # Iterate over validation set
                for val_step in range(self.args['val_synset_ids'] // self.args['batch_size']):
                    items = next(val_loader)
                    items = [item.to(self.args["device"], non_blocking=True) for item in items]

                    with torch.no_grad():
                        z_i, z_j = self.model(items)
                        loss: torch.Tensor = self.criterion(z_i, z_j)
                    # Add loss to epoch
                    val_loss += loss.item()
                
                val_loss /= (val_step + 1)
                
                # Print some stats
                steps.set_description(f"ValLoss: {val_loss:.4f}", refresh=True)
                training_info["Val"] = val_loss

            # Save if there is something to save
            if step % self.args["save_steps"] == 0:
                self.save_model(step)

            # Add metrics to TB
            self.writer.add_scalars("Loss", training_info, step)
            self.writer.add_scalar("Misc/LR", lr, step)
            self.writer.flush()



