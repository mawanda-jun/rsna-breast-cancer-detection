from pathlib import Path
from typing import Tuple
import torch
import torch.nn as nn
from torch.optim import AdamW, Adam
import network
from tqdm import tqdm
from torchvision.transforms.functional_tensor import normalize
from metrics import *
from torch.cuda.amp import GradScaler, autocast
from timeit import default_timer as timer
import numpy as np
from torch.utils.tensorboard import SummaryWriter



def load_optimizer(args, model: nn.Module) -> Tuple[torch.optim.Optimizer, torch.optim.lr_scheduler.CosineAnnealingLR]:
    # optimized using LARS with linear learning rate scaling
    # (i.e. LearningRate = 0.3 × BatchSize/256) and weight decay of 10−6.
    optimizer = Adam(
        params=model.parameters(),
        lr=args['lr'],
        weight_decay=args['weight_decay']
    )
    schedulers = {}
    # "decay the learning rate with the cosine decay schedule without restarts"
    cosine_annealing = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, args['epochs']*args['train_steps'], eta_min=0, last_epoch=-1
    )
    torch.optim.lr_scheduler.CosineAnnealingWarmRestarts()
    # Add CyclicLR to escape local minima
    cyclicLR = torch.optim.lr_scheduler.CyclicLR(
        optimizer=optimizer,
        base_lr=optimizer.param_groups[0]['lr'],
        max_lr=optimizer.param_groups[0]['lr'] / 10,
        step_size_up = args['train_steps'] // 2, 
        step_size_down = args['train_steps'] // 2,
        cycle_momentum=False,
        last_epoch=-1 
    )
    cyclic_annealing = torch.optim.lr_scheduler.ChainedScheduler([cyclicLR, cosine_annealing])
    schedulers['cyclic_annealing'] = cyclic_annealing

    # Add warmup 
    warmup = torch.optim.lr_scheduler.CyclicLR(
        optimizer=optimizer,
        base_lr=1e-8,
        max_lr=args['lr'],
        step_size_up=args['warmup_steps'],
        cycle_momentum=False
    )
    schedulers['warmup'] = warmup

    return optimizer, schedulers


class RSNABCE(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        
        network_name = list(self.args['network'][0].keys())[0]
        network_params = list(self.args['network'][0].values())[0]
        self.model: nn.Module = network.__dict__[network_name](**network_params)
        
        self.model.to(self.args["device"])
        self.mean = (torch.tensor([0.485, 0.456, 0.406])*(2**self.args['color_space'] - 1)).to(self.args['device'])
        self.std = (torch.tensor([0.229, 0.224, 0.225])*(2**self.args['color_space'] - 1)).to(self.args['device'])

        self.criterion = nn.BCEWithLogitsLoss()
        self.optimizer, self.schedulers = load_optimizer(args, self.model)

        self.writer = SummaryWriter(log_dir=Path(args["exp_path"]) / Path("log_dir"))

    def save_model(self, epoch):
        out_path = Path(self.args["exp_path"]) / Path(f"checkpoint_{epoch}.tar")
        torch.save(self.model.state_dict(), out_path)
    
    def load_model(self, ckpt_path, device):
        self.model.to(device)
        ckpt = torch.load(ckpt_path, map_location=device)
        self.model.load_state_dict(ckpt)

    def train(self, train_loader, val_loader, test_loader):
        
        # Make train_loader and val_loader as iterators, so it's possible to iterate over them indefinitely
        train_iter = iter(train_loader)
        scaler = GradScaler()

        num_images = 0
        start_time = timer()
        for epoch in range(self.args['epochs']):
            elapsed_time = timer() - start_time
            remaining_time = (self.args['epochs'] - (epoch + 1)) * elapsed_time / (epoch+1)
            time_string = f"{round(elapsed_time // 60):n}:{round(elapsed_time % 60):n} < {round(remaining_time // 60):n}:{round(elapsed_time % 60):n}"
            print(f"############ EPOCH {epoch + 1}/{self.args['epochs']}\tTime:{time_string} ############")
            # print(f"############ EPOCH {epoch + 1}/{self.args['epochs']} ############")

            ############
            # TRAINING #
            ############
            self.model.train()
            train_progressbar = tqdm(range(self.args['train_steps']))

            total_train_loss = 0.
            for train_step in range(self.args['train_steps']):

                # warmup LR
                if train_step < self.args['warmup_steps'] and epoch == 0:
                    self.schedulers['warmup'].step()

                # Fetch data
                batch = next(train_iter)
                train_images = batch[0].to(self.args['device'])
                # Normalize images
                train_images = normalize(
                    train_images, 
                    mean=self.mean,
                    std=self.std
                )
                train_classes = batch[1].to(self.args['device'])

                # Actual training
                self.optimizer.zero_grad()
                
                with autocast():
                    pred_logits = self.model(train_images)
                    loss = self.criterion(pred_logits, train_classes)

                scaler.scale(loss).backward()
                scaler.step(self.optimizer)
                scaler.update()

                # Print some info
                lr = self.optimizer.param_groups[0]["lr"]
                train_progressbar.set_description(f"TrainLoss: {loss.item():.4f}  LR: {lr:.6f}", refresh=True)
                train_progressbar.update()

                # Add metrics to TB
                # Set the number of patients that the network has seen.
                # This is useful when comparing multiple networks.
                # This value will be used all over the method
                num_images += self.args['batch_size']
                if (train_step+1) % 10 == 0:
                    self.writer.add_scalars("Loss", {"Train": total_train_loss / (train_step + 1)}, num_images)
                else:
                    total_train_loss += loss.item()

                # Update batch scheduler
                self.writer.add_scalar("Misc/LR", lr, num_images)
                if train_step >= self.args['warmup_steps'] or epoch > 0:
                    self.schedulers['cyclic_annealing'].step()

            # Update scheduler
            # self.schedulers['cosine_annealing'].step()

            ##############
            # VALIDATION #
            ##############
            self.model.eval()
            val_loader = tqdm(val_loader)
            total_val_loss = 0.
            for batch in val_loader:
                # Fetch data
                val_images = batch[0].to(self.args['device'])
                # Normalize images
                val_images = normalize(
                    val_images, 
                    mean=self.mean,
                    std=self.std
                )
                val_classes = batch[1].to(self.args['device'])

                with torch.no_grad():
                    pred_logits = self.model(val_images)
                    val_loss = self.criterion(pred_logits, val_classes)
                
                # Print some stats
                val_loader.set_description(f"ValLoss: {val_loss.item():.4f}", refresh=True)
                val_loader.update()
                # Accumulate val loss
                total_val_loss += val_loss.item()
            self.writer.add_scalars("Loss", {"Val": total_val_loss / len(val_loader)}, num_images)

            ########
            # TEST #
            ########
            if (epoch + 1) % self.args['test_epochs'] == 0:
                # For validation
                predictions = []
                targets = []
                self.model.eval()
                test_loader = tqdm(test_loader)
                total_test_loss = 0.
                for batch in test_loader:
                    # Put model in eval mode
                    # Fetch data
                    test_images = batch[0].to(self.args['device'])
                    # Normalize images
                    test_images = normalize(
                        test_images, 
                        mean=self.mean,
                        std=self.std
                    )
                    test_classes = batch[1].to(self.args['device'])

                    with torch.no_grad():
                        pred_logits = self.model(test_images)
                        test_loss = self.criterion(pred_logits, test_classes)
                    
                    # Print some stats
                    test_loader.set_description(f"TestLoss: {test_loss.item():.4f}", refresh=True)

                    # Remember predictions and targets
                    targets += test_classes.squeeze().tolist()
                    predictions += torch.sigmoid(pred_logits).cpu().squeeze().tolist()

                    # Accumulate test loss
                    total_test_loss += test_loss.item()
                self.writer.add_scalars("Loss", {"Test": total_test_loss / len(test_loader)}, num_images)

                # Add metrics
                labels = np.asarray(targets)
                predictions = np.asarray(predictions)
                
                # Class 1
                beta_1, pf1_score_1 = best_pfbeta(labels, predictions)
                precision_1, recall_1 = precision_recall(labels, predictions)

                # Thresholded/Class 1
                predictions_T = np.copy(predictions)
                predictions_T[predictions_T >= 0.5] = 1
                predictions_T[predictions_T < 0.5] = 0 
                beta, pf1_score = best_pfbeta(labels, predictions_T)
                precision, recall = precision_recall(labels, predictions_T)
                self.writer.add_scalar("ThresholdedAt0.5/best_beta", beta, num_images)
                self.writer.add_scalar("ThresholdedAt0.5/pF1Atbest_beta", pf1_score, num_images)
                self.writer.add_scalar("ThresholdedAt0.5/precision", precision, num_images)
                self.writer.add_scalar("ThresholdedAt0.5/recall", recall, num_images)
                
                # Find pF1 score at given betas
                pf1_scores = {}
                for beta in np.linspace(0, 1, 5):
                    pf1_scores[str(beta)] = pfbeta(labels, predictions, beta)
                self.writer.add_scalars(f"pF1", pf1_scores, num_images)
                
                # For class 0
                labels_0 = np.logical_not(labels)
                predictions_0 = 1 - predictions
                beta_0, pf1_score_0 = best_pfbeta(labels_0, predictions_0)
                precision_0, recall_0 = precision_recall(labels_0, predictions_0)

                self.writer.add_scalars("PerClass/beta", {'beta_0': beta_0, 'beta_1': beta_1}, num_images)
                self.writer.add_scalars("PerClass/pF1", {'pf1_score_0': pf1_score_0, 'pf1_score_1': pf1_score_1}, num_images)
                self.writer.add_scalars("PerClass/precision", {'precision_0': precision_0, 'precision_1': precision_1}, num_images)
                self.writer.add_scalars("PerClass/recall", {'recall_0': recall_0, 'recall_1': recall_1}, num_images)

                print(f"Found best F1 of {pf1_score:.4f} at beta {beta:.2f}.")

            # Save if there is something to save
            if epoch % self.args["save_epochs"] == 0:
                self.save_model(num_images)

            self.writer.flush()



