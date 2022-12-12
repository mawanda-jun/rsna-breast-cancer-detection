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
    # Register cosine annealing just to set the base_lr right
    cosine_annealing = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, args['epochs']*args['train_steps'], eta_min=0, last_epoch=-1
    )

    # Add warmup 
    warmup = torch.optim.lr_scheduler.CyclicLR(
        optimizer=optimizer,
        base_lr=1e-8,
        max_lr=args['lr'],
        step_size_up=args['warmup_steps'],
        cycle_momentum=False
    )

    schedulers['warmup'] = warmup
    # Add CyclicLR to escape local minima
    cyclic_lr = torch.optim.lr_scheduler.CyclicLR(
        optimizer=optimizer,
        base_lr=optimizer.param_groups[0]['lr'],
        max_lr=optimizer.param_groups[0]['lr'] / 10,
        step_size_up = args['train_steps'], 
        step_size_down = args['train_steps'],
        cycle_momentum=False,
        last_epoch=args['warmup_steps'] 
    )
    # cyclic_annealing = torch.optim.lr_scheduler.ChainedScheduler([cyclicLR, cosine_annealing])
    schedulers['cyclic_lr'] = cyclic_lr

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

            for train_step in range(self.args['train_steps']):

                # # warmup LR
                if train_step < self.args['warmup_steps'] and epoch == 0:
                    self.schedulers['warmup'].step()

                # Fetch data
                train_batch = next(train_iter)
                train_images = train_batch[0].to(self.args['device'])
                # Normalize images
                train_images = normalize(
                    train_images, 
                    mean=self.mean,
                    std=self.std
                )
                train_classes = train_batch[1].to(self.args['device'])

                # Actual training
                self.optimizer.zero_grad()
                
                with autocast():
                    train_pred_logits = self.model(train_images)
                    train_pred_logits, train_sim_loss = train_pred_logits
                    train_cat_loss = self.criterion(train_pred_logits, train_classes)
                    train_loss = train_cat_loss + train_sim_loss

                scaler.scale(train_loss).backward()
                scaler.step(self.optimizer)
                scaler.update()

                # Print some info
                lr = self.optimizer.param_groups[0]["lr"]
                train_progressbar.set_description(f"TrainLoss: {train_loss.item():.4f}  LR: {lr:.6f}", refresh=True)
                train_progressbar.update()

                # Add metrics to TB
                # Set the number of patients that the network has seen.
                # This is useful when comparing multiple networks.
                # This value will be used all over the method
                num_images += self.args['batch_size']

                if (train_step + 1) % 10 == 0:
                    self.writer.add_scalars("Loss", {
                        "train_total": train_loss.item(),
                        "train_cat":  train_cat_loss.item(),
                        "train_sim": train_sim_loss.item()
                    }, num_images)

                # Update batch scheduler
                self.writer.add_scalar("Misc/LR", lr, num_images)
                if train_step >= self.args['warmup_steps'] or epoch > 0:
                   self.schedulers['cyclic_lr'].step()
                # self.schedulers['cosine_annealing'].step()

            # Del resources from train so that we empty space
            train_progressbar.close()
            del train_batch
            del train_cat_loss
            del train_sim_loss
            del train_loss
            del train_images
            del train_classes

            # Update scheduler
            # self.schedulers['cosine_annealing'].step()

            ##############
            # VALIDATION #
            ##############
            self.model.eval()
            total_val_loss = 0.
            total_val_cat_loss = 0.
            total_val_sim_loss = 0.
            val_targets = []
            val_predictions = []
            for val_step, val_batch in enumerate(tqdm(val_loader, desc='Validating...')):
                # Fetch data
                val_images = val_batch[0].to(self.args['device'])
                # Normalize images
                val_images = normalize(
                    val_images, 
                    mean=self.mean,
                    std=self.std
                )
                val_classes = val_batch[1].to(self.args['device'])

                with torch.no_grad():
                    val_pred_logits = self.model(val_images)
                    val_pred_logits, val_sim_loss = val_pred_logits
                    val_cat_loss = self.criterion(val_pred_logits, val_classes)
                    val_loss = val_cat_loss + val_sim_loss

                    # Remember predictions and targets
                    val_targets += val_classes.squeeze(-1).tolist()
                    val_predictions += torch.sigmoid(val_pred_logits).cpu().squeeze(-1).tolist()
                
                # Accumulate val loss
                total_val_loss += val_loss.item()
                total_val_cat_loss += val_cat_loss.item()
                total_val_sim_loss += val_sim_loss.item()

            # Add metrics
            val_targets = np.array(val_targets)
            val_predictions = np.array(val_predictions)
            
            # Class 1
            val_beta_1, val_pf1_score_1 = best_pfbeta(val_targets, val_predictions)
            val_precision_1, val_recall_1 = precision_recall(val_targets, val_predictions > val_beta_1)
            
            # Class 0
            val_labels_0 = np.logical_not(val_targets)
            val_predictions_0 = 1 - val_predictions
            val_beta_0, val_pf1_score_0 = best_pfbeta(val_labels_0, val_predictions_0)
            val_precision_0, val_recall_0 = precision_recall(val_labels_0, val_predictions_0 > val_beta_0)

            self.writer.add_scalars("ValMetrics/beta", {'beta_0': val_beta_0, 'beta_1': val_beta_1}, num_images)
            self.writer.add_scalars("ValMetrics/pF1", {'pf1_score_0': val_pf1_score_0, 'pf1_score_1': val_pf1_score_1}, num_images)
            self.writer.add_scalars("ValMetrics/precision", {'precision_0': val_precision_0, 'precision_1': val_precision_1}, num_images)
            self.writer.add_scalars("ValMetrics/recall", {'recall_0': val_recall_0, 'recall_1': val_recall_1}, num_images)

            self.writer.add_scalars("Loss", {
                "val_total": total_val_loss / (val_step + 1),
                "val_cat": total_val_cat_loss / (val_step + 1),
                "val_sim": total_val_sim_loss / (val_step + 1)
            }, num_images)

    
            # Del resources from train so that we empty space
            del val_batch
            del val_cat_loss
            del val_sim_loss
            del val_loss
            del val_images
            del val_classes
            del val_targets
            del val_predictions


            ########
            # TEST #
            ########
            if (epoch + 1) % self.args['test_epochs'] == 0:
                # For validation
                test_predictions = []
                test_targets = []
                self.model.eval()
                total_test_loss = 0.
                total_test_cat_loss = 0.
                total_test_sim_loss = 0.
                for test_batch in tqdm(test_loader, desc='Testing...'):
                    # Put model in eval mode
                    # Fetch data
                    test_images = test_batch[0].to(self.args['device'])
                    # Normalize images
                    test_images = normalize(
                        test_images, 
                        mean=self.mean,
                        std=self.std
                    )
                    test_classes = test_batch[1].to(self.args['device'])

                    with torch.no_grad():
                        test_pred_logits = self.model(test_images)
                        test_pred_logits, test_sim_loss = test_pred_logits
                        test_cat_loss = self.criterion(test_pred_logits, test_classes)
                        test_loss = test_cat_loss + test_sim_loss
                        
                        # Remember predictions and targets
                        test_targets += test_classes.cpu().squeeze(-1).tolist()
                        test_predictions += torch.sigmoid(test_pred_logits).cpu().squeeze(-1).tolist()

                    # Accumulate test loss
                    total_test_loss += test_loss.item()
                    total_test_cat_loss += test_cat_loss.item()
                    total_test_sim_loss += test_sim_loss.item()
                self.writer.add_scalars("Loss", {
                    "test_total": total_test_loss / len(test_loader),
                    "test_cat": total_test_cat_loss / len(test_loader),
                    "test_sim": total_test_sim_loss / len(test_loader)
                }, num_images)

                # Add metrics
                test_targets = np.array(test_targets)
                test_predictions = np.array(test_predictions)
                
                # Class 1
                test_beta_1, test_pf1_score_1 = best_pfbeta(test_targets, test_predictions)
                test_precision_1, test_recall_1 = precision_recall(test_targets, test_predictions > test_beta_1)
                
                # Class 0
                test_labels_0 = np.logical_not(test_targets)
                test_predictions_0 = 1 - test_predictions
                test_beta_0, test_pf1_score_0 = best_pfbeta(test_labels_0, test_predictions_0)
                test_precision_0, test_recall_0 = precision_recall(test_labels_0, test_predictions_0 > test_beta_0)

                self.writer.add_scalars("Metrics/beta", {'beta_0': test_beta_0, 'beta_1': test_beta_1}, num_images)
                self.writer.add_scalars("Metrics/pF1", {'pf1_score_0': test_pf1_score_0, 'pf1_score_1': test_pf1_score_1}, num_images)
                self.writer.add_scalars("Metrics/precision", {'precision_0': test_precision_0, 'precision_1': test_precision_1}, num_images)
                self.writer.add_scalars("Metrics/recall", {'recall_0': test_recall_0, 'recall_1': test_recall_1}, num_images)

                # Find pF1 score at given betas
                pf1_scores = {}
                precisions = {}
                recalls = {}
                for beta in np.linspace(0, 1, 5):
                    pf1_scores[str(beta)] = pfbeta(test_targets, test_predictions > beta, beta)
                    precision, recall = precision_recall(test_targets, test_predictions > beta)
                    precisions[str(beta)] = precision
                    recalls[str(beta)] = recall
                self.writer.add_scalars(f"AtBeta/pF1", pf1_scores, num_images)
                self.writer.add_scalars(f"AtBeta/precision", precisions, num_images)
                self.writer.add_scalars(f"AtBeta/recall", recalls, num_images)
                
                print(f"Found best F1 of {test_pf1_score_1:.4f} at beta {test_beta_1:.2f}.")

                # Del resources from train so that we empty space
                del test_batch
                del test_cat_loss
                del test_sim_loss
                del test_loss
                del test_images
                del test_classes
                del test_targets
                del test_predictions

            # Save if there is something to save
            if (epoch+1) % self.args["save_epochs"] == 0:
                self.save_model(num_images)

            self.writer.flush()



