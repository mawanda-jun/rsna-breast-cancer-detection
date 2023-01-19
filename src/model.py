from pathlib import Path
from typing import Tuple
import torch
import torch.nn as nn
from torch.optim import AdamW, Adam
from custom_schedulers import CosineAnnealingWarmupRestarts
import network
from tqdm import tqdm
from torchvision.transforms.functional_tensor import normalize
from metrics import *
from torch.cuda.amp import GradScaler, autocast
from timeit import default_timer as timer
import numpy as np
from torch.utils.tensorboard import SummaryWriter
from torchmetrics.classification import BinaryAUROC
from regularization import L2Regularization


class RSNABCE(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args

        self.mean = (torch.tensor([0.485, 0.456, 0.406])*(
            2**self.args['color_space'] - 1)).to(self.args['device']).to(torch.float16)
        self.std = (torch.tensor([0.229, 0.224, 0.225])*(
            2**self.args['color_space'] - 1)).to(self.args['device']).to(torch.float16)

        self.criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(
            self.args['pos_weight']) if 'pos_weight' in self.args.keys() else None)
        self.metric = BinaryAUROC()
        self.regularization = L2Regularization(
            self.args['l2_lambda'], self.args['l2_layers'])

        self.writer = SummaryWriter(log_dir=Path(
            args["exp_path"]) / Path("log_dir"))

        self.__init_model()
        self.__init_optim()
        self.__init_schedulers()

    def __init_model(self):
        network_name = list(self.args['network'][0].keys())[0]
        network_params = list(self.args['network'][0].values())[0]
        self.model: nn.Module = network.__dict__[
            network_name](**network_params)

        self.model.to(self.args["device"])

    def __init_optim(self):
        # optimized using LARS with linear learning rate scaling
        # (i.e. LearningRate = 0.3 × BatchSize/256) and weight decay of 10−6.
        self.optimizer = Adam(
            params=self.model.parameters(),
            lr=self.args['lr'],
            weight_decay=self.args['weight_decay']
        )

    def __init_schedulers(self):
        self.schedulers = {}
        # # "decay the learning rate with the cosine decay schedule without restarts"
        # # Register cosine annealing just to set the base_lr right
        # cosine_annealing = torch.optim.lr_scheduler.CosineAnnealingLR(
        #     optimizer, args['epochs'], eta_min=0, last_epoch=-1
        # )
        # schedulers['cosine_annealing'] = cosine_annealing

        # # Add warmup
        # warmup = torch.optim.lr_scheduler.CyclicLR(
        #     optimizer=optimizer,
        #     base_lr=1e-8,
        #     max_lr=args['lr'],
        #     step_size_up=args['warmup_steps'] * args['gradient_acc_iters'],
        #     cycle_momentum=False
        # )

        # schedulers['warmup'] = warmup
        # # Add CyclicLR to escape local minima
        # cyclic_lr = torch.optim.lr_scheduler.CyclicLR(
        #     optimizer=optimizer,
        #     base_lr=optimizer.param_groups[0]['lr'],
        #     max_lr=optimizer.param_groups[0]['lr'] / 10,
        #     step_size_up = args['train_steps']*args['gradient_acc_iters'],
        #     step_size_down = args['train_steps']*args['gradient_acc_iters'],
        #     cycle_momentum=False,
        #     last_epoch=args['warmup_steps']
        # )
        # # cyclic_annealing = torch.optim.lr_scheduler.ChainedScheduler([cyclicLR, cosine_annealing])
        # schedulers['cyclic_lr'] = cyclic_lr

        train_steps = self.args['train_steps'] * \
            self.args['gradient_acc_iters']
        scheduler = CosineAnnealingWarmupRestarts(
            optimizer=self.optimizer,
            first_cycle_steps=train_steps,
            cycle_mult=1.,
            max_lr=self.args['lr'],
            min_lr=self.args['lr'] / 500,
            warmup_steps=train_steps // 5,
            gamma=1.,
            last_epoch=-1
        )
        self.schedulers['cosine_annealing_warmup_restarts'] = scheduler

    def save_model(self, epoch):
        out_path = Path(self.args["exp_path"]) / \
            Path(f"checkpoint_{epoch}.tar")
        torch.save(self.model.state_dict(), out_path)

    def load_model(self, ckpt_path, device):
        self.model.to(device)
        ckpt = torch.load(ckpt_path, map_location=device)
        self.model.load_state_dict(ckpt)

    def _register_metrics(self, pred, target, num_images, global_key):
        # Add metrics
        targets = np.array(target)
        predictions = np.array(pred)

        # Class 1
        beta_1, pf1_score_1 = best_pfbeta(targets, predictions)
        precision_1, recall_1 = precision_recall(targets, predictions > beta_1)

        # Class 0
        labels_0 = np.logical_not(targets)
        predictions_0 = 1 - predictions
        beta_0, pf1_score_0 = best_pfbeta(labels_0, predictions_0)
        precision_0, recall_0 = precision_recall(
            labels_0, predictions_0 > beta_0)

        self.writer.add_scalars(
            f"{global_key}/beta", {'beta_0': beta_0, 'beta_1': beta_1}, num_images)
        self.writer.add_scalars(
            f"{global_key}/pF1", {'pf1_score_0': pf1_score_0, 'pf1_score_1': pf1_score_1}, num_images)
        self.writer.add_scalars(
            f"{global_key}/precision", {'precision_0': precision_0, 'precision_1': precision_1}, num_images)
        self.writer.add_scalars(
            f"{global_key}/recall", {'recall_0': recall_0, 'recall_1': recall_1}, num_images)

        # Register AUC
        auc = self.metric(torch.tensor(predictions), torch.tensor(targets))
        self.writer.add_scalar(f"{global_key}/AUC", auc, num_images)

        return pf1_score_1, beta_1

    def _train_batch(self, train_iter, num_images):
        total_train_loss = 0.
        total_train_cat_loss = 0.
        total_train_sim_loss = 0.

        for _ in range(self.args['gradient_acc_iters']):
            train_batch = next(train_iter)
            train_images = train_batch[0].to(self.args['device'])

            # Stack multiple images to imitate the third channel dimension
            train_images = torch.cat(
                [train_images, train_images, train_images], -3)
            # Normalize images
            train_images = normalize(
                train_images,
                mean=self.mean,
                std=self.std
            )

            train_classes = train_batch[1].to(self.args['device'])

            # Actual training
            with autocast():
                train_pred_logits, train_sim_loss = self.model(train_images)
                train_cat_loss = self.criterion(
                    train_pred_logits, train_classes)
                if self.args['projection_sim']:
                    train_loss = train_cat_loss + train_sim_loss
                else:
                    train_loss = train_cat_loss

                # Accumulate gradient
                train_loss = train_loss / self.args['gradient_acc_iters']
                reg = self.regularization(self.model) / self.args['gradient_acc_iters']

            # Add L2 regularization
            self.scaler.scale(train_loss + reg).backward()

            # Accumulate train loss
            total_train_loss += train_loss.item()
            total_train_cat_loss += train_cat_loss.item() / self.args['gradient_acc_iters']
            total_train_sim_loss += train_sim_loss.item() / self.args['gradient_acc_iters']

            # Set the number of patients that the network has seen.
            # This is useful when comparing multiple networks.
            # This value will be used all over the method
            num_images += self.args['batch_size']

            # Update batch scheduler
            self.schedulers['cosine_annealing_warmup_restarts'].step()

            # Print some info
            lr = self.optimizer.param_groups[0]["lr"]

            # Add lr to tensorboard
            self.writer.add_scalar("Misc/LR", lr, num_images)

        # Update optimizer
        self.scaler.step(self.optimizer)
        self.scaler.update()
        # Reset optimizer
        self.optimizer.zero_grad()

        scalars = {
            "train_total": total_train_loss,
            "train_cat":  total_train_cat_loss,
            "train_sim": total_train_sim_loss
        }
        self.writer.add_scalars("Loss", scalars, num_images)

        return total_train_loss, num_images

    def _eval_model(self, loader, num_images, mode):
        self.model.eval()
        total_loss = 0.
        total_cat_loss = 0.
        total_sim_loss = 0.
        targets = []
        predictions = []
        for batch in tqdm(loader, desc=f'{mode}...'):
            # Fetch data
            images = batch[0].to(self.args['device'])

            images = torch.cat([images, images, images], -3)
            # Normalize images
            images = normalize(
                images,
                mean=self.mean,
                std=self.std
            )
            classes = batch[1].to(self.args['device'])

            with torch.no_grad():
                pred_logits = self.model(images)
                pred_logits, sim_loss = pred_logits
                cat_loss = self.criterion(pred_logits, classes)
                if self.args['projection_sim']:
                    loss = cat_loss + sim_loss
                else:
                    loss = cat_loss

                # Remember predictions and targets
                targets += classes.squeeze(-1).tolist()
                predictions += torch.sigmoid(
                    pred_logits).cpu().squeeze(-1).tolist()

            # Accumulate val loss
            total_loss += loss.item()
            total_cat_loss += cat_loss.item()
            total_sim_loss += sim_loss.item()

        scalars = {
            f"{mode}_total": total_loss / len(loader)}
        if sim_loss is not None:
            scalars.update({
                f"{mode}_cat": total_cat_loss / len(loader),
                f"{mode}_sim": total_sim_loss / len(loader)
            })
        self.writer.add_scalars("Loss", scalars, num_images)

        return predictions, targets

    def train(self, train_loader, val_loader, test_loader):

        # Make train_loader and val_loader as iterators, so it's possible to iterate over them indefinitely
        self.scaler = GradScaler()

        num_images = 0
        start_time = timer()
        for epoch in range(self.args['epochs']):
            elapsed_time = timer() - start_time
            remaining_time = (self.args['epochs'] -
                              (epoch + 1)) * elapsed_time / (epoch+1)
            time_string = f"{round(elapsed_time // 60):n}:{round(elapsed_time % 60):n} < {round(remaining_time // 60):n}:{round(elapsed_time % 60):n}"
            print(
                f"############ EPOCH {epoch + 1}/{self.args['epochs']}\tTime:{time_string} ############")
            # print(f"############ EPOCH {epoch + 1}/{self.args['epochs']} ############")

            ############
            # TRAINING #
            ############
            train_iter = iter(train_loader)
            self.model.train()
            train_steps = tqdm(range(self.args['train_steps']))
            for _ in train_steps:
                total_train_loss, num_images = self._train_batch(train_iter, num_images)
                lr = self.optimizer.param_groups[0]["lr"]
                train_steps.set_description(f"TrainLoss: {total_train_loss:.4f}  LR: {lr:.6f}", refresh=True)

            train_steps.close()

            ##############
            # VALIDATION #
            ##############
            val_predictions, val_targets = self._eval_model(val_loader, num_images, mode='val')
            self._register_metrics(val_predictions, val_targets, num_images, global_key="ValMetrics")

            ########
            # TEST #
            ########
            if (epoch + 1) % self.args['test_epochs'] == 0:
                test_predictions, test_targets = self._eval_model(test_loader, num_images, mode='test')
                test_pf1_score_1, test_beta_1 = self._register_metrics(test_predictions, test_targets, num_images, global_key='Metrics')

                print(f"Found best F1 of {test_pf1_score_1:.4f} at beta {test_beta_1:.2f}.")

            # Save if there is something to save
            if (epoch+1) % self.args["save_epochs"] == 0:
                self.save_model(num_images)

            self.writer.flush()

    def eval(self, test_loader, num_images, global_key="DDSM_Metric"):
        test_predictions, test_targets = self._eval_model(
            test_loader, num_images, mode='test')
        test_pf1_score_1, test_beta_1 = self._register_metrics(
            test_predictions, test_targets, num_images, global_key=global_key)

        print(
            f"Found best F1 of {test_pf1_score_1:.4f} at beta {test_beta_1:.2f}.")

        self.writer.flush()

    def _set_learning_rate(self, new_lrs):
        if not isinstance(new_lrs, list):
            new_lrs = [new_lrs] * len(self.optimizer.param_groups)
        if len(new_lrs) != len(self.optimizer.param_groups):
            raise ValueError(
                "Length of `new_lrs` is not equal to the number of parameter groups "
                + "in the given optimizer"
            )

        for param_group, new_lr in zip(self.optimizer.param_groups, new_lrs):
            param_group["lr"] = new_lr

    def range_test(self, train_loader, val_loader, min_lr, num_iters=70):
        lrs = []
        losses = []

        writer = open('lrfinder.txt', 'w')
        # Make train_loader as iterators, so it's possible to iterate over them indefinitely
        train_iter = iter(train_loader)
        scaler = GradScaler()

        self.model.train()
        self._set_learning_rate(min_lr)
        iterator = tqdm(range(num_iters))
        scheduler = torch.optim.lr_scheduler.ExponentialLR(
            self.optimizer, gamma=1.18)
        try:
            for step in iterator:
                # Set initial LR
                self._set_learning_rate(scheduler.get_last_lr()[0])

                lr = self.optimizer.param_groups[0]["lr"]
                iterator.set_description(f"Testing LR: {lr:.6f}")

                # Actual training
                total_train_loss = 0.
                for _ in range(self.args['gradient_acc_iters']):
                    # Fetch data
                    train_batch = next(train_iter)
                    train_images = train_batch[0].to(self.args['device'])

                    train_images = torch.cat([train_images, train_images, train_images], -3)

                    # Normalize images
                    train_images = normalize(
                        train_images,
                        mean=self.mean,
                        std=self.std
                    )
                    train_classes = train_batch[1].to(self.args['device'])
                    with autocast():
                        train_pred_logits, train_sim_loss = self.model(
                            train_images)
                        train_cat_loss = self.criterion(
                            train_pred_logits, train_classes)
                        if train_sim_loss is not None:
                            train_loss = train_cat_loss + train_sim_loss
                        else:
                            train_sim_loss = torch.zeros(1)
                            train_loss = train_cat_loss

                        # Accumulate gradient
                        train_loss = train_loss / self.args['gradient_acc_iters']
                        reg = self.regularization(self.model) / self.args['gradient_acc_iters']

                    # Add L2 regularization
                    # Replaces pow(2.0) with abs() for L1 regularization
                    # l2_norm = sum(p.pow(2.0).sum() for p in self.model.parameters())
                    if train_loss.item() > 10:
                        raise ValueError("Finished")
                    scaler.scale(train_loss + reg).backward()
                    total_train_loss += train_loss.item()

                # Update optimizer
                scaler.step(self.optimizer)
                scaler.update()
                # Reset optimizer
                self.optimizer.zero_grad()

                # Test model on val loader
                total_val_loss = 0.
                for val_batch in val_loader:
                    # Fetch data
                    val_images = val_batch[0].to(self.args['device'])
                    val_images = torch.cat([val_images, val_images, val_images], -3)
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
                        val_cat_loss = self.criterion(
                            val_pred_logits, val_classes)
                        if val_sim_loss is not None:
                            val_loss = val_cat_loss + val_sim_loss
                        else:
                            val_sim_loss = torch.zeros(1)
                            val_loss = val_cat_loss

                    total_val_loss += val_loss.item()

                # Keep track of lrs and losses
                lrs.append(lr)
                losses.append(total_val_loss / len(val_loader))

                # Write it to disk
                writer.write(f"{total_val_loss / len(val_loader)} {lr}\n")
                writer.flush()

                # Reset model and optimizers
                self.__init_model()
                self.__init_optim()
                # self.model.load_state_dict(self.model_state_dict)
                # self.optimizer.load_state_dict(self.optim_state_dict)

                scheduler.step()
        except ValueError:
            pass

        writer.close()

        import matplotlib.pyplot as plt
        plt.plot(lrs, losses)
        plt.ticklabel_format(style='sci', axis='x', scilimits=(1, 4))
        plt.xscale('log')
        plt.savefig('lrfinder.png')
