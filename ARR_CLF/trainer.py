import torch
import torchvision
import torch.nn as nn
from tqdm import tqdm
from torch.autograd import Variable
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
import torch.nn.functional as F
from sklearn.metrics import confusion_matrix
from model import LeNet, ANN
import config
from utils import AverageMeter, accuracy_clf
import time
import os
import shutil
import numpy as np
from network import network_forward
feature_dim = 22

class Trainer():
    def __init__(self, data_loader):
        if isinstance(data_loader, tuple):
            self.train_loader = data_loader[0]
            self.valid_loader = data_loader[1]
            self.num_train = len(self.train_loader.dataset)
            self.num_valid = len(self.valid_loader.dataset)            
        else:
            self.infer_loader = data_loader
            self.num_infer = len(self.infer_loader.dataset)
        self.model_name = 'ann'
        self.num_classes = 5


        self.epochs = config.epochs
        self.finetune_epochs = config.finetune_epochs
        self.lr = config.lr
        self.batch_size = config.batch_size
        self.loss_ce = nn.CrossEntropyLoss()

        self.device = config.device
        if self.model_name == 'lenet':
            self.model = LeNet()
        elif self.model_name == 'ann':
            self.model = ANN(feature_dim,32,self.num_classes)
        self.model.to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(),lr=self.lr,weight_decay=1e-5)  
        self.scheduler = ReduceLROnPlateau(self.optimizer,'min',patience=10,factor=0.8,min_lr=1e-8)
        self.best_valid_accs = 0

        self.is_qat = config.is_qat
        self.ckpt_dir = './ckpt'

    def train(self):

        print("\n[*] Train on {} samples, validate on {} samples".format(
            self.num_train, self.num_valid)
        )        

        for epoch in range(self.epochs):
            print(
                '\nEpoch: {}/{} - LR: {:.6f}'.format(
                    epoch+1, self.epochs, self.optimizer.param_groups[0]['lr']
                )
            )
            train_losses, train_accs = self.train_one_epoch(epoch)
            valid_losses, valid_accs = self.validate(epoch)
            is_best = valid_accs.avg > self.best_valid_accs
            msg1 = "train loss: {:.3f} - train acc: {:.3f} "
            msg2 = "- val loss: {:.3f} - val acc: {:.3f}"  
            if is_best:
                msg2 += " [*]"
            msg = msg1 + msg2
            print(msg.format(train_losses.avg, train_accs.avg, valid_losses.avg, valid_accs.avg))
            self.best_valid_accs = max(valid_accs.avg, self.best_valid_accs)
            self.save_checkpoint(
                {'epoch': epoch + 1,
                'model_state': self.model.state_dict(),
                'optim_state': self.optimizer.state_dict(),
                'best_valid_acc': self.best_valid_accs,
                }, is_best
            )            


    def train_one_epoch(self, epoch):
        batch_time =  AverageMeter()
        losses = AverageMeter()
        accs = AverageMeter()

        self.model.train()

        tic = time.time()
        with tqdm(total = self.num_train) as pbar:
            for idx, (images, labels) in enumerate(self.train_loader):
                images, labels = images.float().to(self.device, non_blocking=True), labels.float().to(self.device, non_blocking=True)
                images, labels = Variable(images), Variable(labels)   
                # print(images.view(images.shape[0], -1).shape)
                if self.is_qat:
                    images, labels = images.cpu(), labels.cpu()
                    images = images.view(images.shape[0], -1)
                output = self.model(images) 

                loss = self.loss_ce(output.float(), labels.long())
                acc = accuracy_clf(output, labels)

                losses.update(loss.item(), images.size()[0])
                accs.update(acc, images.size()[0])

                loss.requires_grad_(True)
                # compute gradients and update SGD
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                # measure elapsed time
                toc = time.time()
                batch_time.update(toc-tic)

                pbar.set_description(
                    (
                        "{:.1f}s - model1_loss: {:.3f} - model1_acc: {:.3f}".format(
                            (toc-tic), losses.avg, accs.avg
                        )
                    )
                )

                self.batch_size = images.shape[0]
                pbar.update(self.batch_size)

        return  losses, accs

    def validate(self, epoch):

        losses = AverageMeter()
        accs = AverageMeter()

        self.model.eval()

        for i, (images, labels) in enumerate(self.valid_loader):
            images, labels = images.float().to(self.device), labels.float().to(self.device)
            if self.is_qat:
                images, labels = images.cpu(), labels.cpu()
                images = images.view(images.shape[0], -1)
            images, labels = Variable(images), Variable(labels)

            output = self.model(images)
            loss = self.loss_ce(output.float(), labels.long())
            acc = accuracy_clf(output, labels)

            losses.update(loss.item(), images.size()[0])
            accs.update(acc, images.size()[0])

        return  losses, accs            

    def infer(self, is_finetune):
        if self.is_qat:
            self.model = self.model.cpu()
            self.model.eval()
            torch.quantization.fuse_modules(self.model,[['ann.linear1','ann.relu1'], ['ann.linear2', 'ann.relu2']], inplace = True)
            self.model = nn.Sequential(torch.quantization.QuantStub(),
                    *self.model.ann,
                    torch.quantization.DeQuantStub())
            self.model.train()
            self.model.qconfig = torch.quantization.get_default_qconfig('default')
            torch.quantization.prepare_qat(self.model, inplace=True)
            torch.quantization.convert(self.model, inplace=True) 
            


        self.load_checkpoint(best=True, is_finetune = is_finetune, is_qat = self.is_qat)
        
        self.model.eval()
        all_outputs = []
        all_images = []
        all_predicted = []
        for j, images in enumerate(self.infer_loader):

            images = images.float().to(self.device)
            if self.is_qat:
                images = images.cpu()

            images = Variable(images)
             
            # outputs = self.model(images.view(images.shape[0], -1))
            outputs = network_forward(images.view(images.shape[0], -1))
            predicted = torch.argmax(outputs.data, 1)
            all_outputs.append(outputs.data.cpu().numpy())
            all_images.append(images.cpu().numpy())
            all_predicted.extend(predicted.cpu().numpy())

        all_outputs = np.concatenate(all_outputs, axis=0)
        all_images = np.concatenate(all_images, axis=0)
        all_predicted = np.array(all_predicted)

        return all_predicted        

    def finetune(self):
        print("\n[*] Finetuning on {} samples".format(
            self.num_train)
        )        

        for epoch in range(self.finetune_epochs):
            print(
                '\nEpoch: {}/{} - LR: {:.6f}'.format(
                    epoch+1, self.finetune_epochs, self.optimizer.param_groups[0]['lr']
                )
            )
            train_losses, train_accs = self.finetune_one_epoch(epoch)
            valid_losses, valid_accs = self.validate(epoch)
            is_best = valid_accs.avg > self.best_valid_accs
            msg1 = "train loss: {:.3f} - train acc: {:.3f} "
            msg2 = "- val loss: {:.3f} - val acc: {:.3f}"  
            if is_best:
                msg2 += " [*]"
            msg = msg1 + msg2
            print(msg.format(train_losses.avg, train_accs.avg, valid_losses.avg, valid_accs.avg))
            self.best_valid_accs = max(valid_accs.avg, self.best_valid_accs)
            self.save_checkpoint(
                {'epoch': epoch + 1,
                'model_state': self.model.state_dict(),
                'optim_state': self.optimizer.state_dict(),
                'best_valid_acc': self.best_valid_accs,
                }, is_best, is_finetune = True
            )         

    def finetune_one_epoch(self,epoch):
        batch_time =  AverageMeter()
        losses = AverageMeter()
        accs = AverageMeter()
        if epoch == 0:
            print('successfully loaded model')
            self.load_checkpoint(best=True, is_finetune = False)
        self.model.train()

        tic = time.time()
        with tqdm(total = self.num_train) as pbar:
            for idx, (images, labels) in enumerate(self.train_loader):
                images, labels = images.float().to(self.device, non_blocking=True), labels.float().to(self.device, non_blocking=True)
                images, labels = Variable(images), Variable(labels)   
                # print(images.shape)
                output = self.model(images) 

                loss = self.loss_ce(output.float(), labels.long())
                acc = accuracy_clf(output, labels)

                losses.update(loss.item(), images.size()[0])
                accs.update(acc, images.size()[0])

                loss.requires_grad_(True)
                # compute gradients and update SGD
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                # measure elapsed time
                toc = time.time()
                batch_time.update(toc-tic)

                pbar.set_description(
                    (
                        "{:.1f}s - model1_loss: {:.3f} - model1_acc: {:.3f}".format(
                            (toc-tic), losses.avg, accs.avg
                        )
                    )
                )

                self.batch_size = images.shape[0]
                pbar.update(self.batch_size)


        return  losses, accs        
    
    def qat(self):
        self.model = self.model.cpu()
        self.optimizer = torch.optim.Adam(self.model.parameters(),lr=1e-5,weight_decay=1e-5)  
        self.load_checkpoint(best = True, is_finetune = False )
        self.model.eval()
        torch.quantization.fuse_modules(self.model,[['ann.linear1','ann.relu1'], ['ann.linear2', 'ann.relu2']], inplace = True)
        self.model = nn.Sequential(torch.quantization.QuantStub(),
                  *self.model.ann,
                  torch.quantization.DeQuantStub())
        self.model.train()
        self.model.qconfig = torch.quantization.get_default_qconfig('default')
        torch.quantization.prepare_qat(self.model, inplace=True)
        
        n_epochs = 50
        for epoch in range(n_epochs):
            print(
                '\nEpoch: {}/{} - LR: {:.6f}'.format(
                    epoch+1, n_epochs, self.optimizer.param_groups[0]['lr'],)
            )
            train_losses, train_accs = self.train_one_epoch(epoch)
            if epoch > 4:
                self.model.apply(torch.ao.quantization.disable_observer)

            self.model.eval()
            torch.quantization.convert(self.model, inplace=True)

            valid_losses, valid_accs = self.validate(epoch)

            # self.scheduler.step(valid_losses.avg)
            is_best = valid_accs.avg > self.best_valid_accs
            msg1 = "train loss: {:.3f} - train acc: {:.3f} "
            msg2 = "- val loss: {:.3f} - val acc: {:.3f}"  
            if is_best:
                msg2 += " [*]"
            msg = msg1 + msg2
            print(msg.format(train_losses.avg, train_accs.avg, valid_losses.avg, valid_accs.avg))
            


            
            self.best_valid_accs = max(valid_accs.avg, self.best_valid_accs)
            self.save_checkpoint(
                {'epoch': epoch + 1,
                'model_state': self.model.state_dict(),
                'optim_state': self.optimizer.state_dict(),
                'best_valid_acc': self.best_valid_accs,
                }, is_best, is_qat = self.is_qat
            )            

    def save_checkpoint(self, state, is_best, is_finetune = False, is_qat = False):
        """
        Save a copy of the model so that it can be loaded at a future
        date. This function is used when the model is being evaluated
        on the test data.

        If this model has reached the best validation accuracy thus
        far, a seperate file with the suffix `best` is created.
        """
        print("[*] Saving model to {}".format(self.ckpt_dir))

        if is_finetune:
            filename = self.model_name +'_model_finetuned.pth'
            ckpt_path = os.path.join(self.ckpt_dir, filename)
            torch.save(state, ckpt_path) 
            if is_best:
                filename = self.model_name + '_model_finetuned_best.pth'
                shutil.copyfile(
                    ckpt_path, os.path.join(self.ckpt_dir, filename)
                )            
        else:
            if is_qat:           
                filename = self.model_name +'_model_qat.pth'
                ckpt_path = os.path.join(self.ckpt_dir, filename)
                torch.save(state, ckpt_path)

                if is_best:
                    filename = self.model_name + '_model_best_qat.pth'
                    shutil.copyfile(
                        ckpt_path, os.path.join(self.ckpt_dir, filename)
                    )
            else:
                filename = self.model_name +'_model.pth'
                ckpt_path = os.path.join(self.ckpt_dir, filename)
                torch.save(state, ckpt_path)

                if is_best:
                    filename = self.model_name + '_model_best.pth'
                    shutil.copyfile(
                        ckpt_path, os.path.join(self.ckpt_dir, filename)
                    )        
    def load_checkpoint(self, best = True, is_finetune = False, is_qat = False):
        # print("[*] Loading model from {}".format(self.ckpt_dir))
        if is_finetune:
            filename = self.model_name +'_model_finetuned.pth'
            if best:
                filename = self.model_name + '_model_finetuned_best.pth'
            ckpt_path = os.path.join(self.ckpt_dir, filename)
            ckpt = torch.load(ckpt_path)
            self.best_valid_acc = ckpt['best_valid_acc']
            self.model.load_state_dict(ckpt['model_state'])  
        else:       
            if is_qat:
                print("qat_loading...........")
                filename = self.model_name +'_model_qat.pth'
                if best:
                    filename = self.model_name + '_model_best_qat.pth'                
            else:     
                filename = self.model_name +'_model.pth'
                if best:
                    filename = self.model_name + '_model_best.pth'
            ckpt_path = os.path.join(self.ckpt_dir, filename)
            ckpt = torch.load(ckpt_path)
            self.best_valid_acc = ckpt['best_valid_acc']
            self.model.load_state_dict(ckpt['model_state'])   
            if best:
                print(
                    "[*] Loaded {} checkpoint @ epoch {} "
                    "with best valid acc of {:.3f}".format(
                        filename, ckpt['epoch'], ckpt['best_valid_acc'])
                )
            else:
                print(
                    "[*] Loaded {} checkpoint @ epoch {}".format(
                        filename, ckpt['epoch'])
                )














        

