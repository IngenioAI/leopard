import torch
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm

from pytorch_kid34k.models.base_model import BaseModel, define_criterion, define_optimizer, define_scheduler, AWP
from pytorch_kid34k.models.base_network import define_net
from pytorch_kid34k.utils.util import AverageMeter, Accuracy

from pytorch_cifar100.utils import get_network

class Classifier(BaseModel):
    def __init__(self, args, test_mode=False):
        super().__init__(args)
        self.local_rank = args.local_rank
        self.model_name = args.model_names
        self.epoch = 1
        self.device = 'gpu' if torch.cuda.is_available() else 'cpu'
        self.net = define_net(
            self.model_name, 
            no_pretrained=args.no_pretrained, 
            n_classes=3 if self.args.use_3class else 2,
            drop_rate=args.drop_rate
        )
        
        # self.net = get_network()
        if self.device == 'gpu':
            self.net = self.net.cuda(self.local_rank) 
        if args.use_DDP:
            self.net = nn.SyncBatchNorm.convert_sync_batchnorm(self.net)
            self.net = DDP(self.net, device_ids=[args.local_rank])
        if not test_mode:
            self.criterion = define_criterion(args.criterion_name, args.label_smoothing)
            self.optimizer = define_optimizer(
                model=self.net, 
                name=args.optimizer_name, 
                lr=args.lr if not args.use_DDP else args.lr * args.eff_batch_size / 32,
                momentum=args.momentum,
                weight_decay=args.weight_decay,
                betas=args.betas
            )
            self.scheduler = define_scheduler(
                optimizer=self.optimizer, 
                name=args.scheduler_name, 
                step_size=args.steplr_step_size, 
                gamma=args.steplr_gamma, 
                T_0=args.cosinelr_T_0, 
                T_mult=args.cosinelr_T_mult,
                eta_min=args.cosinelr_eta_min
            )
            if "use_awp" in self.args.__dict__.keys() and self.args.use_awp:
                self.awp = AWP(
                    model=self.net,
                    optimizer=self.optimizer,
                    adv_lr=self.args.awp_lr,
                    adv_eps=self.args.awp_eps
                )
            else:
                self.awp = None
    def rand_bbox(self, size, lam):
        W = size[2]
        H = size[3]
        cut_rat = np.sqrt(1. - lam)
        cut_w = np.int_(W * cut_rat)
        cut_h = np.int_(H * cut_rat)

        # uniform
        cx = np.random.randint(W)
        cy = np.random.randint(H)

        bbx1 = np.clip(cx - cut_w // 2, 0, W)
        bby1 = np.clip(cy - cut_h // 2, 0, H)
        bbx2 = np.clip(cx + cut_w // 2, 0, W)
        bby2 = np.clip(cy + cut_h // 2, 0, H)

        return bbx1, bby1, bbx2, bby2
    
    def train_one_epoch(self, loader, log_pref=""):
        self.net.train()
        acc1_meter = AverageMeter()
        loss_meter = AverageMeter()
        self.optimizer.zero_grad()
        for iter_idx, (img, label, _) in enumerate(loader):
            img = img.cuda(self.local_rank)
            label = label.cuda(self.local_rank)
            r = np.random.rand(1)
            if self.args.use_cutmix and self.args.cutmix_beta > 0 and r < self.args.cutmix_prob:
                lam = np.random.beta(self.args.cutmix_beta, self.args.cutmix_beta)
                rand_index = torch.randperm(img.shape[0]).cuda(self.local_rank)
                label_a = label
                label_b = label[rand_index]
                bbx1, bby1, bbx2, bby2 = self.rand_bbox(img.shape, lam)
                img[:,:,bbx1:bbx2, bby1:bby2] = img[rand_index, :, bbx1:bbx2, bby1:bby2]
                lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (img.size()[-1] * img.size()[-2]))
                
                output = self.net(img)
                loss = self.criterion(output, label_a) * lam + self.criterion(output, label_b) * (1. - lam)
            else:
                output = self.net(img)
                loss = self.criterion(output, label) / self.args.accum_iter

            # AWP reference: https://www.kaggle.com/code/junkoda/fast-awp
            if self.awp is not None: 
                if self.epoch >= self.args.awp_warmup:
                    self.awp.perturb(img, label, self.epoch, self.criterion)
            loss.backward()
            if self.awp is not None:
                self.awp.restore()
            
            if (iter_idx+1) % self.args.accum_iter == 0:
                self.optimizer.step()
                self.optimizer.zero_grad()

            bs = img.shape[0]
            acc1_val = Accuracy(output, label)[0].item()
            loss_val = loss.item()
            acc1_meter.update(acc1_val, bs)
            loss_meter.update(loss_val, 1)
        self.scheduler.step()
        self.epoch += 1
        if log_pref:
            print("{}: Accuracy {:.3f}, Loss {:.3f}".format(log_pref, acc1_val, loss_val))
        return acc1_meter.avg, loss_meter.avg
    
    @torch.no_grad()
    def test(self, loader, log_pref=""):
        self.net.eval()
        acc1_meter = AverageMeter()
        loss_meter = AverageMeter()
        c_mat = np.zeros((3 if self.args.use_3class else 2,3))
       
        for img, label, class_ in loader:
            img = img.cuda(self.local_rank)
            label = label.cuda(self.local_rank)
            output = self.net(img)
            loss = self.criterion(output, label)
            
            bs = img.shape[0]
            acc1_val = Accuracy(output, label)[0].item()
            loss_val = loss.item()
            acc1_meter.update(acc1_val, bs)
            loss_meter.update(loss_val, 1)
            _, pred = torch.max(output, dim=-1)
            for o, c in zip(pred.tolist(), class_.tolist()):
                c_mat[o, c] += 1
        if log_pref:
            print("{}: Accuracy {:.3f}, Loss {:.3f}".format(log_pref, acc1_val, loss_val))
        return acc1_meter.avg, loss_meter.avg, c_mat
    
    @torch.no_grad()
    def inference(self, img):
        self.net.eval()
        if self.device == 'gpu':
            img = img.cuda(self.local_rank)
        output = self.net(img)
        prob = F.softmax(output, dim=-1)
        conf, pred = torch.max(prob, dim=-1)
        return {"conf": conf, "pred": pred}
    
    def save(self, to_path, acc, loss):
        save_path = f"{to_path}/epoch[{self.epoch}]_acc[{acc}].pth"
        if self.args.local_rank == 0 or self.args.local_rank == 1 or self.args.local_rank == 2 or self.args.local_rank == 3: 
            state = {
                'epoch': self.epoch + 1,
                'acc': acc,
                'loss': loss,
                'state': self.net.state_dict()
            }
            torch.save(state, save_path)
            # torch.save(self.net, to_path)
            # torch.save(self.net.state_dict(), to_path)
        return save_path
    
    def load(self, from_path, verbose=False):
        if self.device == 'gpu':
            state = torch.load(from_path, map_location=f"cuda:{self.args.local_rank}")
        else:
            state = torch.load(from_path, map_location=torch.device('cpu'))
        acc = state['acc']
        if verbose:
            print(f"Load model from {from_path}")
            print(f"Epoch {state['epoch']}, Acc: {state['acc']:.3f}, Loss: {state['loss']:.3f}")
        
        self.net.load_state_dict(state['state'])
        
        return acc
        # print(f"model is loaded from {from_path}")

    def get_lr(self):
        return self.scheduler.optimizer.param_groups[0]['lr']      

    @torch.no_grad()
    def predict_target_sensitivity(self, data_loader, m=2, epsilon=1e-3):
        self.net.eval()
        predict_list = []
        sensitivity_list = []
        target_list = []
        logits = []

        self.net.cuda(self.local_rank)
        with torch.no_grad():        
            for img, targets, _ in tqdm(data_loader):
                img = img.cuda(self.local_rank)
                # label = label.cuda(self.local_rank)
                output = self.net(img)

                predicts = F.softmax(output, dim=-1)
                predict_list.append(predicts.detach().data.cpu())
                target_list.append(targets)
                logits.append(output.cpu().detach())
                # predict_list.append(prob.detach().data.cpu())
                # target_list.append(label.detach().data.cpu())


                if len(img.size()) == 4:
                    x = img.repeat((m, 1, 1, 1))
                elif len(img.size()) == 3:
                    x = img.repeat((m, 1, 1))
                elif len(img.size()) == 2:
                    x = img.repeat((m, 1))
                u = torch.randn_like(x)
                evaluation_points = x + epsilon * u
                new_predicts = F.softmax(self.net(evaluation_points), dim=-1)
                diff = torch.abs(new_predicts - predicts.repeat((m, 1)))
                diff = diff.view(m, -1, 2) #self.num_cls=2
                sensitivity = diff.mean(dim=0) / epsilon
                sensitivity_list.append(sensitivity.detach().data.cpu())

        targets = torch.cat(target_list, dim=0)
        predicts = torch.cat(predict_list, dim=0)
        sensitivities = torch.cat(sensitivity_list, dim=0)
        logits = torch.cat(logits, dim=0)

        if self.args.special_test_ver3:
            indices = 0
        else:
            indices = torch.tensor(data_loader.batch_sampler.sampler.data_source.indices)

        if self.args.org_sample_test:
            #real label : 0
            mask = (targets == 0)
            targets = targets[mask]
            predicts = predicts[mask]
            sensitivities = sensitivities[mask]
            indices = indices[mask]
        

        return predicts, targets, sensitivities, indices, logits
    
    @torch.no_grad()
    def get_logit(self, loader, m, epsilon, n_classes):
        logit, probs, trgs, sens = [], [], [], []
        self.net.eval()
        self.net.cuda(self.local_rank)
        acc1_meter = AverageMeter()
        loss_meter = AverageMeter()
       
        for img, label, _ in tqdm(loader):
            img = img.cuda(self.local_rank)
            label = label.cuda(self.local_rank)
            output = self.net(img)
            prob = torch.softmax(output, dim=-1) 
            loss = self.criterion(output, label)
            
            if len(img.size()) == 4:
                x = img.repeat((m, 1, 1, 1))
            elif len(img.size()) == 3:
                x = img.repeat((m, 1, 1))
            elif len(img.size()) == 2:
                x = img.repeat((m, 1))

            u = torch.randn_like(x)
            evaluation_points = x + epsilon * u
            new_probs = F.softmax(self.net(evaluation_points), dim=-1)
            diff = torch.abs(new_probs - prob.repeat((m, 1)))
            diff = diff.view(m, -1, n_classes)
            sensitivity = diff.mean(dim=0) / epsilon

            bs = img.shape[0]
            acc1_val = Accuracy(output, label)[0].item()
            loss_val = loss.item()
            acc1_meter.update(acc1_val, bs)
            loss_meter.update(loss_val, 1)
            
            logit.append(output.detach().cpu())
            probs.append(prob.detach().cpu())
            sens.append(sensitivity.detach().cpu())
            trgs.append(label.detach().cpu())

        return loss_meter.avg, acc1_meter.avg, logit, probs, sens, trgs
  