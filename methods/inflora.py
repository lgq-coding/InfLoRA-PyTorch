# methods/inflora.py
import math
from copy import deepcopy

import logging
import numpy as np
from tqdm import tqdm
from sklearn.cluster import KMeans

import torch
import torch.nn as nn
from torch import optim
from torch.nn import functional as F
from torch.utils.data import DataLoader

from methods.base import BaseLearner
from utils.toolkit import tensor2numpy, accuracy
from models.sinet_inflora import SiNet
from models.vit_inflora import Attention_LoRA
from utils.schedulers import CosineSchedule


# -------------------------
# Helper utilities
# -------------------------
def _to_tensor(x, device=None, dtype=None):
    """
    Convert numpy array or torch tensor to torch.Tensor.
    If device specified, move tensor to that device.
    """
    if isinstance(x, torch.Tensor):
        return x.to(device) if device is not None else x
    else:
        # assume numpy array or scalar
        try:
            return torch.as_tensor(x, device=device, dtype=dtype)
        except Exception:
            return torch.as_tensor(x, dtype=dtype)


def _matmul(a, b):
    """
    Safe matrix multiply that accepts numpy arrays or torch tensors and returns a torch.Tensor.
    Ensures both operands are torch tensors and b placed on same device as a if possible.
    """
    if isinstance(a, np.ndarray):
        a = torch.as_tensor(a)
    if isinstance(b, np.ndarray):
        try:
            b = torch.as_tensor(b, device=a.device)
        except Exception:
            b = torch.as_tensor(b)
    if not isinstance(a, torch.Tensor):
        a = torch.as_tensor(a)
    if not isinstance(b, torch.Tensor):
        try:
            b = torch.as_tensor(b, device=a.device)
        except Exception:
            b = torch.as_tensor(b)
    return a @ b


# -------------------------
# InfLoRA class
# -------------------------
class InfLoRA(BaseLearner):

    def __init__(self, args):
        super().__init__(args)

        if args["net_type"] == "sip":
            self._network = SiNet(args)
        else:
            raise ValueError('Unknown net: {}.'.format(args["net_type"]))

        # initialize attention modules if they provide init_param
        for module in self._network.modules():
            if isinstance(module, Attention_LoRA) and hasattr(module, "init_param"):
                try:
                    module.init_param()
                except Exception:
                    pass

        self.args = args
        self.optim = args["optim"]
        self.EPSILON = args.get("EPSILON", 1e-8)
        self.init_epoch = args["init_epoch"]
        self.init_lr = args["init_lr"]
        self.init_lr_decay = args.get("init_lr_decay", 0.1)
        self.init_weight_decay = args.get("init_weight_decay", 0.0)
        self.epochs = args["epochs"]
        self.lrate = args["lrate"]
        self.lrate_decay = args.get("lrate_decay", 0.1)
        self.batch_size = args["batch_size"]
        self.weight_decay = args.get("weight_decay", 0.0)
        self.num_workers = args.get("num_workers", 4)
        self.lamb = args.get("lamb", 0.95)
        self.lame = args.get("lame", 1.0)
        self.total_sessions = args["total_sessions"]
        self.dataset = args["dataset"]

        self.topk = 1  # origin is 5
        self.class_num = self._network.class_num
        self.debug = False

        # feature_list: store as CPU torch.Tensor objects (columns are basis vectors)
        self.all_keys = []
        self.feature_list = []
        self.project_type = []

        # quick device
        self._device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # track current task index (starts at -1; incremental_train increments first)
        self._cur_task = -1
        self._known_classes = 0
        self._total_classes = 0

    def after_task(self):
        # after finishing task: update known classes
        self._known_classes = self._total_classes
        logging.info('Exemplar size: {}'.format(getattr(self, "exemplar_size", 0)))

    def incremental_train(self, data_manager):
        # called by trainer to run one incremental step
        self._cur_task += 1
        self._total_classes = self._known_classes + data_manager.get_task_size(self._cur_task)
        self._network.update_fc(self._total_classes)

        logging.info('Learning on {}-{}'.format(self._known_classes, self._total_classes))

        train_dataset = data_manager.get_dataset(np.arange(self._known_classes, self._total_classes),
                                                 source='train', mode='train')
        self.train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True,
                                       num_workers=self.num_workers)
        test_dataset = data_manager.get_dataset(np.arange(0, self._total_classes), source='test', mode='test')
        self.test_loader = DataLoader(test_dataset, batch_size=self.batch_size, shuffle=False,
                                      num_workers=self.num_workers)

        self._train(self.train_loader, self.test_loader)
        self.clustering(self.train_loader)

    def _train(self, train_loader, test_loader):
        """
        Core training loop for one incremental task.
        Key points:
         - collect current features into attention blocks (cur_matrix)
         - build B (bases) from SVD of cur_matrix, store B and freeze it
         - initialize A to zeros and set A.requires_grad = True (trainable)
         - ensure optimizer is created after parameter requires_grad are set
         - train only A_current and classifier_pool[current_task]
        """

        self._network.to(self._device)

        # initial freeze: freeze all parameters
        for name, param in self._network.named_parameters():
            try:
                param.requires_grad_(False)
            except Exception:
                pass

        # Warm-up pass to populate cur_matrix in attention modules (no grad)
        with torch.no_grad():
            # iterate one epoch over the train_loader to accumulate cur_matrix inside attention modules
            for i, (_, inputs, targets) in enumerate(train_loader):
                inputs = inputs.to(self._device)
                # call network in a mode that registers current features into attention blocks
                # we use get_cur_feat=True to ensure attention modules update their cur_matrix
                self._network(inputs, get_cur_feat=True)
                if self.debug and i > 10:
                    break

            # ---- LoRA initialization / update phase ----
            # helper to create LoRA pair with correct shapes:
            # lora_A: nn.Linear(in=embed_dim, out=r)  => weight shape (r, embed_dim)
            # lora_B: nn.Linear(in=r, out=embed_dim)  => weight shape (embed_dim, r)
            def _create_lora_pair(module, rank=None, device=None):
                r = rank or getattr(module, 'rank', None) or 4
                # try infer embed_dim
                embed_dim = getattr(module, 'embed_dim', None)
                if embed_dim is None:
                    try:
                        if hasattr(module, 'qkv') and hasattr(module.qkv, 'weight'):
                            embed_dim = module.qkv.weight.shape[1]
                        else:
                            embed_dim = getattr(self, 'embd_dim', 768)
                    except Exception:
                        embed_dim = getattr(self, 'embd_dim', 768)

                # NOTE: nn.Linear(in_features, out_features)
                new_A = nn.Linear(embed_dim, r, bias=False)   # weight (r, embed_dim)
                new_B = nn.Linear(r, embed_dim, bias=False)   # weight (embed_dim, r)

                # init: B small random, A zeros so initial LoRA is near-zero
                try:
                    nn.init.kaiming_uniform_(new_B.weight, a=math.sqrt(5))
                except Exception:
                    try:
                        nn.init.normal_(new_B.weight, std=0.02)
                    except Exception:
                        pass
                try:
                    nn.init.zeros_(new_A.weight)
                except Exception:
                    pass

                if device is not None:
                    new_A.to(device)
                    new_B.to(device)
                return new_B, new_A

            # Iterate all attention modules, ensure lists exist and lengths sufficient, then init B and A
            kk = 0
            for module in self._network.modules():
                if not isinstance(module, Attention_LoRA):
                    continue

                # Ensure ModuleList containers exist for LoRA pieces
                if not hasattr(module, 'lora_A_k') or module.lora_A_k is None:
                    module.lora_A_k = nn.ModuleList()
                if not hasattr(module, 'lora_B_k') or module.lora_B_k is None:
                    module.lora_B_k = nn.ModuleList()
                if not hasattr(module, 'lora_A_v') or module.lora_A_v is None:
                    module.lora_A_v = nn.ModuleList()
                if not hasattr(module, 'lora_B_v') or module.lora_B_v is None:
                    module.lora_B_v = nn.ModuleList()

                # get cur_matrix for this module (may be CPU tensor)
                cur_matrix = getattr(module, 'cur_matrix', None)
                if cur_matrix is None:
                    kk += 1
                    continue

                # move cur_matrix to device for SVD and further ops
                try:
                    cur_t = cur_matrix.to(self._device) if isinstance(cur_matrix, torch.Tensor) else torch.as_tensor(cur_matrix, device=self._device)
                except Exception:
                    cur_t = torch.as_tensor(cur_matrix, device=self._device)

                # apply projection if available
                try:
                    if hasattr(self, 'feature_mat') and kk < len(self.feature_mat):
                        fm = self.feature_mat[kk]
                        fm_t = fm.to(self._device) if isinstance(fm, torch.Tensor) else torch.as_tensor(fm, device=self._device)
                        if self.project_type[kk] == 'remove':
                            cur_t = cur_t - torch.mm(fm_t, cur_t)
                        else:
                            cur_t = torch.mm(fm_t, cur_t)
                except Exception:
                    pass

                # determine rank & embed dim
                desired_rank = getattr(module, 'rank', 4)
                embed_dim = getattr(module, 'embed_dim', None)
                if embed_dim is None:
                    try:
                        embed_dim = module.qkv.weight.shape[1]
                    except Exception:
                        embed_dim = getattr(self, 'embd_dim', 768)

                # Ensure lists are long enough for the current task index
                needed = self._cur_task + 1
                module_device = getattr(module, 'device', self._device)
                # create missing entries if any
                while len(module.lora_A_k) < needed:
                    newB, newA = _create_lora_pair(module, rank=desired_rank, device=self._device)
                    module.lora_B_k.append(newB)
                    module.lora_A_k.append(newA)
                while len(module.lora_A_v) < needed:
                    newBv, newAv = _create_lora_pair(module, rank=desired_rank, device=self._device)
                    module.lora_B_v.append(newBv)
                    module.lora_A_v.append(newAv)

                # Compute SVD on cur_t and use principal components to form B (the basis)
                try:
                    U, S, Vh = torch.linalg.svd(cur_t, full_matrices=False)
                    # U shape: (embed_dim, k)
                    topU = U[:, :desired_rank] if desired_rank <= U.shape[1] else U
                    # Store B = topU (embed_dim x r) scaled
                    # Ensure we have the current index allocated
                    idx = self._cur_task
                    # copy into B (embed_dim, r)
                    try:
                        module.lora_B_k[idx].weight.data.copy_(topU / math.sqrt(3))
                        module.lora_B_v[idx].weight.data.copy_(topU / math.sqrt(3))
                    except Exception:
                        # if shape mismatch (unexpected), re-create proper pair and assign
                        newB, newA = _create_lora_pair(module, rank=desired_rank, device=self._device)
                        module.lora_B_k[idx] = newB
                        module.lora_A_k[idx] = newA
                        newB.weight.data.copy_(topU / math.sqrt(3))
                        # replicate for v
                        newBv, newAv = _create_lora_pair(module, rank=desired_rank, device=self._device)
                        module.lora_B_v[idx] = newBv
                        module.lora_A_v[idx] = newAv
                        newBv.weight.data.copy_(topU / math.sqrt(3))
                    # Initialize A (r x embed_dim) to zeros and keep it trainable
                    try:
                        nn.init.zeros_(module.lora_A_k[idx].weight)
                        nn.init.zeros_(module.lora_A_v[idx].weight)
                    except Exception:
                        pass

                    # Freeze B (basis), enable A (coefficients) later explicitly
                    try:
                        module.lora_B_k[idx].weight.requires_grad_(False)
                        module.lora_B_v[idx].weight.requires_grad_(False)
                        module.lora_A_k[idx].weight.requires_grad_(True)
                        module.lora_A_v[idx].weight.requires_grad_(True)
                    except Exception:
                        pass

                except Exception as e:
                    print(f"DEBUG: SVD failed for attention module {kk} during LoRA init: {e}")

                # clear cur_matrix buffer
                try:
                    module.cur_matrix.zero_()
                    module.n_cur_matrix = 0
                except Exception:
                    pass

                kk += 1

        # ---- end of LoRA init/update phase ----

        # Enforce requires_grad globally: freeze all, then enable classifier head + A_current
        for name, p in self._network.named_parameters():
            try:
                p.requires_grad_(False)
            except Exception:
                pass

        # enable classifier head for current task
        task_idx = self._network.module.numtask - 1 if isinstance(self._network, nn.DataParallel) else self._network.numtask - 1
        if task_idx < 0:
            task_idx = 0

        for name, p in self._network.named_parameters():
            if f"classifier_pool.{task_idx}" in name:
                try:
                    p.requires_grad_(True)
                except Exception:
                    pass

        # enable A (not B) for current task in all attention modules
        for module in self._network.modules():
            if not isinstance(module, Attention_LoRA):
                continue
            t = self._cur_task
            if t < len(getattr(module, 'lora_A_k', [])):
                try:
                    module.lora_A_k[t].weight.requires_grad_(True)
                    module.lora_A_v[t].weight.requires_grad_(True)
                except Exception:
                    pass
            # ensure B is frozen
            if t < len(getattr(module, 'lora_B_k', [])):
                try:
                    module.lora_B_k[t].weight.requires_grad_(False)
                    module.lora_B_v[t].weight.requires_grad_(False)
                except Exception:
                    pass

        # show enabled params for debug
        enabled = {name for name, p in self._network.named_parameters() if p.requires_grad}
        total_trainable = sum(p.numel() for n, p in self._network.named_parameters() if p.requires_grad)
        print(f"After enforce: trainable param names count: {len(enabled)}, total trainable elements: {total_trainable:,}")
        if self.debug:
            print("Enabled parameter name samples:", list(enabled)[:40])

        # DataParallel wrap if requested
        if len(getattr(self, "_multiple_gpus", [])) > 1:
            self._network = nn.DataParallel(self._network, self._multiple_gpus)

        # Build optimizer & scheduler for this task using only trainable params
        trainable_params = [p for n, p in self._network.named_parameters() if p.requires_grad]
        if len(trainable_params) == 0:
            raise RuntimeError("No trainable parameters found for task {}, aborting.".format(self._cur_task))

        if self._cur_task == 0:
            if self.optim == 'sgd':
                optimizer = optim.SGD(trainable_params, momentum=0.9, lr=self.init_lr, weight_decay=self.init_weight_decay)
                scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer=optimizer, T_max=self.init_epoch)
            elif self.optim == 'adam':
                optimizer = optim.Adam(trainable_params, lr=self.init_lr, weight_decay=self.init_weight_decay, betas=(0.9, 0.999))
                scheduler = CosineSchedule(optimizer=optimizer, K=self.init_epoch)
            else:
                raise Exception("Unsupported optimizer")
            self.run_epoch = self.init_epoch
        else:
            if self.optim == 'sgd':
                optimizer = optim.SGD(trainable_params, momentum=0.9, lr=self.lrate, weight_decay=self.weight_decay)
                scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer=optimizer, T_max=self.epochs)
            elif self.optim == 'adam':
                optimizer = optim.Adam(trainable_params, lr=self.lrate, weight_decay=self.weight_decay, betas=(0.9, 0.999))
                scheduler = CosineSchedule(optimizer=optimizer, K=self.epochs)
            else:
                raise Exception("Unsupported optimizer")
            self.run_epoch = self.epochs

        # Run training loop
        self.train_function(train_loader, test_loader, optimizer, scheduler)

        # If DataParallel, unwrap
        if isinstance(self._network, nn.DataParallel):
            self._network = self._network.module

        # After training: collect cur_matrix again and update GPMs
        with torch.no_grad():
            for i, (_, inputs, targets) in enumerate(train_loader):
                inputs = inputs.to(self._device)
                self._network(inputs, get_cur_feat=True)
                if self.debug and i > 10:
                    break

            mat_list = []
            for module in self._network.modules():
                if isinstance(module, Attention_LoRA):
                    mat_list.append(deepcopy(module.cur_matrix))
                    try:
                        module.cur_matrix.zero_()
                        module.n_cur_matrix = 0
                    except Exception:
                        pass

            # Update DualGPM (keeps original repo semantics)
            self.update_DualGPM(mat_list)

            # Precompute projection matrices on device
            self.feature_mat = []
            for p in range(len(self.feature_list)):
                feat = self.feature_list[p]
                if not isinstance(feat, torch.Tensor):
                    feat_t = torch.as_tensor(feat, device=self._device)
                else:
                    feat_t = feat.to(self._device)
                Uf = feat_t @ feat_t.T
                Uf = Uf.to(self._device)
                print('Layer {} - Projection Matrix shape: {}'.format(p + 1, Uf.shape))
                self.feature_mat.append(Uf)

        return

    def train_function(self, train_loader, test_loader, optimizer, scheduler):
        prog_bar = tqdm(range(self.run_epoch))
        for _, epoch in enumerate(prog_bar):
            self._network.eval()
            losses = 0.0
            correct, total = 0, 0
            for i, (_, inputs, targets) in enumerate(train_loader):

                inputs, targets = inputs.to(self._device), targets.to(self._device)
                mask = (targets >= self._known_classes).nonzero().view(-1)
                if mask.numel() == 0:
                    continue
                inputs = torch.index_select(inputs, 0, mask)
                targets = torch.index_select(targets, 0, mask) - self._known_classes

                outputs = self._network(inputs)
                logits = outputs['logits']
                loss = F.cross_entropy(logits, targets)

                optimizer.zero_grad()
                loss.backward()

                # optional gradient check (debug)
                if self.debug and epoch == 0 and i == 0:
                    # check grads for current A params
                    t = self._cur_task
                    cnt = 0
                    for name, p in self._network.named_parameters():
                        if f"lora_A_k.{t}" in name or f"lora_A_v.{t}" in name:
                            print(f"GRAD CHECK A: {name} grad is None? {p.grad is None}")
                            cnt += 1
                        if f"lora_B_k.{t}" in name or f"lora_B_v.{t}" in name:
                            print(f"GRAD CHECK B: {name} grad is None? {p.grad is None}")
                            cnt += 1
                        if cnt > 20:
                            break

                optimizer.step()
                losses += loss.item()

                _, preds = torch.max(logits, dim=1)
                correct += preds.eq(targets.expand_as(preds)).cpu().sum().item()
                total += len(targets)
                if self.debug and i > 10:
                    break

            scheduler.step()
            train_acc = np.around(tensor2numpy(correct) * 100 / (total if total > 0 else 1), decimals=2)

            info = 'Task {}, Epoch {}/{} => Loss {:.3f}, Train_accy {:.2f}'.format(
                self._cur_task, epoch + 1, self.run_epoch, losses / (len(train_loader) if len(train_loader) > 0 else 1),
                train_acc)
            prog_bar.set_description(info)

        logging.info(info)

    def clustering(self, dataloader):
        features = []
        for i, (_, inputs, targets) in enumerate(dataloader):
            inputs, targets = inputs.to(self._device), targets.to(self._device)
            mask = (targets >= self._known_classes).nonzero().view(-1)
            if mask.numel() == 0:
                continue
            inputs = torch.index_select(inputs, 0, mask)
            with torch.no_grad():
                if isinstance(self._network, nn.DataParallel):
                    feature = self._network.module.extract_vector(inputs)
                else:
                    feature = self._network.extract_vector(inputs)
            feature = feature / feature.norm(dim=-1, keepdim=True)
            features.append(feature)
        if len(features) == 0:
            return
        features = torch.cat(features, 0).cpu().detach().numpy()
        clustering = KMeans(n_clusters=5, random_state=0).fit(features)
        # store CPU tensor
        self.all_keys.append(torch.tensor(clustering.cluster_centers_).cpu())

    def _evaluate(self, y_pred, y_true):
        ret = {}
        grouped = accuracy(y_pred, y_true, self._known_classes, self.class_num)
        ret['grouped'] = grouped
        ret['top1'] = grouped['total']
        return ret

    def _eval_cnn(self, loader):
        self._network.eval()
        y_pred, y_true = [], []
        y_pred_with_task = []
        y_pred_task, y_true_task = [], []
        for _, (_, inputs, targets) in enumerate(loader):
            inputs = inputs.to(self._device)
            targets = targets.to(self._device)

            with torch.no_grad():
                y_true_task.append((targets // self.class_num).cpu())

                if isinstance(self._network, nn.DataParallel):
                    outputs = self._network.module.interface(inputs)
                else:
                    outputs = self._network.interface(inputs)

            predicts = torch.topk(outputs, k=self.topk, dim=1, largest=True, sorted=True)[1].view(-1)  # [bs, topk]
            y_pred_task.append((predicts // self.class_num).cpu())

            outputs_with_task = torch.zeros_like(outputs)[:, :self.class_num]
            for idx, i in enumerate(targets // self.class_num):
                en, be = self.class_num * i, self.class_num * (i + 1)
                outputs_with_task[idx] = outputs[idx, en:be]
            predicts_with_task = outputs_with_task.argmax(dim=1)
            predicts_with_task = predicts_with_task + (targets // self.class_num) * self.class_num

            y_pred.append(predicts.cpu().numpy())
            y_pred_with_task.append(predicts_with_task.cpu().numpy())
            y_true.append(targets.cpu().numpy())

        return np.concatenate(y_pred), np.concatenate(y_pred_with_task), np.concatenate(y_true), torch.cat(y_pred_task), torch.cat(y_true_task)  # [N, topk]

    def _compute_accuracy_domain(self, model, loader):
        model.eval()
        correct, total = 0, 0
        for i, (_, inputs, targets) in enumerate(loader):
            inputs = inputs.to(self._device)
            with torch.no_grad():
                outputs = model(inputs)['logits']

            predicts = torch.max(outputs, dim=1)[1]
            correct += ((predicts % self.class_num).cpu() == (targets % self.class_num)).sum().item()
            total += len(targets)

        return np.around(tensor2numpy(correct) * 100 / (total if total > 0 else 1), decimals=2)

    def update_DualGPM(self, mat_list):
        """
        Update Dual Gradient Projection Matrix (DualGPM).
        mat_list: list of activation matrices (torch tensors or numpy arrays) for each layer (one per attention block)
        We store orthonormal basis vectors in self.feature_list (CPU tensors).
        """
        threshold = (self.lame - self.lamb) * self._cur_task / max(1, self.total_sessions) + self.lamb
        print('Threshold: ', threshold)

        # helper to convert activation to torch tensor on device
        def to_act(x):
            if isinstance(x, torch.Tensor):
                return x.to(self._device)
            else:
                return torch.as_tensor(x, device=self._device)

        if len(self.feature_list) == 0:
            # After First Task: simply derive bases from SVD of activation
            for i, activation in enumerate(mat_list):
                act_t = to_act(activation)
                try:
                    U, S, Vh = torch.linalg.svd(act_t, full_matrices=False)
                except Exception as e:
                    print(f"DEBUG: SVD failed in update_DualGPM first-task for layer {i}: {e}")
                    continue
                sval_total = (S ** 2).sum().item()
                sval_ratio = ((S ** 2).cpu().numpy()) / (sval_total + 1e-12)
                r = int(np.sum(np.cumsum(sval_ratio) < threshold))
                r = max(r, 1)
                Ucols = U[:, :r].cpu()
                self.feature_list.append(Ucols)
                if r < (act_t.shape[0] / 2):
                    self.project_type.append('remove')
                else:
                    self.project_type.append('retain')
        else:
            # For subsequent tasks, follow DualGPM logic (projected rep, criteria)
            for i, activation in enumerate(mat_list):
                act_t = to_act(activation)
                feat_cpu = self.feature_list[i]  # stored on CPU
                feat_t = feat_cpu.to(self._device)
                if self.project_type[i] == 'remove':
                    try:
                        U1, S1, Vh1 = torch.linalg.svd(act_t, full_matrices=False)
                    except Exception as e:
                        print(f"DEBUG: SVD failed in DualGPM remove branch layer {i}: {e}")
                        continue
                    sval_total = (S1 ** 2).sum().item()
                    # Projected Representation (Eq-8)
                    proj = _matmul(_matmul(feat_t, feat_t.T), act_t)
                    act_hat = act_t - proj
                    try:
                        U, S, Vh = torch.linalg.svd(act_hat, full_matrices=False)
                    except Exception as e:
                        print(f"DEBUG: SVD on projected act_hat failed layer {i}: {e}")
                        continue
                    sval_hat = (S ** 2).sum().item()
                    sval_ratio = ((S ** 2).cpu().numpy()) / (sval_total + 1e-12)
                    accumulated_sval = (sval_total - sval_hat) / (sval_total + 1e-12)

                    r = 0
                    for ii in range(sval_ratio.shape[0]):
                        if accumulated_sval < threshold:
                            accumulated_sval += sval_ratio[ii]
                            r += 1
                        else:
                            break
                    if r == 0:
                        print('Skip Updating DualGPM for layer: {}'.format(i + 1))
                        continue
                    # update GPM
                    Ui = torch.cat((feat_cpu.to(self._device), U[:, :r]), dim=1).cpu()
                    if Ui.shape[1] > Ui.shape[0]:
                        self.feature_list[i] = Ui[:, :Ui.shape[0]]
                    else:
                        self.feature_list[i] = Ui
                else:
                    # retain branch
                    try:
                        U1, S1, Vh1 = torch.linalg.svd(act_t, full_matrices=False)
                    except Exception as e:
                        print(f"DEBUG: SVD failed in DualGPM retain branch layer {i}: {e}")
                        continue
                    sval_total = (S1 ** 2).sum().item()
                    # Projected Representation (Eq-8)
                    proj = _matmul(_matmul(feat_t, feat_t.T), act_t)
                    act_hat = proj
                    try:
                        U, S, Vh = torch.linalg.svd(act_hat, full_matrices=False)
                    except Exception as e:
                        print(f"DEBUG: SVD on projected act_hat failed (retain) layer {i}: {e}")
                        continue
                    sval_hat = (S ** 2).sum().item()
                    sval_ratio = ((S ** 2).cpu().numpy()) / (sval_total + 1e-12)
                    accumulated_sval = sval_hat / (sval_total + 1e-12)

                    r = 0
                    for ii in range(sval_ratio.shape[0]):
                        if accumulated_sval >= (1 - threshold):
                            accumulated_sval -= sval_ratio[ii]
                            r += 1
                        else:
                            break
                    if r == 0:
                        print('Skip Updating DualGPM for layer: {}'.format(i + 1))
                        continue

                    # update GPM by Projected Representation (Eq-8)
                    U_part = U[:, :r].to(self._device) if r > 0 else torch.empty((feat_t.shape[0], 0), device=self._device)
                    Ui = (feat_cpu.to(self._device) - _matmul(_matmul(U_part, U_part.T), feat_cpu.to(self._device))).cpu()
                    try:
                        Ui_t = Ui.to('cpu') if isinstance(Ui, torch.Tensor) else torch.as_tensor(Ui, device='cpu')
                        Ui_U, Ui_S, Ui_V = torch.linalg.svd(Ui_t, full_matrices=False)
                        new_cols = max(1, self.feature_list[i].shape[1] - r)
                        self.feature_list[i] = Ui_U[:, :new_cols].cpu()
                    except Exception as e:
                        print(f"DEBUG: SVD on Ui failed for layer {i}: {e}")
                        continue

        # Final cleanup of feature_list according to project_type conditions
        print('-' * 40)
        print('Gradient Constraints Summary')
        print('-' * 40)
        for i in range(len(self.feature_list)):
            feat = self.feature_list[i]
            if self.project_type[i] == 'remove' and (feat.shape[1] > (feat.shape[0] / 2)):
                try:
                    U_f, S_f, V_f = torch.linalg.svd(_to_tensor(feat, device='cpu'), full_matrices=False)
                    if feat.shape[1] < U_f.shape[1]:
                        new_feature = U_f[:, feat.shape[1]:].cpu()
                    else:
                        new_feature = U_f[:, feat.shape[1]:].cpu()
                    self.feature_list[i] = new_feature
                    self.project_type[i] = 'retain'
                except Exception:
                    pass
            elif self.project_type[i] == 'retain':
                assert self.feature_list[i].shape[1] <= (self.feature_list[i].shape[0] / 2)
            print('Layer {} : {}/{} type {}'.format(i + 1, self.feature_list[i].shape[1], self.feature_list[i].shape[0], self.project_type[i]))
        print('-' * 40)

    def update_GPM(self, mat_list):
        """
        Simpler GPM update (without dual logic). Kept for compatibility.
        """
        threshold = (self.lame - self.lamb) * self._cur_task / max(1, self.total_sessions) + self.lamb
        print('Threshold: ', threshold)

        def to_act(x):
            if isinstance(x, torch.Tensor):
                return x.to(self._device)
            else:
                return torch.as_tensor(x, device=self._device)

        if len(self.feature_list) == 0:
            for activation in mat_list:
                act_t = to_act(activation)
                try:
                    U, S, Vh = torch.linalg.svd(act_t, full_matrices=False)
                except Exception as e:
                    print(f"DEBUG: SVD failed in update_GPM: {e}")
                    continue
                sval_total = (S ** 2).sum().item()
                sval_ratio = ((S ** 2).cpu().numpy()) / (sval_total + 1e-12)
                r = int(np.sum(np.cumsum(sval_ratio) < threshold))
                self.feature_list.append(U[:, :max(r, 1)].cpu())
        else:
            for i, activation in enumerate(mat_list):
                act_t = to_act(activation)
                try:
                    U1, S1, Vh1 = torch.linalg.svd(act_t, full_matrices=False)
                except Exception as e:
                    print(f"DEBUG: SVD failed in update_GPM branch: {e}")
                    continue
                sval_total = (S1 ** 2).sum().item()
                feat_t = _to_tensor(self.feature_list[i], device=self._device)
                act_hat = act_t - _matmul(_matmul(feat_t, feat_t.T), act_t)
                try:
                    U, S, Vh = torch.linalg.svd(act_hat, full_matrices=False)
                except Exception as e:
                    print(f"DEBUG: SVD on act_hat failed in update_GPM: {e}")
                    continue
                sval_hat = (S ** 2).sum().item()
                sval_ratio = ((S ** 2).cpu().numpy()) / (sval_total + 1e-12)
                accumulated_sval = (sval_total - sval_hat) / (sval_total + 1e-12)

                r = 0
                for ii in range(sval_ratio.shape[0]):
                    if accumulated_sval < threshold:
                        accumulated_sval += sval_ratio[ii]
                        r += 1
                    else:
                        break
                if r == 0:
                    print('Skip Updating GPM for layer: {}'.format(i + 1))
                    continue
                Ui = torch.cat((self.feature_list[i], U[:, :r].cpu()), dim=1)
                if Ui.shape[1] > Ui.shape[0]:
                    self.feature_list[i] = Ui[:, :Ui.shape[0]]
                else:
                    self.feature_list[i] = Ui

        print('-' * 40)
        print('Gradient Constraints Summary')
        print('-' * 40)
        for i in range(len(self.feature_list)):
            logging.info('Layer {} : {}/{}'.format(i + 1, self.feature_list[i].shape[1], self.feature_list[i].shape[0]))
        print('-' * 40)
