import math
import torch
import torch.nn as nn
import copy

from models.vit_inflora import VisionTransformer, PatchEmbed, Block, resolve_pretrained_cfg, build_model_with_cfg, \
    checkpoint_filter_fn
from models.zoo import CodaPrompt
from copy import deepcopy


class ViT_lora_co(VisionTransformer):
    def __init__(
            self, img_size=224, patch_size=16, in_chans=3, num_classes=1000, global_pool='token',
            embed_dim=768, depth=12, num_heads=12, mlp_ratio=4., qkv_bias=True, representation_size=None,
            drop_rate=0., attn_drop_rate=0., drop_path_rate=0., weight_init='', init_values=None,
            embed_layer=PatchEmbed, norm_layer=None, act_layer=None, block_fn=Block, n_tasks=10, rank=64):
        super().__init__(img_size=img_size, patch_size=patch_size, in_chans=in_chans, num_classes=num_classes,
                         global_pool=global_pool,
                         embed_dim=embed_dim, depth=depth, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias,
                         representation_size=representation_size,
                         drop_rate=drop_rate, attn_drop_rate=attn_drop_rate, drop_path_rate=drop_path_rate,
                         weight_init=weight_init, init_values=init_values,
                         embed_layer=embed_layer, norm_layer=norm_layer, act_layer=act_layer, block_fn=block_fn,
                         n_tasks=n_tasks, rank=rank)

    def forward(self, x, task_id, register_blk=-1, get_feat=False, get_cur_feat=False):
        x = self.patch_embed(x)
        x = torch.cat((self.cls_token.expand(x.shape[0], -1, -1), x), dim=1)

        x = x + self.pos_embed[:, :x.size(1), :]
        x = self.pos_drop(x)

        prompt_loss = torch.zeros((1,), requires_grad=True).to(x.device)
        for i, blk in enumerate(self.blocks):
            x = blk(x, task_id, register_blk == i, get_feat=get_feat, get_cur_feat=get_cur_feat)

        x = self.norm(x)

        return x, prompt_loss


def _create_vision_transformer(variant, pretrained=False, **kwargs):
    if kwargs.get('features_only', None):
        raise RuntimeError('features_only not implemented for Vision Transformer models.')

    # NOTE this extra code to support handling of repr size for in21k pretrained models
    # pretrained_cfg = resolve_pretrained_cfg(variant, kwargs=kwargs)
    pretrained_cfg = resolve_pretrained_cfg(variant)
    default_num_classes = pretrained_cfg['num_classes']
    num_classes = kwargs.get('num_classes', default_num_classes)
    repr_size = kwargs.pop('representation_size', None)
    if repr_size is not None and num_classes != default_num_classes:
        repr_size = None

    model = build_model_with_cfg(
        ViT_lora_co, variant, pretrained,
        pretrained_cfg=pretrained_cfg,
        representation_size=repr_size,
        pretrained_filter_fn=checkpoint_filter_fn,
        pretrained_custom_load='npz' in pretrained_cfg['url'],
        **kwargs)
    return model


class SiNet(nn.Module):

    def __init__(self, args):
        super(SiNet, self).__init__()

        model_kwargs = dict(patch_size=16, embed_dim=768, depth=12, num_heads=12, n_tasks=args["total_sessions"],
                            rank=args["rank"])
        self.image_encoder = _create_vision_transformer('vit_base_patch16_224_in21k', pretrained=True, **model_kwargs)
        # print(self.image_encoder)
        # exit()

        # keep embd_dim available for head creation fallback
        self.embd_dim = args.get("embd_dim", 768)

        self.class_num = 1
        self.class_num = args["init_cls"]
        # create classifier_pool with provided total_sessions to pre-allocate heads
        # (this mirrors original behavior but we will ensure safety in update_fc as well)
        self.classifier_pool = nn.ModuleList([
            nn.Linear(self.embd_dim, self.class_num, bias=True)
            for i in range(args["total_sessions"])
        ])

        self.classifier_pool_backup = nn.ModuleList([
            nn.Linear(self.embd_dim, self.class_num, bias=True)
            for i in range(args["total_sessions"])
        ])

        # self.prompt_pool = CodaPrompt(args["embd_dim"], args["total_sessions"], args["prompt_param"])

        self.numtask = 0

    @property
    def feature_dim(self):
        # prefer image_encoder's out_dim if available
        try:
            return self.image_encoder.out_dim
        except Exception:
            return self.embd_dim

    def extract_vector(self, image, task=None):
        # safe task selection for image_encoder
        if task is None:
            task_to_use = max(0, self.numtask - 1)
        else:
            task_to_use = task
        # clip to available tasks if image_encoder expects certain range; image_encoder should handle task ids
        # but be defensive if necessary
        image_features, _ = self.image_encoder(image, task_to_use)
        image_features = image_features[:, 0, :]
        # image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        return image_features

    def forward(self, image, get_feat=False, get_cur_feat=False, fc_only=False):
        if fc_only:
            fc_outs = []
            for ti in range(self.numtask):
                # defensive: clip ti if out of range
                if ti < len(self.classifier_pool):
                    fc_out = self.classifier_pool[ti](image)
                else:
                    # fallback to last available head
                    fc_out = self.classifier_pool[-1](image)
                fc_outs.append(fc_out)
            return torch.cat(fc_outs, dim=1)

        logits = []
        # safe task id for encoder call
        task_id_to_use = max(0, self.numtask - 1)
        image_features, prompt_loss = self.image_encoder(image, task_id=task_id_to_use, get_feat=get_feat,
                                                         get_cur_feat=get_cur_feat)
        image_features = image_features[:, 0, :]
        image_features = image_features.view(image_features.size(0), -1)

        # Safely index classifier_pool: clip index to available heads and log mismatch if any
        if len(self.classifier_pool) == 0:
            raise RuntimeError("classifier_pool is empty — model not initialized with any classifier heads")
        idx = self.numtask - 1
        if idx < 0:
            idx = 0
        if idx >= len(self.classifier_pool):
            # debug log to help trace missing initialization
            print(
                f"DEBUG: requested classifier index {self.numtask - 1} but only {len(self.classifier_pool)} heads exist. Using last available head {len(self.classifier_pool) - 1}.")
            idx = len(self.classifier_pool) - 1

        for prompts in [self.classifier_pool[idx]]:
            logits.append(prompts(image_features))

        return {
            'logits': torch.cat(logits, dim=1),
            'features': image_features,
            'prompt_loss': prompt_loss
        }

    def interface(self, image, task_id=None):
        # safe task id resolution
        task_to_use = self.numtask - 1 if task_id is None else task_id
        if task_to_use < 0:
            task_to_use = 0
        image_features, _ = self.image_encoder(image, task_to_use)

        image_features = image_features[:, 0, :]
        image_features = image_features.view(image_features.size(0), -1)

        logits = []
        # use only heads that are currently valid
        for prompt in self.classifier_pool[:self.numtask]:
            logits.append(prompt(image_features))

        if len(logits) == 0:
            # fallback: use last available head
            logits = [self.classifier_pool[-1](image_features)]

        logits = torch.cat(logits, 1)
        return logits

    def interface1(self, image, task_ids):
        logits = []
        for index in range(len(task_ids)):
            tid = task_ids[index].item()
            # safe tid
            if tid < 0:
                tid = 0
            if tid >= len(self.classifier_pool):
                tid = len(self.classifier_pool) - 1
            image_features, _ = self.image_encoder(image[index:index + 1], task_id=tid)
            image_features = image_features[:, 0, :]
            image_features = image_features.view(image_features.size(0), -1)

            logits.append(self.classifier_pool_backup[task_ids[index].item()](image_features))

        logits = torch.cat(logits, 0)
        return logits

    def interface2(self, image_features):

        logits = []
        for prompt in self.classifier_pool[:self.numtask]:
            logits.append(prompt(image_features))

        if len(logits) == 0:
            logits = [self.classifier_pool[-1](image_features)]

        logits = torch.cat(logits, 1)
        return logits

    def update_fc(self, nb_classes):
        """
        Increase numtask and ensure classifier_pool contains at least numtask heads.
        Also ensure each attention module has LoRA slots for the new task (auto-expand).
        """
        self.numtask += 1

        # Ensure classifier_pool exists and is ModuleList
        if not hasattr(self, 'classifier_pool') or self.classifier_pool is None:
            self.classifier_pool = nn.ModuleList()

        # Determine device for new heads
        try:
            device = next(self.parameters()).device
        except StopIteration:
            device = torch.device('cpu')

        # Create new heads until length matches numtask (preserve previous cloning behavior)
        while len(self.classifier_pool) < self.numtask:
            if len(self.classifier_pool) > 0:
                # clone structure of the first head
                ref = self.classifier_pool[0]
                if isinstance(ref, nn.Linear):
                    new_head = nn.Linear(ref.in_features, ref.out_features, bias=(ref.bias is not None))
                    # initialize newly created head
                    try:
                        nn.init.kaiming_uniform_(new_head.weight, a=math.sqrt(5))
                    except Exception:
                        try:
                            nn.init.normal_(new_head.weight, std=0.02)
                        except Exception:
                            pass
                    if new_head.bias is not None:
                        nn.init.zeros_(new_head.bias)
                else:
                    # fallback: deepcopy and try reinit weights
                    new_head = deepcopy(ref)
                    try:
                        if hasattr(new_head, 'weight'):
                            nn.init.kaiming_uniform_(new_head.weight, a=math.sqrt(5))
                        if hasattr(new_head, 'bias') and new_head.bias is not None:
                            nn.init.zeros_(new_head.bias)
                    except Exception:
                        pass
                new_head.to(device)
                self.classifier_pool.append(new_head)
            else:
                # No existing head to copy from: create a Linear using best available dims
                in_features = getattr(self, 'embd_dim', None) or getattr(self, 'feature_dim', None) or 768
                out_features = getattr(self, 'class_num', None) or 1
                new_head = nn.Linear(in_features, out_features)
                try:
                    nn.init.kaiming_uniform_(new_head.weight, a=math.sqrt(5))
                except Exception:
                    try:
                        nn.init.normal_(new_head.weight, std=0.02)
                    except Exception:
                        pass
                if new_head.bias is not None:
                    nn.init.zeros_(new_head.bias)
                new_head.to(device)
                self.classifier_pool.append(new_head)

        # --- Ensure LoRA slots for all attention modules ---

        # helper: create one LoRA pair with correct shapes:
        # lora_A: nn.Linear(in=embed_dim, out=r)   -> weight (r, embed_dim)
        # lora_B: nn.Linear(in=r, out=embed_dim)   -> weight (embed_dim, r)
        def _create_one_lora_pair(module, rank=None, device_local=None):
            r = rank or getattr(module, 'rank', None) or getattr(self, 'rank', None) or 4
            # infer embed_dim
            embed_dim = getattr(module, 'embed_dim', None)
            if embed_dim is None:
                try:
                    if hasattr(module, 'qkv') and hasattr(module.qkv, 'weight'):
                        embed_dim = module.qkv.weight.shape[1]
                    else:
                        embed_dim = getattr(self, 'embd_dim', None) or getattr(self, 'feature_dim', None) or 768
                except Exception:
                    embed_dim = getattr(self, 'embd_dim', None) or getattr(self, 'feature_dim', None) or 768

            # A: nn.Linear(in=embed_dim, out=r) -> weight shape (r, embed_dim)
            A = nn.Linear(embed_dim, r, bias=False)
            # B: nn.Linear(in=r, out=embed_dim) -> weight shape (embed_dim, r)
            B = nn.Linear(r, embed_dim, bias=False)

            try:
                nn.init.kaiming_uniform_(B.weight, a=math.sqrt(5))
            except Exception:
                try:
                    nn.init.normal_(B.weight, std=0.02)
                except Exception:
                    pass
            try:
                nn.init.zeros_(A.weight)
            except Exception:
                pass

            if device_local is not None:
                A.to(device_local)
                B.to(device_local)
            return B, A

        # iterate encoder modules and ensure LoRA slots exist for the new task index
        task_idx = self.numtask - 1
        for module in self.image_encoder.modules():
            # if module exposes a custom ensure method, prefer it
            try:
                if hasattr(module, 'ensure_lora_for_task'):
                    # try to call with device and rank
                    try:
                        module.ensure_lora_for_task(task_idx, rank=getattr(module, 'rank', None), device=device)
                    except TypeError:
                        # fallback if the custom method signature differs
                        module.ensure_lora_for_task(task_idx)
                    continue
            except Exception:
                # ignore and fallback to manual creation
                pass

            # Heuristic: treat modules named 'Attention_LoRA' or those with qkv as attention blocks
            mod_name = module.__class__.__name__
            if mod_name != 'Attention_LoRA' and not (hasattr(module, 'qkv') and hasattr(module, 'num_heads')):
                continue

            # ensure ModuleLists exist
            if not hasattr(module, 'lora_A_k') or module.lora_A_k is None:
                module.lora_A_k = nn.ModuleList()
            if not hasattr(module, 'lora_B_k') or module.lora_B_k is None:
                module.lora_B_k = nn.ModuleList()
            if not hasattr(module, 'lora_A_v') or module.lora_A_v is None:
                module.lora_A_v = nn.ModuleList()
            if not hasattr(module, 'lora_B_v') or module.lora_B_v is None:
                module.lora_B_v = nn.ModuleList()

            # device for this module
            try:
                module_device = next(module.parameters()).device
            except Exception:
                module_device = device

            # repair existing entries if shape mismatched, and append until enough entries are present
            desired_rank = getattr(module, 'rank', None) or getattr(self, 'rank', None) or 4
            # infer embed_dim for shape checks
            embed_dim = getattr(module, 'embed_dim', None)
            if embed_dim is None:
                try:
                    if hasattr(module, 'qkv') and hasattr(module.qkv, 'weight'):
                        embed_dim = module.qkv.weight.shape[1]
                    else:
                        embed_dim = getattr(self, 'embd_dim', None) or getattr(self, 'feature_dim', None) or 768
                except Exception:
                    embed_dim = getattr(self, 'embd_dim', None) or getattr(self, 'feature_dim', None) or 768

            # repair existing k branch pairs
            existing_k = min(len(module.lora_A_k), len(module.lora_B_k))
            for t in range(existing_k):
                try:
                    Bw = module.lora_B_k[t].weight
                    Aw = module.lora_A_k[t].weight
                    if (Bw @ Aw).shape != (embed_dim, embed_dim):
                        # replace with correct-shaped pair
                        newB, newA = _create_one_lora_pair(module, rank=desired_rank, device_local=module_device)
                        module.lora_B_k[t] = newB
                        module.lora_A_k[t] = newA
                except Exception:
                    newB, newA = _create_one_lora_pair(module, rank=desired_rank, device_local=module_device)
                    if t < len(module.lora_B_k):
                        module.lora_B_k[t] = newB
                    else:
                        module.lora_B_k.append(newB)
                    if t < len(module.lora_A_k):
                        module.lora_A_k[t] = newA
                    else:
                        module.lora_A_k.append(newA)

            # repair existing v branch pairs
            existing_v = min(len(module.lora_A_v), len(module.lora_B_v))
            for t in range(existing_v):
                try:
                    Bv = module.lora_B_v[t].weight
                    Av = module.lora_A_v[t].weight
                    if (Bv @ Av).shape != (embed_dim, embed_dim):
                        newBv, newAv = _create_one_lora_pair(module, rank=desired_rank, device_local=module_device)
                        module.lora_B_v[t] = newBv
                        module.lora_A_v[t] = newAv
                except Exception:
                    newBv, newAv = _create_one_lora_pair(module, rank=desired_rank, device_local=module_device)
                    if t < len(module.lora_B_v):
                        module.lora_B_v[t] = newBv
                    else:
                        module.lora_B_v.append(newBv)
                    if t < len(module.lora_A_v):
                        module.lora_A_v[t] = newAv
                    else:
                        module.lora_A_v.append(newAv)

            # append until we have enough slots
            while len(module.lora_A_k) < task_idx + 1:
                newB, newA = _create_one_lora_pair(module, rank=desired_rank, device_local=module_device)
                module.lora_B_k.append(newB)
                module.lora_A_k.append(newA)
            while len(module.lora_A_v) < task_idx + 1:
                newBv, newAv = _create_one_lora_pair(module, rank=desired_rank, device_local=module_device)
                module.lora_B_v.append(newBv)
                module.lora_A_v.append(newAv)

    def classifier_backup(self, task_id):
        self.classifier_pool_backup[task_id].load_state_dict(self.classifier_pool[task_id].state_dict())

    def classifier_recall(self):
        self.classifier_pool.load_state_dict(self.old_state_dict)

    def copy(self):
        return copy.deepcopy(self)

    def freeze(self):
        for param in self.parameters():
            param.requires_grad = False
        self.eval()

        return self
