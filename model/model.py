# Global imports
import torch
import torch.nn as nn
import copy
from torchvision import transforms as T
from contextlib import nullcontext
from tqdm import tqdm
import random
from torch.nn import functional as F

# Local imports
from model.modules import BaseRGBModel, CustomRegNetY, FCLayers, MultFCLayers, ROISelector
from model.shift import make_temporal_shift, make_astrm
from model.impl.sam import disable_bn_running_stats, enable_bn_running_stats
from model.impl.softic import ProjectionHead, SoftICLoss
from util.constants import F3SET_ELEMENTS


class AdaSpot(BaseRGBModel):

    class Impl(nn.Module):

        def __init__(self, args = None):
            super().__init__()

            self._temp_arch = args.temporal_arch
            assert self._temp_arch in ['gru'], 'Only gru supported for now'

            self._feature_arch = args.feature_arch
            assert ('rny' in self._feature_arch), 'Only rny supported for now'

            # Max aggregation
            self.aggregation = args.aggregation
            assert self.aggregation in ['max'], 'Only max aggregation supported for now'

            self.lowres_loss = args.lowres_loss
            self.highres_loss = args.highres_loss
            self.roi_size = args.roi_size

            # Get main backbones (low-res and high-res) --> high-res a copy of low-res identical
            # pretrained_backbone=False avoids timm/HF ImageNet download when loading a full checkpoint after.
            _pretrained = getattr(args, 'pretrained_backbone', True)
            _pretrained_path = getattr(args, 'pretrained_backbone_path', None)
            if self._feature_arch.startswith(('rny002', 'rny004', 'rny006', 'rny008')):
                backbone = CustomRegNetY(
                    self._feature_arch,
                    pretrained=_pretrained,
                    pretrained_path=_pretrained_path,
                )
                self.d = backbone.ds[-1]
                backbone.head.fc = nn.Identity()
            else:
                raise NotImplementedError(self._feature_arch)

            if self._feature_arch.endswith('_gsm'):
                make_temporal_shift(backbone, args.clip_len, mode='gsm', blocks_temporal = args.blocks_temporal)
            elif self._feature_arch.endswith('_gsf'):
                make_temporal_shift(backbone, args.clip_len, mode='gsf', blocks_temporal = args.blocks_temporal)
            elif self._feature_arch.endswith('_astrm'):
                # ASTRM is a self-contained spatio-temporal refinement; no
                # GSM/GSF channel-shifting is applied alongside it.
                astrm_red = int(getattr(args, 'astrm_reduction', 4))
                astrm_k = int(getattr(args, 'astrm_kernel_size', 3))
                make_astrm(
                    backbone,
                    args.clip_len,
                    blocks_temporal=args.blocks_temporal,
                    reduction=astrm_red,
                    kernel_size=astrm_k,
                )
                print(
                    f'[backbone] ASTRM enabled '
                    f'(reduction={astrm_red}, kernel_size={astrm_k}, '
                    f'blocks_temporal={args.blocks_temporal})'
                )

            self.lowres_backbone = backbone
            # Adapt padding convolutions backbone
            self.swap_padding(self.lowres_backbone, pad_type=args.padding)  

            self.highres_backbone = copy.deepcopy(self.lowres_backbone)   

            self.lowres_linear = nn.Sequential(
                nn.Linear(self.d, self.d // 2),
                nn.ReLU(),
                nn.Linear(self.d // 2, self.d)
            )

            self.highres_linear = nn.Sequential(
                nn.Linear(self.d, self.d // 2),
                nn.ReLU(),
                nn.Linear(self.d // 2, self.d)
            )                            

            #Positional encoding (temporal)
            self.temp_enc = nn.Parameter(torch.normal(mean = 0, std = 1 / args.clip_len, size = (args.clip_len, self.d)))

            # Temopral module & Prediction head
            if self._temp_arch == 'gru':
                self._temp_fine = nn.GRU(input_size=self.d, hidden_size=self.d, num_layers=1, batch_first=True, bidirectional=True)
                if args.dataset == 'f3set':
                    self._pred_fine = MultFCLayers(self.d * 2, F3SET_ELEMENTS)
                else:
                    _pdim = getattr(args, 'pred_num_classes', args.num_classes + 1)
                    self._pred_fine = FCLayers(self.d * 2, _pdim)

                # Separate heads for high-res and low-res (auxiliar supervision)
                if self.highres_loss:
                    self._temp_fine_highres = nn.GRU(input_size=self.d, hidden_size=self.d, num_layers=1, batch_first=True, bidirectional=True)
                    if args.dataset == 'f3set':
                        self._pred_fine_highres = MultFCLayers(self.d * 2, F3SET_ELEMENTS)
                    else:
                        _pdim = getattr(args, 'pred_num_classes', args.num_classes + 1)
                        self._pred_fine_highres = FCLayers(self.d * 2, _pdim)
                if self.lowres_loss:
                    self._temp_fine_lowres = nn.GRU(input_size=self.d, hidden_size=self.d, num_layers=1, batch_first=True, bidirectional=True)
                    if args.dataset == 'f3set':
                        self._pred_fine_lowres = MultFCLayers(self.d * 2, F3SET_ELEMENTS)
                    else:
                        _pdim = getattr(args, 'pred_num_classes', args.num_classes + 1)
                        self._pred_fine_lowres = FCLayers(self.d * 2, _pdim)
            
            else:
                raise NotImplementedError(self._temp_arch)

            # ----------------------------------------------------------- #
            # Soft-IC contrastive head + memory-bank loss module.
            # The projection head maps post-GRU embeddings (B, T, d*2)
            # to a low-dim feature space (default 128, paper). The bank
            # state is owned by ``self.softic_loss`` so it travels with
            # ``.to(device)`` automatically. Both are created only when
            # Soft-IC is enabled and the dataset supports a single C-dim
            # soft label (i.e. not f3set).
            # ----------------------------------------------------------- #
            self._softic_enabled = bool(getattr(args, 'softic', False)) \
                and args.dataset != 'f3set'
            if self._softic_enabled:
                _sic_feat_dim = int(getattr(args, 'softic_feat_dim', 128))
                _sic_bank_size = int(getattr(args, 'softic_bank_size', 256))
                _sic_temp = float(getattr(args, 'softic_temperature', 0.1))
                _sic_warmup = int(getattr(args, 'softic_warmup_size', 32))
                _sic_omega_min = float(getattr(args, 'softic_omega_min', 0.1))
                # ``_sic_num_classes`` is the C the loss/bank operate over.
                # For 'bce_yolo' we drop the implicit-background channel so
                # the contrastive label space matches the classifier head.
                _cl = getattr(args, 'classification_loss', 'ce')
                if _cl == 'bce_yolo':
                    _sic_num_classes = int(args.num_classes)
                else:
                    _sic_num_classes = int(args.num_classes) + 1
                self.softic_proj = ProjectionHead(
                    in_dim=self.d * 2,
                    out_dim=_sic_feat_dim,
                )
                self.softic_loss = SoftICLoss(
                    num_classes=_sic_num_classes,
                    feat_dim=_sic_feat_dim,
                    bank_size=_sic_bank_size,
                    temperature=_sic_temp,
                    warmup_size=_sic_warmup,
                    omega_min=_sic_omega_min,
                )
                print(
                    f'[Soft-IC] head={self.d * 2}->{self.softic_proj.hidden_dim}'
                    f'->{_sic_feat_dim} | bank={_sic_bank_size} '
                    f'| C={_sic_num_classes} | tau={_sic_temp} '
                    f'| warmup={_sic_warmup} | omega_min={_sic_omega_min}'
                )
            else:
                self.softic_proj = None
                self.softic_loss = None
            
            #HR and LR resizing
            self.resizing_hr = T.Resize((args.hr_dim[0], args.hr_dim[1]))
            self.resizing_lr = T.Resize((args.lr_dim[0], args.lr_dim[1]))

            #HR and LR cropping (if needed)
            self.crop_hr = T.CenterCrop((args.hr_crop[0], args.hr_crop[1]))
            self.crop_lr = T.CenterCrop((args.lr_crop[0], args.lr_crop[1]))

            #Data augmentations
            self.augmentation = T.Compose([
                T.RandomApply([T.ColorJitter(hue = 0.2)], p = 0.25),
                T.RandomApply([T.ColorJitter(saturation = (0.7, 1.2))], p = 0.25),
                T.RandomApply([T.ColorJitter(brightness = (0.7, 1.2))], p = 0.25),
                T.RandomApply([T.ColorJitter(contrast = (0.7, 1.2))], p = 0.25),
                T.RandomApply([T.GaussianBlur(5)], p = 0.25),
                T.RandomHorizontalFlip(),
                T.RandomApply([
                    T.RandomAffine(
                        degrees=15,                    # rotate between -15 and 15 degrees
                        translate=(0.05, 0.05),        # translate up to 5% of image size
                        scale=(0.95, 1.05),            # scale between 95% and 105%
                        shear=(-5, 5, -5, 5)           # shear x between -5 and 5, y between -5 and 5
                    )
                ], p=0.25)
            ])

            #Standarization
            self.standarization = T.Compose([
                T.Normalize(mean = (0.485, 0.456, 0.406), std = (0.229, 0.224, 0.225)) #Imagenet mean and std
            ])

            self.unstandarization = T.Compose([
                T.Normalize(mean=[-m/s for m, s in zip([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])],
                    std=[1/s for s in (0.229, 0.224, 0.225)]) #Imagenet mean and std
            ])

            # RoI Selector module. ``roi_spatial_increase`` and
            # ``roi_size_step`` come from the config when present; the
            # defaults match the previous hard-coded behaviour.
            _roi_inc = int(getattr(args, 'roi_spatial_increase', 8))
            _roi_step = int(getattr(args, 'roi_size_step', 28))
            self.roi_selector = ROISelector(
                roi_size=args.roi_size,
                spatial_increase=_roi_inc,
                threshold=args.threshold,
                original_size=(args.hr_crop[0], args.hr_crop[1]),
                size_step=_roi_step,
            )
            
            self.do_auxiliar_supervision = True # Set to true as default (to false when preparing model just for inference)

        def swap_padding(self, module, pad_type='zero'):
            for name, child in module.named_children():
                if isinstance(child, nn.Conv2d):
                    if child.padding == (0, 0):
                        continue
                    padH, padW = child.padding

                    assert padH == padW, "Asymmetric padding not supported"

                    if pad_type == 'reflect':
                        pad_layer = nn.ReflectionPad2d(padH)
                        padding = 0
                    elif pad_type == 'replicate':
                        pad_layer = nn.ReplicationPad2d(padH)
                        padding = 0
                    else:
                        pad_layer = nn.Identity()
                        padding = padH

                    new_conv = nn.Conv2d(
                        in_channels=child.in_channels,
                        out_channels=child.out_channels,
                        kernel_size=child.kernel_size,
                        stride=child.stride,
                        padding=padding,
                        dilation=child.dilation,
                        groups=child.groups,
                        bias=(child.bias is not None)
                    )

                    new_conv.weight.data.copy_(child.weight.data)
                    if child.bias is not None:
                        new_conv.bias.data.copy_(child.bias.data)

                    new_layer = nn.Sequential(pad_layer, new_conv)

                    setattr(module, name, new_layer)

                else:
                    self.swap_padding(child, pad_type=pad_type)

        def forward(self, x, inference=False):
            
            x = self.normalize(x) #Normalize to 0-1

            if not inference:
                x = self.augment(x) #Augmentations per-batch

            x = self.standarize(x)
            
            # Resize and crop high-resolution 
            x_hr = self.resize(x, hr = True)
            x_hr = self.crop_hr(x_hr)

            # Resize and crop low-resolution
            x = self.resize(x)
            x = self.crop_lr(x)

            b, t, c, h, w = x.shape

            # Low-res processing
            im_feat, maps = self.lowres_backbone(x.view(-1, c, h, w), return_last_layer = True) # maps -> BT x C x H' x W'
            im_feat = im_feat.view(b, t, -1)  # (B, T, C)

            # Get high-res RoIs
            maps = maps.view(b, t, maps.shape[-3], maps.shape[-2], maps.shape[-1]) # B x T x C x H' x W'
            centers, sizes = self.roi_selector(maps.detach())
            rois = self.get_rois(x_hr, centers, sizes)

            # High-res processing of RoIs
            rois_feat = self.highres_backbone(rois.reshape(-1, c, self.roi_size[0], self.roi_size[1]))
            rois_feat = rois_feat.view(b, t, -1)
            
            # Projections
            im_feat = self.lowres_linear(im_feat)
            rois_feat = self.highres_linear(rois_feat)

            # High-res auxiliar supervision
            if self.highres_loss & self.do_auxiliar_supervision:
                im_feat_highres = rois_feat + self.temp_enc.expand(b, -1, -1)
                im_feat_highres = self._temp_fine_highres(im_feat_highres)[0]
                im_feat_highres = self._pred_fine_highres(im_feat_highres)

            # Low-res auxiliar supervision
            if self.lowres_loss & self.do_auxiliar_supervision:
                im_feat_lowres = im_feat + self.temp_enc.expand(b, -1, -1)
                im_feat_lowres = self._temp_fine_lowres(im_feat_lowres)[0]
                im_feat_lowres = self._pred_fine_lowres(im_feat_lowres)

            # Low-res + high-res fusion
            if self.aggregation == 'max':
                im_feat = torch.stack((im_feat, rois_feat), dim=-1)  # (B, T, C + C)
                im_feat = im_feat.max(dim=-1)[0]  # (B, T, C)
            else:
                raise NotImplementedError(self.aggregation)
            
            im_feat = im_feat + self.temp_enc.expand(b, -1, -1)  # Add temporal encoding

            # Temporal module and prediction head
            im_feat = self._temp_fine(im_feat)[0]
            # Save pre-FC, post-GRU per-frame embedding for contrastive losses
            # (e.g. Soft-IC). Shape: (B, T, d*2). Cheap pass-through.
            embedding = im_feat
            im_feat = self._pred_fine(im_feat)

            output_dict = {}
            output_dict['im_feat'] = im_feat
            output_dict['embedding'] = embedding
            # Soft-IC projection: only built when softic is enabled. The
            # head produces L2-normalized 128-D features used downstream
            # by AdaSpot._compute_loss + self.softic_loss.
            if self.softic_proj is not None:
                output_dict['embedding_proj'] = self.softic_proj(embedding)
            if self.highres_loss & self.do_auxiliar_supervision:
                output_dict['im_feat_highres'] = im_feat_highres
            if self.lowres_loss & self.do_auxiliar_supervision:
                output_dict['im_feat_lowres'] = im_feat_lowres

            return output_dict

        def get_rois(self, x, centers, sizes):
            b, t, c, h, w = x.shape
            ph, pw = self.roi_size

            sizes_list = self.roi_selector.sizes
            full_rois = torch.zeros((b, t, c, ph, pw), device=x.device)

            for size_h, size_w in zip(sizes_list[0], sizes_list[1]):
                mask = (sizes[..., 0] == size_h) & (sizes[..., 1] == size_w)
                if mask.sum() == 0:
                    continue
                
                # ---- Step 1: Convert normalized indicators -> integer pixel centers ----
                centers_h = (centers[..., 0] * h).long()
                centers_w = (centers[..., 1] * w).long()
                centers_h = torch.clamp(centers_h, size_h // 2, h - size_h // 2 - 1)
                centers_w = torch.clamp(centers_w, size_w // 2, w - size_w // 2 - 1)

                # ---- Step 2: Build relative offsets for roi grid ----
                dh = torch.arange(-(size_h // 2), size_h // 2, device=x.device)
                dw = torch.arange(-(size_w // 2), size_w // 2, device=x.device)

                # reshape for broadcasting
                dh = dh.view(1, 1, size_h, 1)  # (1,1,1,ph,1)
                dw = dw.view(1, 1, 1, size_w)  # (1,1,1,1,pw)

                # ---- Step 3: Absolute coordinates of each roi pixel ----    
                roi_h = centers_h[..., None, None] + dh   # (B,T,K,ph,pw)
                roi_w = centers_w[..., None, None] + dw   # (B,T,K,ph,pw)

                roi_h = roi_h.unsqueeze(2).expand(-1, -1, c, -1, w)  # (B,T,C,ph,pw)
                roi_w = roi_w.unsqueeze(2).expand(-1, -1, c, size_h, -1)

                x_exp = x.expand(b, t, c, h, w)

                rois = x_exp.gather(-2, roi_h).gather(-1, roi_w)

                if (ph != size_h) | (pw != size_w):
                    rois = F.interpolate(rois.view(-1, c, size_h, size_w), size=(ph, pw), mode='bilinear', align_corners=False)
                    rois = rois.view(b, t, c, ph, pw)
                full_rois[mask] += rois[mask]

            return full_rois

        def resize(self, x, hr = False):
            b, t, c, h, w = x.shape
            x = x.view(-1, c, h, w)  # (B, T, C, H, W) -> (B*T, C, H, W)
            if hr:
                x2 = self.resizing_hr(x)
            else:
                x2 = self.resizing_lr(x)
            return x2.view(b, t, c, x2.shape[-2], x2.shape[-1])  # (B*T, C, H, W) -> (B, T, C, H, W)
        
        def normalize(self, x):
            return x / 255.
        
        def augment(self, x):
            for i in range(x.shape[0]):
                x[i] = self.augmentation(x[i])
            return x

        def standarize(self, x):
            for i in range(x.shape[0]):
                x[i] = self.standarization(x[i])
            return x
        
        def unstandarize(self, x):
            for i in range(x.shape[0]):
                x[i] = self.unstandarization(x[i])
            return x

        def print_stats(self):
            print('Model params:',
                sum(p.numel() for p in self.parameters()))
            
        def clean_modules(self):
            modules = list(self._modules.keys())
            if '_temp_fine_highres' in modules:
                del self._modules['_temp_fine_highres']
            if '_pred_fine_highres' in modules:
                del self._modules['_pred_fine_highres']
            if '_pred_displ_highres' in modules:
                del self._modules['_pred_displ_highres']
            if '_temp_fine_lowres' in modules:
                del self._modules['_temp_fine_lowres']
            if '_pred_fine_lowres' in modules:
                del self._modules['_pred_fine_lowres']
            if '_pred_displ_lowres' in modules:
                del self._modules['_pred_displ_lowres']
            # Soft-IC modules are training-only.
            if 'softic_proj' in modules:
                del self._modules['softic_proj']
                self.softic_proj = None
            if 'softic_loss' in modules:
                del self._modules['softic_loss']
                self.softic_loss = None
            self._softic_enabled = False
            self.do_auxiliar_supervision = False
            print('Eliminated auxiliary supervision modules not required for inference.')


    def __init__(self, device=torch.device('cuda'), args_model=None, args_training=None, classes=None, elements=None):
        self.device = device
        args_model.lowres_loss = args_training.lowres_loss
        args_model.highres_loss = args_training.highres_loss

        cl = getattr(args_training, 'classification_loss', 'ce').lower()
        if cl not in ('ce', 'bce', 'bce_yolo'):
            raise ValueError(
                "training.classification_loss must be 'ce', 'bce', or 'bce_yolo'"
            )
        if args_model.dataset == 'f3set' and cl == 'bce_yolo':
            raise ValueError("classification_loss 'bce_yolo' is only for non-f3set datasets")
        self._classification_loss = cl

        if args_model.dataset != 'f3set':
            if cl == 'bce_yolo':
                args_model.pred_num_classes = args_model.num_classes
            else:
                args_model.pred_num_classes = args_model.num_classes + 1

        # Soft-IC (Soft Instance Contrastive) loss config.
        # Total objective when enabled: L_final = L_classification + lambda * L_SIC.
        self._softic = bool(getattr(args_training, 'softic', False))
        self._softic_lambda = float(getattr(args_training, 'softic_lambda', 1.0))
        self._softic_temperature = float(
            getattr(args_training, 'softic_temperature', 0.1)
        )
        self._softic_feat_dim = int(
            getattr(args_training, 'softic_feat_dim', 128))
        self._softic_bank_size = int(
            getattr(args_training, 'softic_bank_size', 256))
        self._softic_warmup_size = int(
            getattr(args_training, 'softic_warmup_size', 32))
        self._softic_omega_min = float(
            getattr(args_training, 'softic_omega_min', 0.1))
        if self._softic and args_model.dataset == 'f3set':
            raise ValueError(
                "training.softic=True is not supported for the f3set dataset; "
                "its multi-element label space does not map to a single soft "
                "label distribution. Set softic=false in the config."
            )
        # Propagate Soft-IC + classification-loss config to the inner Impl,
        # which builds the projection head + memory-bank module from these.
        args_model.softic = self._softic
        args_model.softic_feat_dim = self._softic_feat_dim
        args_model.softic_bank_size = self._softic_bank_size
        args_model.softic_temperature = self._softic_temperature
        args_model.softic_warmup_size = self._softic_warmup_size
        args_model.softic_omega_min = self._softic_omega_min
        args_model.classification_loss = cl
        if self._softic:
            print(
                f'[loss] Soft-IC enabled '
                f'(lambda={self._softic_lambda}, '
                f'temperature={self._softic_temperature}, '
                f'feat_dim={self._softic_feat_dim}, '
                f'bank_size={self._softic_bank_size}, '
                f'warmup={self._softic_warmup_size}, '
                f'omega_min={self._softic_omega_min})'
            )

        self._model = AdaSpot.Impl(args=args_model)
        self._model.print_stats()
        self._dataset = args_model.dataset

        self._model.to(device)

        self._num_classes = args_model.num_classes + 1
        self._num_logits = getattr(
            args_model, 'pred_num_classes', args_model.num_classes + 1
        )

        self._classes = classes
        self._elements = elements

        # For F3Set
        if self._elements is not None:
            self._inv_classes = {v: k for k, v in self._classes.items()}
            self._inv_elements = [{v: k for k, v in elem_dict.items()} for elem_dict in self._elements]

            self._combo_to_full_id = {}
            for event_str, class_id in self._classes.items():
                elems = event_str.split('_')
                combo = tuple(
                    self._elements[i][elems[i]] for i in range(len(elems))
                )
                self._combo_to_full_id[combo] = class_id

    def clean_modules(self):
        self._model.clean_modules()

    def _multiclass_classification_loss(self, logits, label, ce_kwargs, fg_weight):
        """Non-f3set head: logits (N, C'); label (N,) class index 0..N_fg or (N, N_fg+1) soft (mixup)."""
        c_full = self._num_classes
        if self._classification_loss == 'bce_yolo':
            n_fg = logits.shape[-1]
            if label.dim() == 1:
                tgt = torch.zeros(
                    label.shape[0], n_fg, device=logits.device, dtype=logits.dtype
                )
                fg = label > 0
                if fg.any():
                    tgt[fg, label[fg].long() - 1] = 1.0
            else:
                tgt = label[:, 1:].to(logits.dtype)
            pos_weight = None
            if fg_weight != 1:
                pos_weight = torch.tensor(
                    [float(fg_weight)] * n_fg,
                    device=logits.device,
                    dtype=logits.dtype,
                )
            per = F.binary_cross_entropy_with_logits(
                logits, tgt, pos_weight=pos_weight, reduction='none'
            )
            return per.sum(dim=1).mean()
        if self._classification_loss == 'bce':
            if label.dim() == 1:
                tgt = F.one_hot(label.long(), num_classes=c_full).to(logits.dtype)
            else:
                tgt = label.to(logits.dtype)
            pos_weight = None
            if fg_weight != 1:
                pos_weight = torch.tensor(
                    [1.0] + [float(fg_weight)] * (c_full - 1),
                    device=logits.device,
                    dtype=logits.dtype,
                )
            per = F.binary_cross_entropy_with_logits(
                logits, tgt, pos_weight=pos_weight, reduction='none')
            return per.sum(dim=1).mean()
        return F.cross_entropy(logits, label, **ce_kwargs)

    def _compute_loss(self, frame, label, labelE, ce_kwargs, fg_weight,
                      inference, enqueue_softic='now'):
        """Forward pass + total loss for one (already-mixup-prepared) batch.

        Used both by the standard training step and by the SAM/ASAM second
        forward pass. `labelE` is None for non-f3set datasets.

        Args:
            enqueue_softic: one of 'now', 'pending', or False (see
                ``SoftICLoss.forward`` docstring). 'now' is correct for
                vanilla SGD; 'pending' for the first SAM pass; False for
                the second SAM pass and for validation.
        """
        with torch.autocast("cuda", dtype=torch.bfloat16):
            output = self._model(frame, inference=inference)

            pred = output['im_feat']
            if self._model.highres_loss:
                pred_highres = output['im_feat_highres']

            if self._model.lowres_loss:
                pred_lowres = output['im_feat_lowres']

            loss = 0.

            # Final loss: F3Set vs other datasets
            if self._dataset == 'f3set':
                pred_action = pred[0].squeeze(-1)
                if isinstance(labelE, list):
                    label_action = labelE[0][:, :, 1]
                else:
                    label_action = labelE[:, 0]
                loss_final = F.binary_cross_entropy_with_logits(
                    pred_action, label_action,
                    pos_weight=torch.tensor([fg_weight]).to(self.device)
                )
                for i in range(1, len(pred)):
                    pred_cat = pred[i].reshape(-1, F3SET_ELEMENTS[i - 1])
                    if isinstance(labelE, list):
                        label_cat = labelE[i].reshape(-1, labelE[i].shape[-1])
                        mask = label_cat.sum(dim=1) > 0
                    else:
                        label_cat = labelE[:, i].long().reshape(-1)
                        mask = label_cat != -1
                    pred_cat = pred_cat[mask]
                    label_cat = label_cat[mask]
                    if label_cat.numel() == 0:
                        continue
                    loss_cat = F.cross_entropy(
                        pred_cat, label_cat, reduction='sum') / (
                        label_action.shape[0] * label_action.shape[1])
                    loss_final += loss_cat
            else:
                predictions = pred.reshape(-1, self._num_logits)
                loss_final = self._multiclass_classification_loss(
                    predictions, label, ce_kwargs, fg_weight)

            # High-res auxiliary loss
            if self._model.highres_loss:
                if self._dataset == 'f3set':
                    predictions_highres = pred_highres[0].squeeze(-1)
                    if isinstance(labelE, list):
                        label_action = labelE[0][:, :, 1]
                    else:
                        label_action = labelE[:, 0]
                    loss_highres = F.binary_cross_entropy_with_logits(
                        predictions_highres, label_action,
                        pos_weight=torch.tensor([fg_weight]).to(self.device)
                    )
                    for i in range(1, len(pred_highres)):
                        pred_cat = pred_highres[i].reshape(-1, F3SET_ELEMENTS[i - 1])
                        if isinstance(labelE, list):
                            label_cat = labelE[i].reshape(-1, labelE[i].shape[-1])
                            mask = label_cat.sum(dim=1) > 0
                        else:
                            label_cat = labelE[:, i].long().reshape(-1)
                            mask = label_cat != -1
                        pred_cat = pred_cat[mask]
                        label_cat = label_cat[mask]
                        if label_cat.numel() == 0:
                            continue
                        loss_cat = F.cross_entropy(
                            pred_cat, label_cat, reduction='sum') / (
                            label_action.shape[0] * label_action.shape[1])
                        loss_highres += loss_cat
                else:
                    predictions_highres = pred_highres.reshape(-1, self._num_logits)
                    loss_highres = self._multiclass_classification_loss(
                        predictions_highres, label, ce_kwargs, fg_weight)

            # Low-res auxiliary loss
            if self._model.lowres_loss:
                if self._dataset == 'f3set':
                    predictions_lowres = pred_lowres[0].squeeze(-1)
                    if isinstance(labelE, list):
                        label_action = labelE[0][:, :, 1]
                    else:
                        label_action = labelE[:, 0]
                    loss_lowres = F.binary_cross_entropy_with_logits(
                        predictions_lowres, label_action,
                        pos_weight=torch.tensor([fg_weight]).to(self.device)
                    )
                    for i in range(1, len(pred_lowres)):
                        pred_cat = pred_lowres[i].reshape(-1, F3SET_ELEMENTS[i - 1])
                        if isinstance(labelE, list):
                            label_cat = labelE[i].reshape(-1, labelE[i].shape[-1])
                            mask = label_cat.sum(dim=1) > 0
                        else:
                            label_cat = labelE[:, i].long().reshape(-1)
                            mask = label_cat != -1
                        pred_cat = pred_cat[mask]
                        label_cat = label_cat[mask]
                        if label_cat.numel() == 0:
                            continue
                        loss_cat = F.cross_entropy(
                            pred_cat, label_cat, reduction='sum') / (
                            label_action.shape[0] * label_action.shape[1])
                        loss_lowres += loss_cat
                else:
                    predictions_lowres = pred_lowres.reshape(-1, self._num_logits)
                    loss_lowres = self._multiclass_classification_loss(
                        predictions_lowres, label, ce_kwargs, fg_weight)

            if self._model.highres_loss & self._model.lowres_loss:
                loss += (loss_final + loss_highres + loss_lowres) / 3
            elif self._model.highres_loss:
                loss += (loss_final + loss_highres) / 2
            elif self._model.lowres_loss:
                loss += (loss_final + loss_lowres) / 2
            else:
                loss += loss_final

            # Soft-IC: L_final(.) = L_classification + lambda_SIC * L_SIC.
            # Only enabled for non-f3set; f3set is rejected at __init__ time.
            # Skipped when lambda is 0 to avoid the cost of the contrastive
            # similarity matrix on a zero-weighted term.
            if (self._softic and self._dataset != 'f3set'
                    and self._softic_lambda > 0
                    and self._model.softic_loss is not None):
                # Use the projection-head output (L2-normalized 128-D feats).
                proj = output['embedding_proj']             # (B, T, feat_dim)
                feat = proj.reshape(-1, proj.shape[-1])

                # Build per-instance soft labels in the same C-dim space the
                # contrastive loss / memory bank operate over. The bank's C
                # was set in Impl.__init__ to match the classifier's logits:
                # num_classes+1 for ce/bce, num_classes for bce_yolo.
                if label.dim() == 1:
                    soft = F.one_hot(
                        label.long(), num_classes=self._num_classes
                    ).to(feat.dtype)
                else:
                    soft = label.to(feat.dtype)

                if self._classification_loss == 'bce_yolo':
                    # YOLO-style head omits an explicit background channel;
                    # drop it so the contrastive label space matches the
                    # classifier's output space (and the bank's C).
                    soft = soft[:, 1:]

                l_sic = self._model.softic_loss(
                    feat, soft, enqueue=enqueue_softic
                )
                # SoftICLoss returns fp32 by design (numeric stability);
                # cast to the dtype of `loss` so the autocast graph stays
                # consistent and `.backward()` sees a single-dtype scalar.
                loss = loss + (
                    self._softic_lambda * l_sic
                ).to(loss.dtype)

        return loss

    def epoch(self, loader, optimizer=None, scaler=None, lr_scheduler=None,
            fg_weight=5, sam=None, grad_accum_steps=1):
        """Run one epoch of training (if ``optimizer`` is given) or validation.

        Args:
            grad_accum_steps: number of micro-batches to accumulate before
                stepping. ``effective_batch = micro_batch * grad_accum_steps``.
                Combined with SAM, each micro-batch still does its own
                first/second pass (per-micro-batch perturbation), but the
                sharp gradient is accumulated across ``grad_accum_steps``
                micro-batches before the optimizer step. This keeps memory
                low while recovering the paper's effective batch size.
        """

        if optimizer is None:
            inference = True
            self._model.eval()
        else:
            inference = False
            optimizer.zero_grad()
            self._model.train()

        N_accum = max(1, int(grad_accum_steps))

        # Per-parameter buffer for the accumulated *sharp* gradient when
        # SAM + grad accumulation are combined. Empty otherwise.
        sam_accum = {}

        # Positive classes weights
        ce_kwargs = {}
        if fg_weight != 1:
            ce_kwargs['weight'] = torch.FloatTensor(
                [1] + [fg_weight] * (self._num_classes - 1)).to(self.device)

        epoch_loss = 0.
        with torch.no_grad() if optimizer is None else nullcontext():
            for batch_idx, batch in enumerate(tqdm(loader)):
                frame = batch['frame'].to(self.device).float()
                label = batch['label'].to(self.device)
                if self._dataset == 'f3set':
                    labelE = batch['labelE'].float()
                    labelE = labelE.to(self.device)
                
                if 'frame2' in batch.keys():
                    frame2 = batch['frame2'].to(self.device).float()
                    label2 = batch['label2'].to(self.device)
                    if self._dataset == 'f3set':
                        labelE2 = batch['labelE2'].float()
                        labelE2 = labelE2.to(self.device)

                    l = [random.betavariate(0.2, 0.2) for _ in range(frame2.shape[0])]

                    label_dist = torch.zeros((label.shape[0], label.shape[1], self._num_classes)).to(self.device)
                    if self._dataset == 'f3set':
                        labelE_dist = [torch.zeros((labelE.shape[0], labelE.shape[2], F3SET_ELEMENTS[i-1])).to(self.device) if i > 0 else torch.zeros((labelE.shape[0], labelE.shape[2], 2)).to(self.device) for i in range(labelE.shape[1])]

                    for i in range(frame2.shape[0]):
                        frame[i] = l[i] * frame[i] + (1 - l[i]) * frame2[i]
                        lbl1 = label[i]
                        lbl2 = label2[i]

                        label_dist[i, range(label.shape[1]), lbl1] += l[i]
                        label_dist[i, range(label2.shape[1]), lbl2] += 1 - l[i]

                        if self._dataset == 'f3set':
                            for j in range(labelE.shape[1]):
                                lblE1 = labelE[i, j]
                                lblE2 = labelE2[i, j]
                                
                                if j == 0:
                                    labelE_dist[j][i, range(labelE.shape[2]), lblE1.long()] += l[i]
                                    labelE_dist[j][i, range(labelE.shape[2]), lblE2.long()] += 1 - l[i]
                                else:
                                    if (lblE1.long() != -1).any():
                                        t_idx = torch.arange(labelE.shape[2], device=lblE1.device)[lblE1.long() != -1]
                                        labelE_dist[j][i, t_idx, lblE1.long()[lblE1.long() != -1]] += l[i]
                                    if (lblE2.long() != -1).any():
                                        t_idx = torch.arange(labelE.shape[2], device=lblE2.device)[lblE2.long() != -1]
                                        labelE_dist[j][i, t_idx, lblE2.long()[lblE2.long() != -1]] += 1 - l[i]

                    label = label_dist
                    if self._dataset == 'f3set':
                        labelE = labelE_dist

                # Depends on whether mixup is used
                label = label.flatten() if len(label.shape) == 2 \
                    else label.view(-1, label.shape[-1])

                labelE_arg = labelE if self._dataset == 'f3set' else None

                # First (or only) forward.
                # Soft-IC bank policy:
                #   - validation        -> False (don't pollute bank)
                #   - vanilla training  -> 'now'   (enqueue right after loss)
                #   - SAM training      -> 'pending' (defer; flushed AFTER
                #     the second pass so the second pass's anchors never see
                #     their own features in the bank)
                if optimizer is None:
                    _enq_first = False
                elif sam is not None:
                    _enq_first = 'pending'
                else:
                    _enq_first = 'now'
                loss = self._compute_loss(
                    frame, label, labelE_arg, ce_kwargs, fg_weight, inference,
                    enqueue_softic=_enq_first)

                if optimizer is not None:
                    # Scale the per-micro-batch loss so that the accumulated
                    # gradient equals the mean over the effective batch.
                    micro_loss = loss / N_accum
                    is_step_iter = ((batch_idx + 1) % N_accum == 0) or \
                                   (batch_idx + 1 == len(loader))

                    if sam is None:
                        # Vanilla gradient accumulation: backward N times,
                        # step once. p.grad accumulates naturally.
                        micro_loss.backward()
                        if is_step_iter:
                            optimizer.step()
                            optimizer.zero_grad()
                            if lr_scheduler is not None:
                                lr_scheduler.step()
                    else:
                        # SAM / ASAM with grad accumulation:
                        # 1) First pass: backward to populate p.grad with
                        #    THIS micro-batch's first-pass gradient (we zero'd
                        #    it after the previous micro-batch's accum step).
                        micro_loss.backward()

                        # 2) Perturb based on this micro-batch's grad and
                        #    clear it so the second backward starts fresh.
                        sam.first_step(zero_grad=True)

                        # 3) Second pass at perturbed weights, with BN running
                        #    stats frozen so the running stats don't drift.
                        #    enqueue_softic=False: the bank must be unchanged
                        #    between first and second passes so the second
                        #    pass's anchors don't see their own (clean-pass)
                        #    features as positives. The pending stash from
                        #    pass 1 is flushed AFTER both passes complete.
                        disable_bn_running_stats(self._model)
                        loss2 = self._compute_loss(
                            frame, label, labelE_arg, ce_kwargs, fg_weight,
                            inference, enqueue_softic=False)
                        (loss2 / N_accum).backward()

                        # 4) Restore weights, but DO NOT step yet - we need
                        #    to accumulate the sharp gradient first.
                        sam.second_step(zero_grad=False, do_step=False)
                        enable_bn_running_stats(self._model)

                        # 4b) Flush the stashed first-pass features into the
                        #     Soft-IC memory bank now that both forward passes
                        #     are done.
                        if (self._softic
                                and self._model.softic_loss is not None):
                            self._model.softic_loss.flush_pending()

                        # 5) Add this micro-batch's sharp grad to the
                        #    accumulator, then clear p.grad so the next
                        #    micro-batch's first backward starts clean.
                        for group in optimizer.param_groups:
                            for p in group['params']:
                                if p.grad is None:
                                    continue
                                if p in sam_accum:
                                    sam_accum[p].add_(p.grad)
                                else:
                                    sam_accum[p] = p.grad.detach().clone()
                        optimizer.zero_grad()

                        # 6) Every N_accum micro-batches (or at epoch end),
                        #    move accumulated grads back into p.grad and step.
                        if is_step_iter:
                            for group in optimizer.param_groups:
                                for p in group['params']:
                                    if p in sam_accum:
                                        p.grad = sam_accum[p]
                            optimizer.step()
                            optimizer.zero_grad()
                            sam_accum.clear()
                            if lr_scheduler is not None:
                                lr_scheduler.step()

                epoch_loss += loss.detach().item()

        return epoch_loss / len(loader)

    def predict(self, seq, use_amp=True):
        
        if not isinstance(seq, torch.Tensor):
            seq = torch.FloatTensor(seq)
        if len(seq.shape) == 4: # (L, C, H, W)
            seq = seq.unsqueeze(0)
        if seq.device != self.device:
            seq = seq.to(self.device)
        seq = seq.float()

        self._model.eval()
        with torch.no_grad():
            with torch.autocast("cuda", dtype=torch.bfloat16) if use_amp else nullcontext():
                output = self._model(seq, inference=True)
            pred = output['im_feat']
            if self._dataset == 'f3set':
                pred, pred_cls = self.process_multiple_heads_prediction(
                    pred
                )
                return pred_cls.cpu().float().numpy(), pred.cpu().float().numpy()
            
            if self._classification_loss == 'bce_yolo':
                # YOLO-style: sigmoid per class, background = 1 - max class prob (implicit).
                cls_prob = torch.sigmoid(pred)
                bg = (1.0 - cls_prob.max(dim=2, keepdim=True).values).clamp(min=0.0)
                pred = torch.cat([bg, cls_prob], dim=2)
            elif self._classification_loss == 'bce':
                pred = torch.sigmoid(pred)
            else:
                pred = torch.softmax(pred, dim=2)
            pred_cls = torch.argmax(pred, dim=2)
            return pred_cls.cpu().float().numpy(), pred.cpu().float().numpy()

    def process_multiple_heads_prediction(self, pred):
        """
        pred[0]: binary head logits        [N, T, 1]
        pred[1:]: categorical head logits  list of [N, T, Ci]

        Returns:
            output_probs: [N, T, num_full_classes]
        """

        device = pred[0].device
        N, T = pred[0].shape[:2]

        # ------------------------------------------------------------------
        # Initialize full-class log-probabilities
        # ------------------------------------------------------------------
        output_log_probs = torch.full(
            (N, T, self._num_classes),
            float("-inf"),
            device=device
        )

        # Binary head (class 0 vs rest)
        # class 0 probability = 1 - sigmoid
        binary_logits = pred[0].squeeze(-1)  # [N, T]
        log_p_action = F.logsigmoid(binary_logits)  # log P(action) = log sigmoid
        # class 0 log-probability  
        log_p_no_action = F.logsigmoid(-binary_logits)  # log P(no action) = log (1 - sigmoid)

        output_log_probs[:, :, 0] = log_p_no_action  # P(class 0)

        # Categorical heads → log-probs
        log_probs = [F.log_softmax(l, dim=-1) for l in pred[1:]]
        K = len(log_probs)

        # Valid category combinations (precomputed mapping)
        valid_combos = torch.tensor(
            list(self._combo_to_full_id.keys()),
            device=device,
            dtype=torch.long
        )  # [M, K]

        full_class_ids = torch.tensor(
            [self._combo_to_full_id[tuple(c)] for c in valid_combos.tolist()],
            device=device,
            dtype=torch.long
        )  # [M]

        M = valid_combos.shape[0]

        # Compute joint log-probabilities
        joint_log_probs = torch.zeros(N, T, M, device=device)
        for k in range(K):
            # log_probs[k]: [N, T, Ck]
            # valid_combos[:, k]: [M]
            joint_log_probs += log_probs[k][:, :, valid_combos[:, k]]

        # Multiply by P(action) (log-space addition)
        joint_log_probs += log_p_action.unsqueeze(-1)

        # Scatter joint log-probs into full class space
        output_log_probs[:, :, full_class_ids] = joint_log_probs

        # Convert to probabilities (if needed downstream)
        output_pred = torch.exp(output_log_probs)
        output_pred_cls = torch.argmax(output_pred, axis=2)

        return output_pred, output_pred_cls