# ------------------------------------------------------------------------
# HOTR official code : hotr/models/hotr.py
# Copyright (c) Kakao Brain, Inc. and its affiliates. All Rights Reserved
# ------------------------------------------------------------------------
import torch
import torch.nn as nn
import torch.nn.functional as F
import copy
import time
import datetime

from hotr.util.misc import NestedTensor, nested_tensor_from_tensor_list
from .feed_forward import MLP

class HOTR(nn.Module):
    def __init__(self, detr,
                 num_hoi_queries,
                 num_actions,
                 interaction_transformer,
                 augpath_name,
                 share_dec_param,
                 stop_grad_stage,
                 freeze_detr,
                 share_enc,
                 pretrained_dec,
                 temperature,
                 hoi_aux_loss,
                 return_obj_class=None):
        super().__init__()

        # * Instance Transformer ---------------
        self.detr = detr
        if freeze_detr:
            # if this flag is given, freeze the object detection related parameters of DETR
            for p in self.parameters():
                p.requires_grad_(False)
        hidden_dim = detr.transformer.d_model
        # --------------------------------------

        # * Interaction Transformer -----------------------------------------
        self.num_queries = num_hoi_queries
        self.query_embed = nn.Embedding(self.num_queries, hidden_dim)
        self.H_Pointer_embed   = MLP(hidden_dim, hidden_dim, hidden_dim, 3)
        self.O_Pointer_embed   = MLP(hidden_dim, hidden_dim, hidden_dim, 3)
        self.action_embed = nn.Linear(hidden_dim, num_actions+1)
        # --------------------------------------------------------------------


        # * HICO-DET FFN heads ---------------------------------------------
        self.return_obj_class = (return_obj_class is not None)
        if return_obj_class: self._valid_obj_ids = return_obj_class + [return_obj_class[-1]+1]
        # ------------------------------------------------------------------
        # * Transformer Options ---------------------------------------------
        self.interaction_transformer = interaction_transformer

        if share_enc: # share encoder
            self.interaction_transformer.encoder = detr.transformer.encoder

        if pretrained_dec: # free variables for interaction decoder
            self.interaction_transformer.decoder = copy.deepcopy(detr.transformer.decoder)
            for p in self.interaction_transformer.decoder.parameters():
                p.requires_grad_(True)
        # ---------------------------------------------------------------------
        #Augmented paths

        self.aug_paths = augpath_name

        if 'p2' in augpath_name:
            if not share_dec_param:
                self.xtoHO_interaction_decoder = copy.deepcopy(self.interaction_transformer.decoder)
                self.HOtoI_interaction_decoder = copy.deepcopy(self.interaction_transformer.decoder)
            else:
                self.xtoHO_interaction_decoder = self.interaction_transformer.decoder
                self.HOtoI_interaction_decoder = self.interaction_transformer.decoder

            self.query_embed_HOtoI = nn.Embedding(self.num_queries, hidden_dim)
            self.query_embed_HOtoI2 = nn.Embedding(self.num_queries, hidden_dim)
            self.H_Pointer_embed_HOtoI   = MLP(hidden_dim, hidden_dim, hidden_dim, 3)
            self.O_Pointer_embed_HOtoI   = MLP(hidden_dim, hidden_dim, hidden_dim, 3)
            self.action_embed_HOtoI = nn.Linear(hidden_dim, num_actions+1)

        if 'p3' in augpath_name:
            if not share_dec_param:
                self.xtoHI_interaction_decoder = copy.deepcopy(self.interaction_transformer.decoder)
                self.HItoO_interaction_decoder = copy.deepcopy(self.interaction_transformer.decoder)
            else:
                self.xtoHI_interaction_decoder = self.interaction_transformer.decoder
                self.HItoO_interaction_decoder = self.interaction_transformer.decoder

            self.query_embed_HItoO = nn.Embedding(self.num_queries, hidden_dim)
            self.query_embed_HItoO2 = nn.Embedding(self.num_queries, hidden_dim)
            self.H_Pointer_embed_HItoO   = MLP(hidden_dim, hidden_dim, hidden_dim, 3)
            self.O_Pointer_embed_HItoO   = MLP(hidden_dim, hidden_dim, hidden_dim, 3)
            self.action_embed_HItoO = nn.Linear(hidden_dim, num_actions+1)

        if 'p4' in augpath_name:
            if not share_dec_param:
                self.xtoOI_interaction_decoder = copy.deepcopy(self.interaction_transformer.decoder)
                self.OItoH_interaction_decoder = copy.deepcopy(self.interaction_transformer.decoder)
            else:
                self.xtoOI_interaction_decoder = self.interaction_transformer.decoder
                self.OItoH_interaction_decoder = self.interaction_transformer.decoder

            self.query_embed_OItoH = nn.Embedding(self.num_queries, hidden_dim)
            self.query_embed_OItoH2 = nn.Embedding(self.num_queries, hidden_dim)
            self.H_Pointer_embed_OItoH  = MLP(hidden_dim, hidden_dim, hidden_dim, 3)
            self.O_Pointer_embed_OItoH  = MLP(hidden_dim, hidden_dim, hidden_dim, 3)
            self.action_embed_OItoH = nn.Linear(hidden_dim, num_actions+1)

        self.stop_grad_stage = stop_grad_stage

        # * Loss Options -------------------
        self.tau = temperature
        self.hoi_aux_loss = hoi_aux_loss
        # ----------------------------------

    def forward(self, samples: NestedTensor):
        if isinstance(samples, (list, torch.Tensor)):
            samples = nested_tensor_from_tensor_list(samples)
        
        # >>>>>>>>>>>>  BACKBONE LAYERS  <<<<<<<<<<<<<<<
        features, pos = self.detr.backbone(samples)
        bs = features[-1].tensors.shape[0]
        src, mask = features[-1].decompose()
        assert mask is not None
        # ----------------------------------------------

        # >>>>>>>>>>>> OBJECT DETECTION LAYERS <<<<<<<<<<
        start_time = time.time()
        hs, memory = self.detr.transformer(self.detr.input_proj(src), mask, self.detr.query_embed.weight, pos[-1])
        inst_repr = F.normalize(hs[-1], p=2, dim=2) # instance representations

        # Prediction Heads for Object Detection
        outputs_class = self.detr.class_embed(hs)
        outputs_coord = self.detr.bbox_embed(hs).sigmoid()
        object_detection_time = time.time() - start_time
        # -----------------------------------------------

        # >>>>>>>>>>>> HOI DETECTION LAYERS <<<<<<<<<<<<<<<
        start_time = time.time()
        assert hasattr(self, 'interaction_transformer'), "Missing Interaction Transformer."
        H_Pointer_reprs_bag,O_Pointer_reprs_bag,outputs_action=[],[],[]
        # main path P1
        interaction_hs= self.interaction_transformer(self.detr.input_proj(src), mask, self.query_embed.weight, pos[-1])[0] # interaction representations
        H_Pointer_reprs_bag.append(F.normalize(self.H_Pointer_embed(interaction_hs), p=2, dim=-1))
        O_Pointer_reprs_bag.append(F.normalize(self.O_Pointer_embed(interaction_hs), p=2, dim=-1))
        outputs_action.append(self.action_embed(interaction_hs))
        
        if len(self.aug_paths)!=0:
            pos_aug = pos[-1].flatten(2).permute(2, 0, 1)
            mask_aug = mask.flatten(1)
        
        # P2 (x->HO->I)
        if 'p2' in self.aug_paths:
            tgt_2 = torch.zeros_like(self.query_embed_HOtoI.weight.unsqueeze(1).repeat(1, bs, 1))
            hs_HOtoI = self.xtoHO_interaction_decoder(tgt_2,memory,memory_key_padding_mask=mask_aug, pos=pos_aug, query_pos=self.query_embed_HOtoI.weight.unsqueeze(1).repeat(1, bs, 1)).transpose(1,2)
            tgt_HOtoI = hs_HOtoI.transpose(1,2)[-1] if not self.stop_grad_stage else hs_HOtoI.clone().detach().transpose(1,2)[-1]
            hs2_HOtoI = self.HOtoI_interaction_decoder(tgt_HOtoI,memory,memory_key_padding_mask=mask_aug, pos=pos_aug, query_pos=self.query_embed_HOtoI2.weight.unsqueeze(1).repeat(1, bs, 1)).transpose(1,2)
            H_Pointer_reprs_bag.append(F.normalize(self.H_Pointer_embed_HOtoI(hs_HOtoI), p=2, dim=-1))
            O_Pointer_reprs_bag.append(F.normalize(self.O_Pointer_embed_HOtoI(hs_HOtoI), p=2, dim=-1))
            outputs_action.append(self.action_embed_HOtoI(hs2_HOtoI))
        # P3 (x->HI->O)
        if 'p3' in self.aug_paths:
            tgt_3 = torch.zeros_like(self.query_embed_HItoO.weight.unsqueeze(1).repeat(1, bs, 1))
            hs_HItoO = self.xtoHI_interaction_decoder(tgt_3,memory,memory_key_padding_mask=mask_aug, pos=pos_aug, query_pos=self.query_embed_HItoO.weight.unsqueeze(1).repeat(1, bs, 1)).transpose(1,2)
            tgt_HItoO = hs_HItoO.transpose(1,2)[-1] if not self.stop_grad_stage else hs_HItoO.clone().detach().transpose(1,2)[-1]
            hs2_HItoO = self.HItoO_interaction_decoder(tgt_HItoO,memory,memory_key_padding_mask=mask_aug, pos=pos_aug, query_pos=self.query_embed_HItoO2.weight.unsqueeze(1).repeat(1, bs, 1)).transpose(1,2)
            H_Pointer_reprs_bag.append(F.normalize(self.H_Pointer_embed_HItoO(hs_HItoO), p=2, dim=-1))
            O_Pointer_reprs_bag.append(F.normalize(self.O_Pointer_embed_HItoO(hs2_HItoO), p=2, dim=-1))
            outputs_action.append(self.action_embed_HItoO(hs_HItoO))
        # P4 (x->OI->H)
        if 'p4' in self.aug_paths:
            tgt_4 = torch.zeros_like(self.query_embed_OItoH.weight.unsqueeze(1).repeat(1, bs, 1))
            hs_OItoH = self.xtoOI_interaction_decoder(tgt_3,memory,memory_key_padding_mask=mask_aug, pos=pos_aug, query_pos=self.query_embed_OItoH.weight.unsqueeze(1).repeat(1, bs, 1)).transpose(1,2)
            tgt_OItoH = hs_OItoH.transpose(1,2)[-1] if not self.stop_grad_stage else hs_OItoH.clone().detach().transpose(1,2)[-1]
            hs2_OItoH = self.OItoH_interaction_decoder(tgt_OItoH,memory,memory_key_padding_mask=mask_aug, pos=pos_aug, query_pos=self.query_embed_OItoH2.weight.unsqueeze(1).repeat(1, bs, 1)).transpose(1,2)
            H_Pointer_reprs_bag.append(F.normalize(self.H_Pointer_embed_OItoH(hs2_OItoH), p=2, dim=-1))
            O_Pointer_reprs_bag.append(F.normalize(self.O_Pointer_embed_OItoH(hs_OItoH), p=2, dim=-1))
            outputs_action.append(self.action_embed_OItoH(hs_OItoH))

        inst_repr_all=inst_repr.transpose(1,2).repeat(1+len(self.aug_paths),1,1)

        H_Pointer_reprs_bag=torch.cat(H_Pointer_reprs_bag,1)
        O_Pointer_reprs_bag=torch.cat(O_Pointer_reprs_bag,1)
        
        outputs_hidx = [(torch.bmm(H_Pointer_repr, inst_repr_all)) / self.tau for H_Pointer_repr in H_Pointer_reprs_bag] #(dec_layer,(1+len(aug))*bs,dec_q,hidden_dim)
        outputs_oidx = [(torch.bmm(O_Pointer_repr, inst_repr_all)) / self.tau for O_Pointer_repr in O_Pointer_reprs_bag]

        outputs_action=torch.stack(outputs_action,dim=2) #(dec_layer,bs,1+#aug,dec_q,#action)
        
        # --------------------------------------------------
        hoi_detection_time = time.time() - start_time
        hoi_recognition_time = max(hoi_detection_time - object_detection_time, 0)
        # -------------------------------------------------------------------

        # [Target Classification]
        if self.return_obj_class:
            detr_logits = outputs_class[-1, ..., self._valid_obj_ids]
            o_indices = [output_oidx.max(-1)[-1].view(1+len(self.aug_paths),bs,self.num_queries).transpose(0,1) for output_oidx in outputs_oidx]
            obj_logit_stack = [torch.stack([detr_logits[batch_, o_idx, :] for batch_, o_idc in enumerate(o_indice) for o_idx in o_idc], 0) for o_indice in o_indices]
            outputs_obj_class = obj_logit_stack

        out = {
            "pred_logits": outputs_class[-1],
            "pred_boxes": outputs_coord[-1],
            "pred_hidx": outputs_hidx[-1],
            "pred_oidx": outputs_oidx[-1],
            "pred_actions": outputs_action[-1],
            "hoi_recognition_time": hoi_recognition_time,
        }

        if self.return_obj_class: out["pred_obj_logits"] = outputs_obj_class[-1]
        # import pdb;pdb.set_trace()
        if self.hoi_aux_loss: # auxiliary loss
            out['hoi_aux_outputs'] = \
                self._set_aux_loss_with_tgt(outputs_class, outputs_coord, outputs_hidx, outputs_oidx, outputs_action, outputs_obj_class) \
                if self.return_obj_class else \
                self._set_aux_loss(outputs_class, outputs_coord, outputs_hidx, outputs_oidx, outputs_action)

        return out

    @torch.jit.unused
    def _set_aux_loss(self, outputs_class, outputs_coord, outputs_hidx, outputs_oidx, outputs_action):
        return [{'pred_logits': a,  'pred_boxes': b, 'pred_hidx': c, 'pred_oidx': d, 'pred_actions': e}
                for a, b, c, d, e in zip(
                    outputs_class[-1:].repeat((outputs_action.shape[0], 1, 1, 1)),
                    outputs_coord[-1:].repeat((outputs_action.shape[0], 1, 1, 1)),
                    outputs_hidx[:-1],
                    outputs_oidx[:-1],
                    outputs_action[:-1])]

    @torch.jit.unused
    def _set_aux_loss_with_tgt(self, outputs_class, outputs_coord, outputs_hidx, outputs_oidx, outputs_action, outputs_tgt):
        return [{'pred_logits': a,  'pred_boxes': b, 'pred_hidx': c, 'pred_oidx': d, 'pred_actions': e, 'pred_obj_logits': f}
                for a, b, c, d, e, f in zip(
                    outputs_class[-1:].repeat((outputs_action.shape[0], 1, 1, 1)),
                    outputs_coord[-1:].repeat((outputs_action.shape[0], 1, 1, 1)),
                    outputs_hidx[:-1],
                    outputs_oidx[:-1],
                    outputs_action[:-1],
                    outputs_tgt[:-1])]