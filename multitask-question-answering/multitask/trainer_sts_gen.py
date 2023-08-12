# -*- coding: utf-8 -*-
"""
Created on Thu Apr  1 14:12:48 2021

@author: ozcan
"""

from typing import Any, Dict, Union

import torch
from torch import nn

from transformers import Trainer as HFTrainer
from transformers.file_utils import is_apex_available

if is_apex_available():
    from apex import amp

from utils import label_smoothed_nll_loss

class Trainer(HFTrainer):
    def __init__(self, label_smoothing: float = 0, lambda_mt5: float = 0, lambda_mnrl: float = 0, **kwargs):
        super().__init__(**kwargs)
        self.label_smoothing = label_smoothing
        self.lambda_mt5 = lambda_mt5
        self.lambda_mnrl = lambda_mnrl

    def compute_loss(
        self,
        model: nn.Module,
        inputs,
        return_outputs=False,
    ):  

        #print(f"input: {inputs.keys()}")
        labels = inputs.get("labels")
        #print(f"labels_mnrl:{labels_mnrl.shape}")
        decoder_input_ids = inputs.get("decoder_input_ids")
        #print(f"decoder_input_ids: {decoder_input_ids.shape}")
        # forward pass

        input_ids = inputs.get("input_ids")
        #print(f"input_ids: {input_ids}")
        attention_mask = inputs.get("attention_mask")

        outputs = model(**inputs)
        loss_mt5 = outputs[0]
        # print(f"loss_mt5: {loss_mt5}")

        def mean_pool(token_embeds, attention_mask):
            # reshape attention_mask to cover 768-dimensions embeddings
            in_mask = attention_mask.squeeze().expand(
                token_embeds.size()
            ).float()
            #in_mask = attention_mask.float()
            # perform mean-pooling but exclude padding tokens (specified by in mask)
            pool = torch.sum(token_embeds * in_mask, 1) / torch.clamp(
                in_mask.sum(1), min=1e-9
            )
            return pool

        cos_sim = torch.nn.CosineSimilarity()
        scores = []
        '''
        a = mean_pool(input_ids, attention_mask)
        print(f"a: {a.shape}")
        print(f"a: {a}")
        '''
        labels.resize_(input_ids.shape)
        '''
        p = mean_pool(labels, attention_mask)
        print(f"p: {p.shape}")
        print(f"p: {p}")
        '''
        #for a_i in a:
            # print(f"a_i.shape: {a_i.shape}")
            #scores.append(cos_sim(torch.reshape(a_i, a.shape[0]), p))
        #for a_i in range(a.shape[0]):
        #    scores.append(cos_sim(a[a_i], p[a_i]))
        anchor = input_ids.to(torch.float64)
        positive = labels.to(torch.float64)
        
        #print(f"anchor: {anchor.shape}")
        #print(f"positive: {positive.shape}")
        
        scores = []
        for a_i in anchor:
            scores.append(cos_sim(a_i.reshape(1, a_i.shape[0]), positive))
        #scores = cos_sim(anchor, positive)
        scores = torch.stack(scores)
        scores = scores.requires_grad_()
        #print(f"scores: {scores}")
        #print(f"scores.shape: {scores.shape}")

        labels_sim = torch.tensor(range(len(scores)), dtype=torch.long, device=scores.device)
        #print(f"labels_sim: {labels_sim}")

        loss_func = torch.nn.CrossEntropyLoss(ignore_index=-100)
        loss_mnrl = loss_func(scores, labels_sim)
        print(f"loss_mnrl: {loss_mnrl}\t loss_mt5: {loss_mt5}")

        loss = self.lambda_mt5 * loss_mt5 + self.lambda_mnrl * loss_mnrl
        print(f"loss : {loss}")

        return (loss, outputs) if return_outputs else loss


    # override to support label smoothing
    def _training_step(
        self, model: nn.Module, inputs: Dict[str, Union[torch.Tensor, Any]], optimizer: torch.optim.Optimizer
    ) -> float:
        model.train()
        for k, v in inputs.items():
            if isinstance(v, torch.Tensor):
                inputs[k] = v.to(self.args.device)


        # Our model outputs do not work with DataParallel, so forcing return tuple.
        if isinstance(model, nn.DataParallel):
            inputs["return_tuple"] = True

        if self.label_smoothing == 0:
            outputs = model(**inputs)
            loss = outputs[0]  # model outputs are always tuple in transformers (see doc)
        else:
            labels = inputs.pop("labels")
            labels[labels == -100] = model.config.pad_token_id
            outputs = model(**inputs)
            lprobs = torch.nn.functional.log_softmax(outputs[0], dim=-1)
            loss, nll_loss = label_smoothed_nll_loss(
                lprobs, labels, self.label_smoothing, ignore_index=model.config.pad_token_id
            )

        if self.args.n_gpu > 1:
            loss = loss.mean()  # mean() to average on multi-gpu parallel training
        if self.args.gradient_accumulation_steps > 1:
            loss = loss / self.args.gradient_accumulation_steps

        if self.args.fp16:
            with amp.scale_loss(loss, optimizer) as scaled_loss:
                scaled_loss.backward()
        else:
            loss.backward()

        print(f"loss: {loss.item()}")
        return loss.item()