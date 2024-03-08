from __future__ import print_function

import torch
import torch.nn as nn

class SupCLCPLoss(nn.Module):

    def __init__(self, temperature=0.07, base_temperature=0.07):
        super(SupCLCPLoss, self).__init__()
        self.temperature = temperature
        self.base_temperature = base_temperature

    def forward(self, features, labels):
        """
        Args:
            features: [bsz, out_layer_dim].
            labels: [bsz].
            mask: [bsz, bsz] for contrastive, mask_{i,j}=1 if sample i and j has the same class.
        Returns:
            A loss scalar.
        """

        device = (torch.device('cuda') if features.is_cuda else torch.device('cpu'))

        # check input dim
        if len(features.shape) != 2:
            print(features.shape)
            raise ValueError('`features` needs to be [bsz, out_layer_dim]')

        batch_size = features.shape[0]

        # check labels and create mask
        if labels is not None:
            labels = labels.contiguous().view(-1, 1)
            if labels.shape[0] != batch_size:
                raise ValueError('Num of labels does not match num of features')
            mask = torch.eq(labels, labels.T).float().to(device)
        else:
            raise ValueError('No labels are found')

        # compute logits
        feature_dot = torch.div(
            torch.matmul(features, features.T),
            self.temperature)

        # for numerical stability
        logits_max, _ = torch.max(feature_dot, dim=1, keepdim=True)
        logits = feature_dot - logits_max.detach()

        # mask-out self-contrast cases
        logits_mask = torch.scatter(
            input = torch.ones_like(mask),
            dim = 1,
            index = torch.arange(batch_size).view(-1, 1).to(device),
            value = 0
        )
        mask = mask * logits_mask

        # compute log_prob
        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))

        # compute mean of log-likelihood over positive
        mean_log_prob_pos = (mask * log_prob).sum(1) / mask.sum(1)

        # compute loss
        loss = - (self.temperature / self.base_temperature) * mean_log_prob_pos
        loss = loss.view(1, batch_size).mean()

        return loss