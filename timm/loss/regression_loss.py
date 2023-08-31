import torch
from torch import nn
import torch.nn.functional as F


def weighted_mse_loss(inputs, targets, weights=torch.tensor(100.0)):
    loss = (inputs - targets) ** 2
    if weights is not None:
        loss *= weights.expand_as(loss)
    loss = torch.mean(loss)
    return loss


# def weighted_l1_loss(inputs, targets, weights=None):
#     loss = F.l1_loss(inputs, targets, reduction='none')
#     bs = inputs.shape[0]
#     if weights is not None:
#         loss *= weights.expand_as(loss)
#     _, idxs = loss.topk(bs-1, largest=False)
#     loss = loss.index_select(0, idxs)
#     loss = torch.mean(loss)
#     return loss

def weighted_l1_loss(inputs, targets, weights=None):
    loss = F.l1_loss(inputs, targets, reduction='none')
    if weights is not None:
        loss *= weights.expand_as(loss)
    loss = torch.mean(loss)
    return loss

def weighted_l1_dex_loss(inputs, targets, weights=None):
    labels = torch.tensor(torch.range(1,191)).cuda()
    inputs = torch.softmax(inputs, dim=1)
    inputs = labels * inputs
    inputs = torch.sum(inputs, dim=1)
    loss = F.l1_loss(inputs, targets, reduction='none')
    if weights is not None:
        loss *= weights.expand_as(loss)
    loss = torch.mean(loss)
    return loss

def weighted_focal_mse_loss(inputs, targets, weights=None, activate='sigmoid', beta=.2, gamma=1):
    loss = (inputs - targets) ** 2
    loss *= (torch.tanh(beta * torch.abs(inputs - targets))) ** gamma if activate == 'tanh' else \
        (2 * torch.sigmoid(beta * torch.abs(inputs - targets)) - 1) ** gamma
    if weights is not None:
        loss *= weights.expand_as(loss)
    loss = torch.mean(loss)
    return loss


def weighted_focal_l1_loss(inputs, targets, weights=None, activate='sigmoid', beta=.2, gamma=1):
    loss = F.l1_loss(inputs, targets, reduction='none')
    loss *= (torch.tanh(beta * torch.abs(inputs - targets))) ** gamma if activate == 'tanh' else \
        (2 * torch.sigmoid(beta * torch.abs(inputs - targets)) - 1) ** gamma
    if weights is not None:
        loss *= weights.expand_as(loss)
    loss = torch.mean(loss)
    return loss


def weighted_huber_loss(inputs, targets, weights=None, beta=1.):
    l1_loss = torch.abs(inputs - targets)
    cond = l1_loss < beta
    loss = torch.where(cond, 0.5 * l1_loss ** 2 / beta, l1_loss - 0.5 * beta)
    if weights is not None:
        loss *= weights.squeeze().expand_as(loss)
    loss = torch.mean(loss)
    return loss



def coral_loss(logits, levels, imp, mode='coral', use_ousm=False):
    bs = logits.shape[0]
    if mode == 'coral':
        loss = (-torch.sum((F.logsigmoid(logits)*levels
                        + (F.logsigmoid(logits) - logits)*(1-levels))*imp,
            dim=1))
    elif mode == 'ordinal':
        loss = (-torch.sum((F.log_softmax(logits, dim=2)[:, :, 1]*levels
                      + F.log_softmax(logits, dim=2)[:, :, 0]*(1-levels))*imp, dim=1))


    if use_ousm:
        _, idxs = loss.topk(bs-1, largest=False)
        loss = loss.index_select(0, idxs)
    return torch.mean(loss)


def loss_conditional_v2(logits, y_train, imp, mode='corn', NUM_CLASSES=192):
    """Compared to the previous conditional loss, here, the loss is computed 
       as the average loss of the total samples, instead of firstly averaging 
       the cross entropy inside each task and then averaging over tasks equally. 
    """
    sets = []
    for i in range(NUM_CLASSES-1):
        label_mask = y_train > i-1
        label_tensor = (y_train[label_mask] > i).to(torch.int64)
        sets.append((label_mask, label_tensor))

    num_examples = 0
    losses = 0.
    for task_index, s in enumerate(sets):
        train_examples = s[0]
        train_labels = s[1]

        if len(train_labels) < 1:
            continue

        num_examples += len(train_labels)
        pred = logits[train_examples, task_index]
        
        loss = -torch.sum( F.logsigmoid(pred)*train_labels
                                + (F.logsigmoid(pred) - pred)*(1-train_labels) )
        losses += loss
    return losses/num_examples


#     if weights is not None:
#         loss *= weights.expand_as(loss)
#     
#     loss = loss.index_select(0, idxs)

class MeanVarianceLoss(nn.Module):

    def __init__(self, lambda_1, lambda_2, start_age=1, end_age=191):
        super().__init__()
        self.lambda_1 = lambda_1
        self.lambda_2 = lambda_2
        self.start_age = start_age
        self.end_age = end_age

    def forward(self, input, target):

        N = input.size()[0]
        target = target.type(torch.FloatTensor).cuda()
        m = nn.Softmax(dim=1)
        p = m(input)
        # mean loss
        a = torch.arange(self.start_age, self.end_age + 1, dtype=torch.float32).cuda()
        mean = torch.squeeze((p * a).sum(1, keepdim=True), dim=1)
        mse = (mean - target)**2
        mean_loss = mse.mean() / 2.0

        # variance loss
        b = (a[None, :] - mean[:, None])**2
        variance_loss = (p * b).sum(1, keepdim=True).mean()
        
        return self.lambda_1 * mean_loss, self.lambda_2 * variance_loss