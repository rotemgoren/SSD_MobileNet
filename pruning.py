#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sat Jun  9 15:45:16 2019

@author: viswanatha
"""

from utils import *
from datasets import PascalVOCDataset
from tqdm import tqdm
from pprint import PrettyPrinter

from PIL import Image, ImageDraw, ImageFont
import torch
import argparse
from mobilenet_ssd_priors import priors
import torch.nn.functional as F
from utils import detect_objects
import torch.nn as nn
from utils import save_checkpoint
from loss import MultiBoxLoss

import torch.nn.utils.prune as prune
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
priors_cxcy = priors
priors_cxcy = priors_cxcy.to(device)
n_classes = 4


# Good formatting when printing the APs for each class and mAP
pp = PrettyPrinter()

# Parameters
data_folder = 'dataset'
keep_difficult = True  # difficult ground truth objects must always be considered in mAP calculation, because these objects DO exist!
batch_size = 64
workers = 0
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
checkpoint = './model_prune_0.01_mag.pth.tar'#''./BEST_checkpoint_ssd300.pth.tar'

# Load model checkpoint that is to be evaluated
checkpoint = torch.load(checkpoint)
model = checkpoint['model']
model = model.to(device)

# Switch to eval mode
model.eval()

# Load test data
test_dataset = PascalVOCDataset(data_folder,
                                split='test',
                                keep_difficult=keep_difficult)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False,
                                          collate_fn=test_dataset.collate_fn, num_workers=workers, pin_memory=True)



def evaluate(test_loader, model):
    """
    Evaluate.
    :param test_loader: DataLoader for test data
    :param model: model
    """

    # Make sure it's in eval mode
    model.eval()

    # Lists to store detected and true boxes, labels, scores
    det_boxes = list()
    det_labels = list()
    det_scores = list()
    true_boxes = list()
    true_labels = list()
    true_difficulties = list()  # it is necessary to know which objects are 'difficult', see 'calculate_mAP' in utils.py

    with torch.no_grad():
        # Batches
        for i, (images, boxes, labels, difficulties) in enumerate(tqdm(test_loader, desc='Evaluating')):
            images = images.to(device)  # (N, 3, 300, 300)

            # Forward prop.
            predicted_locs, predicted_scores = model(images)

            # Detect objects in SSD output
            # Detect objects in SSD output
            det_boxes_batch, det_labels_batch, det_scores_batch = detect_objects(model, priors_cxcy, predicted_locs, predicted_scores,
                                                               min_score=0.2,
                                                               max_overlap=0.5, top_k=200,
                                                               n_classes=n_classes)

            # Evaluation MUST be at min_score=0.01, max_overlap=0.45, top_k=200 for fair comparision with the paper's results and other repos

            # Store this batch's results for mAP calculation
            boxes = [b.to(device) for b in boxes]
            labels = [l.to(device) for l in labels]
            difficulties = [d.to(device) for d in difficulties]

            det_boxes.extend(det_boxes_batch)
            det_labels.extend(det_labels_batch)
            det_scores.extend(det_scores_batch)
            true_boxes.extend(boxes)
            true_labels.extend(labels)
            true_difficulties.extend(difficulties)

        # Calculate mAP
        APs, mAP = calculate_mAP(det_boxes, det_labels, det_scores, true_boxes, true_labels, true_difficulties)

    # Print AP for each class
    pp.pprint(APs)

    print('\nMean Average Precision (mAP): %.3f' % mAP)
    return APs, mAP




class SynFlow(prune.BasePruningMethod):

    PRUNING_TYPE = 'unstructured'
    def __init__(self):
        self.parameters_to_prune=[]
        self.scores = {}
        self.mask=[]

    def masked_parameters(self,model):
        r"""Returns an iterator over models prunable parameters, yielding both the
        mask and parameter tensors.
        """
        print(model)
        for module in model.children():
            if isinstance(module, nn.Conv2d):
                self.parameters_to_prune.append((module, 'weight'))
            else:
                self.masked_parameters(module)

    def score(self,model, dataloader, device):

        @torch.no_grad()
        def linearize(model):
            # model.double()
            signs = {}
            for name, param in model.state_dict().items():
                signs[name] = torch.sign(param)
                param.abs_()
            return signs

        @torch.no_grad()
        def nonlinearize(model, signs):
            # model.float()
            for name, param in model.state_dict().items():
                param.mul_(signs[name])

        signs = linearize(model)

        data = next(iter(dataloader))[0]
        input_dim = list(data[0].shape)
        input = torch.ones([1] + input_dim, dtype=torch.float32).to(device)
        output = model(input)
        R=torch.sum(torch.cat(output, dim=2))
        R.backward()

        self.masked_parameters(model)
        self.parameters_to_prune = tuple(self.parameters_to_prune)

        for p,_ in self.parameters_to_prune:
            self.scores[id(p)] = torch.clone(p.weight.grad * p.weight).detach().abs_()
            p.weight.grad.data.zero_()

        nonlinearize(model, signs)

    def global_mask(self, sparsity):
        r"""Updates masks of model with scores by sparsity level globally.
        """
        # # Set score for masked parameters to -inf
        # for mask, param in self.masked_parameters:
        #     score = self.scores[id(param)]
        #     score[mask == 0.0] = -np.inf

        # Threshold scores
        global_scores = torch.cat([torch.flatten(v) for v in self.scores.values()])
        k = int((1.0 - sparsity) * global_scores.numel())
        if not k < 1:
            threshold, _ = torch.kthvalue(global_scores, k)
            for param,_ in self.parameters_to_prune:
                score = self.scores[id(param)]
                zero = torch.tensor([0.]).to(device)
                one = torch.tensor([1.]).to(device)
                self.mask=(torch.where(score <= threshold, zero, one))

    def compute_mask(self, t, default_mask): #t tensor to prune ,default_mask -mask from previous pruning iteration

        return self.mask



def foobar_unstructured(module, name):
    """Prunes tensor corresponding to parameter called `name` in `module`
    by removing every other entry in the tensors.
    Modifies module in place (and also return the modified module)
    by:
    1) adding a named buffer called `name+'_mask'` corresponding to the
    binary mask applied to the parameter `name` by the pruning method.
    The parameter `name` is replaced by its pruned version, while the
    original (unpruned) parameter is stored in a new parameter named
    `name+'_orig'`.

    Args:
        module (nn.Module): module containing the tensor to prune
        name (string): parameter name within `module` on which pruning
                will act.

    Returns:
        module (nn.Module): modified (i.e. pruned) version of the input
            module

    Examples:
        >>> m = nn.Linear(3, 4)
        >>> foobar_unstructured(m, name='bias')
    """
    FooBarPruningMethod.apply(module, name)
    return module


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Network Compression')
    # Training Hyperparameters
    training_args = parser.add_argument_group('training')
    training_args.add_argument('--dataset', type=str, default='cifar10',
                               choices=['mnist', 'cifar10', 'cifar100', 'tiny-imagenet', 'imagenet'],
                               help='dataset (default: mnist)')
    training_args.add_argument('--model-class', type=str, default='lottery',
                               choices=['default', 'lottery', 'tinyimagenet', 'imagenet'],
                               help='model class (default: default)')
    training_args.add_argument('--dense-classifier', type=bool, default=False,
                               help='ensure last layer of model is dense (default: False)')
    training_args.add_argument('--pretrained', type=bool, default=False,
                               help='load pretrained weights (default: False)')
    training_args.add_argument('--optimizer', type=str, default='adam', choices=['sgd', 'momentum', 'adam', 'rms'],
                               help='optimizer (default: adam)')
    training_args.add_argument('--train-batch-size', type=int, default=64,
                               help='input batch size for training (default: 64)')
    training_args.add_argument('--test-batch-size', type=int, default=256,
                               help='input batch size for testing (default: 256)')
    training_args.add_argument('--pre-epochs', type=int, default=10,
                               help='number of epochs to train before pruning (default: 0)')
    training_args.add_argument('--post-epochs', type=int, default=10,
                               help='number of epochs to train after pruning (default: 10)')
    training_args.add_argument('--lr', type=float, default=0.001,
                               help='learning rate (default: 0.001)')
    training_args.add_argument('--lr-drops', type=int, nargs='*', default=[],
                               help='list of learning rate drops (default: [])')
    training_args.add_argument('--lr-drop-rate', type=float, default=0.1,
                               help='multiplicative factor of learning rate drop (default: 0.1)')
    training_args.add_argument('--weight-decay', type=float, default=0.0,
                               help='weight decay (default: 0.0)')
    # Pruning Hyperparameters
    pruning_args = parser.add_argument_group('pruning')
    pruning_args.add_argument('--pruner', type=str, default='synflow',
                              choices=['rand', 'mag', 'snip', 'grasp', 'synflow'],
                              help='prune strategy (default: rand)')
    pruning_args.add_argument('--compression', type=float, default=3.0,
                              help='quotient of prunable non-zero prunable parameters before and after pruning (default: 1.0)')
    pruning_args.add_argument('--prune-epochs', type=int, default=5,
                              help='number of iterations for scoring (default: 1)')
    pruning_args.add_argument('--schedule', type=str, default='exponential',
                              choices=['linear', 'exponential'],
                              help='whether to use a linear or exponential compression schedule (default: exponential)')

    args = parser.parse_args()

    #APs, mAP=evaluate(test_loader, model)

    ## Prune ##
    print('Pruning with {} for {} epochs.'.format(args.pruner, args.prune_epochs))
    epochs =1
    model.eval()
    pruner = SynFlow()
    sparsity = 10 ** (-float(args.compression))
    for epoch in tqdm(range(epochs)):
        #pruner.score(model, test_loader,device)
        pruner.masked_parameters(model)

        total_sum = 0
        total_nelem = 0
        for param in pruner.parameters_to_prune:
            sparsity_ratio = float(torch.sum(param[0].weight == 0)) / float(param[0].weight.nelement())
            total_sum += float(torch.sum(param[0].weight == 0))
            total_nelem += float(param[0].weight.nelement())
            print(sparsity_ratio)
        print('total sparsity= {}'.format(total_sum / total_nelem))

        if args.schedule == 'exponential':
            sparse = sparsity**((epoch + 1) / epochs)
        elif args.schedule == 'linear':
            sparse = 1.0 - (1.0 - sparsity)*((epoch + 1) / epochs)

        # Invert scores

    prune.global_unstructured(
        pruner.parameters_to_prune,
        pruning_method=prune.L1Unstructured,
        amount=0.01,
    )
    total_sum=0
    total_nelem=0
    for param in pruner.parameters_to_prune:
        sparsity_ratio = float(torch.sum(param[0].weight==0)) / float(param[0].weight.nelement())
        total_sum+=float(torch.sum(param[0].weight==0))
        total_nelem+=float(param[0].weight.nelement())
        print(sparsity_ratio)
    print('total sparsity= {}'.format(total_sum/total_nelem))

    #save_checkpoint(epoch=0, epochs_since_improvement=0, model=model, optimizer=None, loss=1000, best_loss=1000, is_best=False, filename='model_prune_0.01_mag.pth.tar')
    # pruner = pruner(args.pruner)(
    #     generator.masked_parameters(model, args.prune_bias, args.prune_batchnorm, args.prune_residual), args)
    # sparsity = 10 ** (-float(args.compression))
    # prune.prune_loop(model, [],pruner, test_loader, device, sparsity,
    #            args.compression_schedule, args.mask_scope, args.prune_epochs, args.reinitialize, args.prune_train_mode,
    #            args.shuffle, args.invert)


