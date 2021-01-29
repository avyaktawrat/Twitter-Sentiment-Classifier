import os
import argparse
import datetime
import torch
import torchtext.data as data
import torchtext.datasets as datasets
import model
import train
import mydatasets
import pandas as pd
import random

from itertools import product
# import numpy as np

# python -W ignore foo.py
random.seed(42)
p = {
      # 'lr':torch.linspace(1e-5,1e-3,10), 
      'lr': [0.001],
      'dropout':torch.linspace(0.3,0.5,5), 
      # 'dropout':[0.5],
      'max_norm':torch.logspace(-3,-1,50),
      # 'max_norm': [0.001],
      'kernel_num':torch.linspace(40,100,61),
      # 'kernel_num': [50],
      'kernel_size':['2,3,4','5,6,7','8,9,10','11,12,13','2,4,6','3,5,7']
      # 'kernel_size':['2,3,4']

      }
best_score = {'Train Accuracy %': 0, 'Validation Accuracy %': 0, 'Steps': 0}
# function product takes cartesian product
# list of dicts
p_list = list((dict(zip(p.keys(), values)) for values in product(*p.values())))
random.shuffle(p_list)

for i_dict in p_list[:10]:
    parser = argparse.ArgumentParser(description='CNN hyper parameter tuning')
    # learning
    parser.add_argument('-lr', type=float, default=float(i_dict['lr']), help='initial learning rate [default: 0.001]')
    parser.add_argument('-epochs', type=int, default=32, help='number of epochs for train [default: 256]')
    parser.add_argument('-batch-size', type=int, default=200, help='batch size for training [default: 64]')
    parser.add_argument('-log-interval',  type=int, default=1,   help='how many steps to wait before logging training status [default: 1]')
    parser.add_argument('-test-interval', type=int, default=100, help='how many steps to wait before testing [default: 100]')
    parser.add_argument('-save-interval', type=int, default=500, help='how many steps to wait before saving [default:500]')
    parser.add_argument('-save-dir', type=str, default='snapshot', help='where to save the snapshot')
    parser.add_argument('-early-stop', type=int, default=100, help='iteration numbers to stop without performance increasing')
    parser.add_argument('-save-best', type=bool, default=False, help='whether to save when get best performance')
    # data 
    parser.add_argument('-shuffle', action='store_true', default=False, help='shuffle the data every epoch')
    # model
    parser.add_argument('-dropout', type=float, default=float(i_dict['dropout']), help='the probability for dropout [default: 0.5]')
    parser.add_argument('-max-norm', type=float, default=float(i_dict['max_norm']), help='l2 constraint of parameters [default: 3.0]')
    parser.add_argument('-embed-dim', type=int, default=64, help='number of embedding dimension [default: 128]')
    parser.add_argument('-kernel-num', type=int, default=int(i_dict['kernel_num']), help='number of each kind of kernel')
    parser.add_argument('-kernel-sizes', type=str, default=i_dict['kernel_size'], help='comma-separated kernel size to use for convolution')
    parser.add_argument('-static', action='store_true', default=False, help='fix the embedding')
    parser.add_argument('-pretrained-embed-words', type=bool, default=False, help='Use pre-trained embedding for words')
    parser.add_argument('-pretrained-embed-users', type=bool, default=False, help='Use pre-trained embedding for users')
    # device
    parser.add_argument('-device', type=int, default=-1, help='device to use for iterate data, -1 mean cpu [default: -1]')
    parser.add_argument('-no-cuda', action='store_true', default=False, help='disable the gpu')
    # option
    parser.add_argument('-snapshot', type=str, default=None, help='filename of model snapshot [default: None]')
    parser.add_argument('-predict', type=str, default=None, help='predict the sentence given')
    parser.add_argument('-test', action='store_true', default=False, help='train or test')
    # hyper param learning, if true, outputs only best test acc
    parser.add_argument('-param-search', type=bool, default=True, help='Tuning hyper parameters')
    args = parser.parse_args()

    # load MR dataset
    def mr(text_field, label_field, user_field, **kargs):
        train_data, dev_data = mydatasets.MR.splits(text_field, label_field, user_field, args = args)
        if args.pretrained_embed_words:
            text_field.build_vocab(train_data, dev_data, vectors = args.custom_embed)
        else:
            text_field.build_vocab(train_data, dev_data)

        label_field.build_vocab(train_data, dev_data)
        user_field.build_vocab(train_data, dev_data)
        train_iter, dev_iter = data.Iterator.splits(
                                    (train_data, dev_data), 
                                    batch_sizes=(args.batch_size, len(dev_data)),
                                    **kargs)
        return train_iter, dev_iter


    # load data
    # print("\nLoading data...")
    text_field = data.Field(lower=True)
    label_field = data.Field(sequential=False)
    user_field = data.Field()
    train_iter, dev_iter = mr(text_field, label_field, user_field, device='cpu', repeat=False)
    
    # update args and print
    args.vectors = text_field.vocab.vectors
    args.embed_num = len(text_field.vocab)
    args.class_num = len(label_field.vocab) - 1
    args.embed_num_users = len(user_field.vocab)
    args.cuda = (not args.no_cuda) and torch.cuda.is_available(); del args.no_cuda
    args.kernel_sizes = [int(k) for k in args.kernel_sizes.split(',')]
    args.save_dir = os.path.join(args.save_dir, datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S'))

    # print("\nParameters:")
    # for attr, value in sorted(args.__dict__.items()):
    #     print("\t{}={}".format(attr.upper(), value))

    # model
    cnn = model.CNN_Text(args)
    if args.snapshot is not None:
        print('\nLoading model from {}...'.format(args.snapshot))
        cnn.load_state_dict(torch.load(args.snapshot))

    if args.cuda:
        torch.cuda.set_device(args.device)
        cnn = cnn.cuda()

    # train or predict
    if args.predict is not None:
        label = train.predict(args.predict, cnn, text_field, label_field, user_field, args.cuda)
        print('\n[Text]  {}\n[Label] {}\n'.format(args.predict, label))
    elif args.test:
        try:
            train.eval(test_iter, cnn, args) 
        except Exception as e:
            print("\nSorry. The test dataset doesn't  exist.\n")
    else:
        print()
        try:
            tr_acc, dev_acc, last_step = train.train(train_iter, dev_iter, cnn, args)
        except KeyboardInterrupt:
            print('\n' + '-' * 89)
            print('Exiting from training early')
    
    if args.param_search:
        print("\n Performance of the Model with Params defined below",'\n',
        "Train Accuracy: {:.2f} % \n Validation Accuracy: {:.2f}% \n Steps: {:.0f}".format(tr_acc,dev_acc,last_step))
        print("\n Hyper Parameters:")
        for attr, value in sorted(i_dict.items()):
            print("\t{} = {}".format(attr.upper(), value))

        print('\n ############################################################ \n')
        if dev_acc > best_score['Validation Accuracy %']:
            best_score['Train Accuracy %'] = tr_acc
            best_score['Validation Accuracy %'] = dev_acc
            best_score['Steps'] = last_step
            p_best = i_dict.copy()
    
# print("")
print("Best set of hyper parameters are \n:")
for attr, value in sorted(p_best.items()):
        print("\t{} = {}".format(attr.upper(), value))
print('\n\n',"With Performance of \n")
for attr, value in sorted(best_score.items()):
        print("\t{} = {:.2f}".format(attr, float(value)))