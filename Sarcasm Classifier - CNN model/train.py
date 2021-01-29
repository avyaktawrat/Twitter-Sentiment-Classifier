import os
import sys
import torch
import torch.autograd as autograd
import torch.nn.functional as F

# device = torch.device('cpu')

def train(train_iter, dev_iter, model, args):
    if args.cuda:
        model.cuda()

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.max_norm)

    steps = 0
    best_acc = 0
    tr_acc = 0
    last_step = 0
    for epoch in range(1, args.epochs+1):
        for batch in train_iter:
            model.train()
            feature, target, users = batch.text, batch.label, batch.user
            # print(users.size())
            feature.t_(), users.t_(), target.sub_(1)  # batch first, index align
            # print(feature.size())
            if args.cuda:
                feature, target, users = feature.cuda(), target.cuda(), users.cuda()

            optimizer.zero_grad()
            # users = 0
            logit = model(feature, users)
            loss = F.cross_entropy(logit, target)
            loss.backward()
            optimizer.step()

            steps += 1
            if steps % args.log_interval == 0:
                corrects = (torch.max(logit, 1)[1].view(target.size()).data == target.data).sum()
                accuracy = 100.0 * corrects/batch.batch_size
                if not (args.param_search):
                    sys.stdout.write(
                        '\rBatch[{}] - loss: {:.6f}  acc: {:.4f}%({}/{})'.format(steps, 
                                                                                loss.item(), 
                                                                                accuracy.item(),
                                                                                corrects.item(),
                                                                                batch.batch_size))
            if steps % args.test_interval == 0:
                dev_acc = eval(dev_iter, model, args)
                if dev_acc > best_acc:
                    best_acc = dev_acc
                    last_step = steps
                    tr_acc = accuracy
                    if args.save_best:
                        save(model, args.save_dir, 'best', steps)
                else:
                    if steps - last_step >= args.early_stop and (not args.param_search):
                        print('early stop by {} steps.'.format(args.early_stop))
            elif steps % args.save_interval == 0:
                save(model, args.save_dir, 'snapshot', steps)
    return tr_acc, best_acc, last_step


def eval(data_iter, model, args):
    model.eval()
    corrects, avg_loss = 0, 0
    for batch in data_iter:
        feature, target, users = batch.text, batch.label, batch.user
        feature.t_(), users.t_(), target.sub_(1)  # batch first, index align
        if args.cuda:
            feature, target, users = feature.cuda(), target.cuda(), users.cuda()

        logit = model(feature, users)
        loss = F.cross_entropy(logit, target, size_average=False)

        avg_loss += loss.item()
        corrects += (torch.max(logit, 1)
                     [1].view(target.size()).data == target.data).sum()

    size = len(data_iter.dataset)
    avg_loss /= size
    accuracy = 100.0 * corrects/size
    if not args.param_search:
        print('\nEvaluation - loss: {:.6f}  acc: {:.4f}%({}/{}) \n'.format(avg_loss, 
                                                                          accuracy, 
                                                                          corrects, 
                                                                          size))
    return accuracy


def predict(text, model, text_field, label_feild, cuda_flag):
    assert isinstance(text, str)
    model.eval()
    # text = text_field.tokenize(text)
    text = text_field.preprocess(text)
    text = [[text_field.vocab.stoi[x] for x in text]]
    x = torch.tensor(text)
    x = autograd.Variable(x)
    if cuda_flag:
        x = x.cuda()
    print(x)
    output = model(x)
    _, predicted = torch.max(output, 1)
    return label_feild.vocab.itos[predicted.item()+1]


def save(model, save_dir, save_prefix, steps):
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)
    save_prefix = os.path.join(save_dir, save_prefix)
    save_path = '{}_steps_{}.pt'.format(save_prefix, steps)
    torch.save(model.state_dict(), save_path)
