import sys
import os

curPath = os.path.abspath(os.path.dirname(__file__))
rootPath = os.path.split(curPath)[0]
sys.path.append(rootPath)


from configuration import config as cf
from util import util_metric
from train.model_operation import save_model, adjust_model
from train.visualization import dimension_reduction, penultimate_feature_visulization
from model import prot_bert
from util import data_loader_protBert

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import os
import sys
import time
import pickle
import seaborn as sns
import random

# os.environ["CUDA_VISIBLE_DEVICES"] = '0,1'

SEED = 3407
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic=True
torch.backends.cudnn.benchmark = False
def D(p, z, version='simplified'): # negative cosine similarity
    if version == 'original':
        z = z.detach() # stop gradient
        p = F.normalize(p, dim=1) # l2-normalize
        z = F.normalize(z, dim=1) # l2-normalize
        return -(p*z).sum(dim=1).mean()

    elif version == 'simplified':# same thing, much faster. Scroll down, speed test in __main__
        return 1 - F.cosine_similarity(p, z, dim=-1)
    else:
        raise Exception
class ContrastiveLoss(torch.nn.Module):
    def __init__(self, margin=2.0):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin

    def forward(self, output1, output2, label):
        # euclidean_distance: [128]
        # euclidean_distance = F.pairwise_distance(output1, output2)
        cos_distance = D(output1, output2)
        # print("ED",euclidean_distance)
        loss_contrastive = torch.mean((1 - label) * torch.pow(cos_distance, 2) +  # calmp夹断用法
                                      (label) * torch.pow(torch.clamp(self.margin - cos_distance, min=0.0), 3))

        return loss_contrastive

def load_data(config):
    train_iter_orgin, test_iter = data_loader_protBert.load_data(config)
    # print('-' * 20, 'data construction over', '-' * 20)
    return train_iter_orgin, test_iter


def cal_loss_dist_by_cosine(model):
    embedding = model.embedding
    loss_dist = 0

    vocab_size = embedding[0].tok_embed.weight.shape[0]
    d_model = embedding[0].tok_embed.weight.shape[1]

    Z_norm = vocab_size * (len(embedding) ** 2 - len(embedding)) / 2

    for i in range(len(embedding)):
        for j in range(len(embedding)):
            if i < j:
                cosin_similarity = torch.cosine_similarity(embedding[i].tok_embed.weight, embedding[j].tok_embed.weight)
                loss_dist -= torch.sum(cosin_similarity)
                # print('cosin_similarity.shape', cosin_similarity.shape)
    loss_dist = loss_dist / Z_norm
    return loss_dist


def get_loss(logits, label, criterion):
    loss = criterion(logits, label)
    loss = loss.float()
    # flooding method
    loss = (loss - config.b).abs() + config.b

    return loss

def get_val_loss(logits, label, criterion):
    loss = criterion(logits.view(-1, config.num_class), label.view(-1))
    loss = (loss.float()).mean()
    # flooding method
    loss = (loss - config.b).abs() + config.b
    Q_sum = len(logits)
    logits = F.softmax(logits, dim=1)  # softmax归一化
    hat_sum_p0 = logits[:, 0].sum()/Q_sum  # 负类的概率和
    hat_sum_p1 = logits[:, 1].sum()/Q_sum  # 正类的概率和
    mul_hat_p0 = hat_sum_p0.mul(torch.log(hat_sum_p0))
    mul_hat_p1 = hat_sum_p1.mul(torch.log(hat_sum_p1))
    mul_p0 = logits[:, 0].mul(torch.log(logits[:, 0])).sum()/Q_sum
    mul_p1 = logits[:, 1].mul(torch.log(logits[:, 1])).sum()/Q_sum
    # sum_loss = loss+(-1)*(mul_hat_p0+mul_hat_p1) + 0.1*(mul_p0+mul_p1)
    sum_loss = loss+(mul_hat_p0+mul_hat_p1)-0.1*(mul_p0+mul_p1)
    return sum_loss

def periodic_test(test_iter, model, criterion, config, sum_epoch):
    print('#' * 60 + 'Periodic Test' + '#' * 60)
    test_metric, test_loss, test_repres_list, test_label_list, \
    test_roc_data, test_prc_data = model_eval(test_iter, model, criterion, config)

    print('test current performance')
    # print('[ACC,\t\tPrecision,\t\tSensitivity,\t\tSpecificity,\t\tF1,\t\tAUC,\t\tMCC,\t\tTP,\t\tFP,\t\tTN,\t\tFN]')
    print('[ACC,\t\tPrecision,\t\tSensitivity,\tSpecificity,\t\tF1,\t\tAUC,\t\t\tMCC,\t\t TP,    \t\tFP,\t\t\tTN, \t\t\tFN]')
    # print(test_metric.numpy())
    plmt = test_metric.numpy()
    print('%.5g\t\t' % plmt[0], '%.5g\t\t' % plmt[1], '%.5g\t\t' % plmt[2], '%.5g\t\t' % plmt[3], '%.5g\t' % plmt[4],
          '%.5g\t\t' % plmt[5], '%.5g\t\t' % plmt[6], '%.5g\t\t' % plmt[7], '  %.5g\t\t' % plmt[8], '  %.5g\t\t' % plmt[9], ' %.5g\t\t' % plmt[10])
    print('#' * 60 + 'Over' + '#' * 60)

    step_test_interval.append(sum_epoch)
    test_acc_record.append(test_metric[0])
    test_loss_record.append(test_loss)

    return test_metric, test_loss, test_repres_list, test_label_list


def periodic_valid(valid_iter, model, criterion, config, sum_epoch):
    print('#' * 60 + 'Periodic Validation' + '#' * 60)

    valid_metric, valid_loss, valid_repres_list, valid_label_list, \
    valid_roc_data, valid_prc_data = model_eval(valid_iter, model, criterion, config)

    print('validation current performance')
    print('[ACC,\tPrecision,\tSensitivity,\tSpecificity,\tF1,\tAUC,\tMCC]')
    print(valid_metric.numpy())
    print('#' * 60 + 'Over' + '#' * 60)

    step_valid_interval.append(sum_epoch)
    valid_acc_record.append(valid_metric[0])
    valid_loss_record.append(valid_loss)

    return valid_metric, valid_loss, valid_repres_list, valid_label_list


def train_model(train_iter, valid_iter, test_iter, model, optimizer, criterion, contras_criterion, config, iter_k):
    best_acc = 0
    best_performance = 0
    train_batch_loss = 0
    for epoch in range(1, config.epoch + 1):
        steps = 0
        train_epoch_loss = 0
        train_correct_num = 0
        train_total_num = 0
        current_batch_size = 0
        repres_list = []
        label_list = []
        label_b = []
        output_b = []
        logits_b = []
        model.train()
        random.shuffle(train_iter)
        for batch in train_iter:
            input, label = batch
            label = torch.tensor(label, dtype=torch.long).cuda()
            output = model.forward(input)
            logits = model.get_logits(input)
            # repres_list.extend(output.cpu().detach().numpy())
            # label_list.extend(label.cpu().detach().numpy())
            output = output.view(-1, output.size(-1))
            logits = logits.view(-1, logits.size(-1))
            label = label[1:-1]
            logits = logits[1:-1]
            output = output[1:-1]
            output_b.append(output)
            logits_b.append(logits)
            label_b.append(label)

            current_batch_size += 1
            if current_batch_size % config.batch_size == 0:
                output_b = torch.cat(output_b, dim=0)
                logits_b = torch.cat(logits_b, dim=0)
                label_b = torch.cat(label_b, dim=0)
                label_b = label_b.view(-1)
                logits_b = logits_b.view(-1, logits_b.size(-1))
                output_b = output_b.view(-1, output_b.size(-1))
                #contrastive loss
                label_ls = []
                # weight_ls = []
                contras_len = len(output_b) // 2
                label1 = label_b[:contras_len]
                label2 = label_b[contras_len:contras_len*2]
                for i in range(contras_len):
                    xor_label = (label1[i] ^ label2[i])
                    label_ls.append(xor_label.unsqueeze(0))
                contras_label = torch.cat(label_ls)
                # contras_weight = torch.cat(weight_ls)
                output1 = output_b[:contras_len]
                output2 = output_b[contras_len:contras_len*2]
                contras_loss = contras_criterion(output1, output2, contras_label)

                # ce_loss = get_loss(logits, label, criterion)
                ce_loss = criterion(logits_b, label_b)
                loss = ce_loss + contras_loss
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                steps = steps + 1
                train_batch_loss = loss.item()
                train_epoch_loss += train_batch_loss

                corre = (torch.max(logits_b, 1)[1] == label_b).int()
                corrects = corre.sum()
                train_correct_num += corrects
                the_batch_size = label_b.size(0)
                train_total_num += the_batch_size
                train_acc = 100.0 * corrects / the_batch_size

                label_b = []
                output_b = []
                logits_b = []

                '''Periodic Train Log'''
                if steps % config.interval_log == 0:
                    sys.stdout.write(
                        '\rEpoch[{}] Batch[{}] - loss: {:.6f} | ACC: {:.4f}%({}/{})'.format(epoch, steps,
                                                                                            train_batch_loss,
                                                                                            train_acc,
                                                                                            corrects,
                                                                                            the_batch_size))
                    print()

                    step_log_interval.append(steps)
                    train_acc_record.append(train_acc)
                    train_loss_record.append(train_batch_loss)

        sum_epoch = iter_k * config.epoch + epoch
        print(f"Train - Epoch[{epoch}] - loss: {train_epoch_loss/(len(train_iter)//config.batch_size)} | ACC: {(train_correct_num/train_total_num)*100:.4f}%({train_correct_num}/{train_total_num})")

        '''Periodic Validation'''
        if valid_iter and sum_epoch % config.interval_valid == 0:
            valid_metric, valid_loss, valid_repres_list, valid_label_list = periodic_valid(valid_iter,
                                                                                           model,
                                                                                           criterion,
                                                                                           config,
                                                                                           sum_epoch)
            valid_acc = valid_metric[0]
            if valid_acc > best_acc:
                best_acc = valid_acc
                best_performance = valid_metric

        '''Periodic Test'''
        if test_iter and sum_epoch % config.interval_test == 0:
            time_test_start = time.time()

            test_metric, test_loss, test_repres_list, test_label_list = periodic_test(test_iter,
                                                                                      model,
                                                                                      criterion,
                                                                                      config,
                                                                                      sum_epoch)
            '''Periodic Save'''
            # save the model if specific conditions are met
            test_acc = test_metric[5]
            if test_acc > best_acc:
                best_acc = test_acc
                best_performance = test_metric
            repres_list.extend(test_repres_list)
            label_list.extend(test_label_list)

            '''feature dimension reduction'''
            if sum_epoch % 1 == 0 or epoch == 1:
                 dimension_reduction(repres_list, label_list, epoch)

            '''reduction feature visualization'''

    return best_performance


def model_eval(data_iter, model, criterion, config):
    device = torch.device("cuda" if config.cuda else "cpu")
    label_pred = torch.empty([0], device=device)
    label_real = torch.empty([0], device=device)
    pred_prob = torch.empty([0], device=device)

    print('model_eval data_iter', len(data_iter))

    iter_size, corrects, avg_loss = 0, 0, 0
    repres_list = []
    label_list = []

    model.eval()
    with torch.no_grad():
        # random.shuffle(data_iter)
        for batch in data_iter:
            input, label = batch
            # input = input.cuda()
            label = torch.tensor(label, dtype=torch.long).cuda()
            lll = label.clone()
            label = torch.unsqueeze(label, 0)
            logits = model.get_logits(input)
            output = model.forward(input)
            # logits = torch.unsqueeze(logits[:, :2], 0)

            repres_list.extend(output.cpu().detach().numpy())
            label_list.extend(lll.cpu().detach().numpy())

            # loss = criterion(logits.view(-1, config.num_class), label.view(-1))
            logits = logits.view(-1, logits.size(-1))
            label = label.view(-1)
            label = label[1:-1]
            logits = logits[1:-1]
            loss = criterion(logits, label)
            # loss = (loss.float()).mean()
            avg_loss += loss.item()

            logits = torch.unsqueeze(logits, 0)
            label = torch.unsqueeze(label, 0)
            pred_prob_all = F.softmax(logits, dim=2)
            # Prediction probability [batch_size, seq_len, class_num]
            pred_prob_positive = pred_prob_all[:, :, 1]
            positive = torch.empty([0], device=device)
            # Probability of predicting positive classes [batch_size, seq_len]
            pred_prob_sort = torch.max(pred_prob_all, 2)
            # The maximum probability of prediction in each sample [batch_size]
            pred_class = pred_prob_sort[1]
            p_class = torch.empty([0], device=device)
            la = torch.empty([0], device=device)
            positive = torch.cat([positive, pred_prob_positive[0][:]])
            p_class = torch.cat([p_class, pred_class[0][:]])
            la = torch.cat([la, label[0][:]])
                # for j in range(1, pro_len + 1):
                #     m[i][j] = 1

            corre = (pred_class == label).int()
            corrects += corre.sum()
            iter_size += label.size(1)
            label_pred = torch.cat([label_pred, p_class.float()])
            label_real = torch.cat([label_real, la.float()])
            pred_prob = torch.cat([pred_prob, positive])


    metric, roc_data, prc_data = util_metric.caculate_metric(label_pred, label_real, pred_prob)
    avg_loss /= len(data_iter)
    # accuracy = 100.0 * corrects / iter_size
    accuracy = metric[0]
    print('Evaluation - loss: {:.6f}  ACC: {:.4f}%({}/{})'.format(avg_loss,
                                                                  100*accuracy,
                                                                  corrects,
                                                                  iter_size))

    return metric, avg_loss, repres_list, label_list, roc_data, prc_data


def train_test(train_iter, test_iter, config):
    # 加载
    model = prot_bert.BERT(config)
    if config.cuda:
        model.cuda()
        
    optimizer = torch.optim.AdamW(params=model.parameters(), lr=config.lr, weight_decay=config.reg)
    contras_criterion = ContrastiveLoss()
    criterion = nn.CrossEntropyLoss(weight=torch.FloatTensor([1, 17])).to(config.device)  # weighted update (1:17)
    print('=' * 50 + 'Start Training' + '=' * 50)
    best_performance = train_model(train_iter, None, test_iter, model, optimizer, criterion, contras_criterion, config, 0)
    print('=' * 50 + 'Train Finished' + '=' * 50)

    print('*' * 60 + 'The Last Test' + '*' * 60)
    last_test_metric, last_test_loss, last_test_repres_list, last_test_label_list, \
    last_test_roc_data, last_test_prc_data = model_eval(test_iter, model, criterion, config)
    print('[ACC,\t\tPrecision,\t\tSensitivity,\t\tSpecificity,\t\tF1,\t\tAUC,\t\tMCC,\t\tTP,\t\tFP,\t\tTN,\t\tFN]')

    lmt = last_test_metric.numpy()
    print('%.5g\t\t' % lmt[0] , '%.5g\t\t' % lmt[1], '%.5g\t\t' % lmt[2], '%.5g\t\t' % lmt[3], '%.5g\t' % lmt[4], '%.5g\t\t' % lmt[5], '%.5g\t\t' % lmt[6],
          '%.5g\t\t' % lmt[7], '  %.5g\t\t' % lmt[8], '  %.5g\t\t' % lmt[9], ' %.5g\t\t' % lmt[10])
    print('*' * 60 + 'The Last Test Over' + '*' * 60)

    return model, best_performance, last_test_metric


def select_dataset():
    path_train_data = '/data2024/yyyl/code/PepBCL-master/data/Dataset1_train.tsv'
    path_test_data = '/data2024/yyyl/code/PepBCL-master/data/Dataset1_test.tsv'
    return path_train_data, path_test_data


def load_config():

    '''Set the required variables in the configuration'''
    train_name = 'PepBCL'
    path_config_data = None
    path_train_data, path_test_data = select_dataset()

    '''Get configuration'''
    if path_config_data is None:
        config = cf.get_train_config()
    else:
        config = pickle.load(open(path_config_data, 'rb'))

    '''Modify default configuration'''
    # config.epoch = 50

    '''Set other variables'''
    # flooding method
    b = 0.06

    config.if_multi_scaled = False

    '''initialize result folder'''
    # result_folder = '../result' + config.learn_name
    result_folder = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../result', config.learn_name)
    if not os.path.exists(result_folder):
        os.makedirs(result_folder)

    '''Save all variables in configuration'''
    config.train_name = train_name
    config.path_train_data = path_train_data
    config.path_test_data = path_test_data

    config.b = b
    config.result_folder = result_folder

    return config


if __name__ == '__main__':
    np.set_printoptions(linewidth=400, precision=4)
    time_start = time.time()

    '''load configuration'''
    config = load_config()

    '''set device'''
    torch.cuda.set_device(config.device)

    '''load data'''
    train_iter, test_iter = load_data(config)
    print('=' * 20, 'load data over', '=' * 20)

    '''draw preparation'''
    step_log_interval = []
    train_acc_record = []
    train_loss_record = []
    step_valid_interval = []
    valid_acc_record = []
    valid_loss_record = []
    step_test_interval = []
    test_acc_record = []
    test_loss_record = []

    '''train procedure'''
    valid_performance = 0
    best_performance = 0
    last_test_metric = 0

    if config.k_fold == -1:
        # train and test
        model, best_performance, last_test_metric = train_test(train_iter, test_iter, config)
        # 保存
        # torch.save(model, 'bert_finetuned_model.pkl')
        # torch.save(model.state_dict(), 'bert_finetuned_model.pkl')
        pass

    
    '''report result'''
    valid_performance_list = []
    print('*=' * 50 + 'Result Report' + '*=' * 50)
    if config.k_fold != -1:
        print('valid_performance_list', valid_performance_list)
        tensor_list = [x.view(1, -1) for x in valid_performance_list]
        cat_tensor = torch.cat(tensor_list, dim=0)
        metric_mean = torch.mean(cat_tensor, dim=0)

        print('valid mean performance')
        print('[ACC,\t\tPrecision,\t\tSensitivity,\t\tSpecificity,\t\tF1,\t\tAUC,\t\tMCC,\t\tTP,\t\tFP,\t\tTN,\t\tFN]')
        lmt = metric_mean.numpy()
        print('%.5g\t\t' % lmt[0], '%.5g\t\t' % lmt[1], '%.5g\t\t' % lmt[2], '%.5g\t\t' % lmt[3], '%.5g\t' % lmt[4],
              '%.5g\t\t' % lmt[5], '%.5g\t\t' % lmt[6],
              '%.5g\t\t' % lmt[7], '  %.5g\t\t' % lmt[8], '  %.5g\t\t' % lmt[9], ' %.5g\t\t' % lmt[10])

        print('valid_performance list')
        print('\t[ACC,\tPrecision,\tSensitivity,\tSpecificity,\tF1,\tAUC,\tMCC]')
        for tensor_metric in valid_performance_list:
            print('\t{}'.format(tensor_metric.numpy()))
    else:
        print('last test performance')
        print('[ACC,\t\tPrecision,\t\tSensitivity,\t\tSpecificity,\t\tF1,\t\tAUC,\t\tMCC,\t\tTP,\t\tFP,\t\tTN,\t\tFN]')
        lmt = last_test_metric.numpy()
        print('%.5g\t\t' % lmt[0], '%.5g\t\t' % lmt[1], '%.5g\t\t' % lmt[2], '%.5g\t\t' % lmt[3], '%.5g\t' % lmt[4],
              '%.5g\t\t' % lmt[5], '%.5g\t\t' % lmt[6],
              '%.5g\t\t' % lmt[7], '  %.5g\t\t' % lmt[8], '  %.5g\t\t' % lmt[9], ' %.5g\t\t' % lmt[10])
        print()
        print('best_performance')
        print('[ACC,\t\tPrecision,\t\tSensitivity,\t\tSpecificity,\t\tF1,\t\tAUC,\t\tMCC,\t\tTP,\t\tFP,\t\tTN,\t\tFN]')
        lmt = best_performance.numpy()
        print('%.5g\t\t' % lmt[0], '%.5g\t\t' % lmt[1], '%.5g\t\t' % lmt[2], '%.5g\t\t' % lmt[3], '%.5g\t' % lmt[4],
              '%.5g\t\t' % lmt[5], '%.5g\t\t' % lmt[6],
              '%.5g\t\t' % lmt[7], '  %.5g\t\t' % lmt[8], '  %.5g\t\t' % lmt[9], ' %.5g\t\t' % lmt[10])

    print('*=' * 50 + 'Report Over' + '*=' * 50)

    '''save train result'''
    # save the model if specific conditions are met
    if config.k_fold == -1:
        best_acc = best_performance[0]
        last_test_acc = last_test_metric[0]
        if last_test_acc > best_acc:
            best_acc = last_test_acc
            best_performance = last_test_metric
        if config.save_best and best_acc >= config.threshold:
            save_model(model.state_dict(), best_acc, config.result_folder, config.learn_name)

    # save the model configuration
    with open(config.result_folder + '/config.pkl', 'wb') as file:
        pickle.dump(config, file)
    print('-' * 50, 'Config Save Over', '-' * 50)

    time_end = time.time()
    print('total time cost', time_end - time_start, 'seconds')
