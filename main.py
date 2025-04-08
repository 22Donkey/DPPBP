import argparse, os, torch
import random
import numpy as np
from utils.data import build_data
from trainer.trainer import Trainer
# from trainer.tester import Tester
from models.setpred4RE import SetPred4RE

import os  
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"

# 将表示布尔值的字符串转换为Python中的True或False
def str2bool(v):
    return v.lower() in ('true')


def add_argument_group(name):
    arg = parser.add_argument_group(name)
    return arg


def get_args():
    args, unparsed = parser.parse_known_args()
    if len(unparsed) > 1:
        print("Unparsed args: {}".format(unparsed))
    return args, unparsed


def set_seed(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    data_arg = add_argument_group('Data')  
    
    data_arg.add_argument('--dataset_name', type=str, default="protein")
    data_arg.add_argument('--train_file', type=str, default="./data/train2.json")
    data_arg.add_argument('--valid_file', type=str, default="./data/test2.json")
    # data_arg.add_argument('--test_file', type=str, default="./data/test1.json")

    data_arg.add_argument('--generated_data_directory', type=str, default="./data/data2/")
    data_arg.add_argument('--generated_param_directory', type=str, default="./data/data2/ex1/")
    data_arg.add_argument('--bert_directory', type=str, default="/data/yyl/model/prot_bert/")
    data_arg.add_argument('--log_dir', type=str, default="./log/")
    data_arg.add_argument("--partial", type=str2bool, default=False)
    learn_arg = add_argument_group('Learning')
    learn_arg.add_argument('--model_name', type=str, default="PBSD")
    learn_arg.add_argument('--num_generated_triples', type=int, default=40)
    learn_arg.add_argument('--num_decoder_layers', type=int, default=10) # 4
    learn_arg.add_argument('--matcher', type=str, default="avg", choices=['avg', 'min'])
    learn_arg.add_argument('--na_rel_coef', type=float, default=1)
    learn_arg.add_argument('--rel_loss_weight', type=float, default=1)
    learn_arg.add_argument('--head_ent_loss_weight', type=float, default=2)
    learn_arg.add_argument('--fix_bert_embeddings', type=str2bool, default=True)
    learn_arg.add_argument('--batch_size', type=int, default=2)               
    learn_arg.add_argument('--max_epoch', type=int, default=100)
    learn_arg.add_argument('--gradient_accumulation_steps', type=int, default=1)
    learn_arg.add_argument('--decoder_lr', type=float, default=2e-5)
    learn_arg.add_argument('--encoder_lr', type=float, default=1e-5)
    learn_arg.add_argument('--lr_decay', type=float, default=0.01)
    learn_arg.add_argument('--weight_decay', type=float, default=1e-5)
    learn_arg.add_argument('--max_grad_norm', type=float, default=1)
    learn_arg.add_argument('--optimizer', type=str, default='AdamW', choices=['Adam', 'AdamW'])
    evaluation_arg = add_argument_group('Evaluation')
    evaluation_arg.add_argument('--n_best_size', type=int, default=250)
    evaluation_arg.add_argument('--max_span_length', type=int, default=15) 
    misc_arg = add_argument_group('MISC')
    misc_arg.add_argument('--refresh', type=str2bool, default=False)
    misc_arg.add_argument('--use_gpu', type=str2bool, default=True)
    misc_arg.add_argument('--visible_gpu', type=int, default=1)
    misc_arg.add_argument('--random_seed', type=int, default=1)




    args, unparsed = get_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = "2"
    for arg in vars(args):
        print(arg, ":",  getattr(args, arg))
    set_seed(args.random_seed)
    data = build_data(args)
    model = SetPred(args, data.relational_alphabet.size())
    trainer = Trainer(model, data, args)
    trainer.train_model()
    # tester = Tester(model, data, args)
    # tester.load_model("/data2024/yyyl/code/ProtBert/data/generated_data/ex8/Set-Prediction-Networks_protein_epoch_24_f1_0.1603.pth")  # 替换为最佳模型权重的路径
    # results = tester.test_model()
    # print("Test Results:", results)