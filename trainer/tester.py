import torch
from torch import nn
from tqdm import tqdm
from transformers import AdamW
from utils.average_meter import AverageMeter
from utils.functions import formulate_gold
from utils.metric import metric

class Pred_Triple:
    def __init__(self, pred_rel, rel_prob, head_start_index, head_end_index):
        self.pred_rel = pred_rel
        self.rel_prob = rel_prob
        self.head_start_index = head_start_index
        self.head_end_index = head_end_index

class Tester:
    def __init__(self, model, data, args):
        self.model = model
        self.data = data
        self.args = args

        if args.use_gpu:
            self.device = torch.device('cuda')
        else:
            self.device = torch.device('cpu')
        self.model.to(self.device)  # 将模型放到相应设备

    def load_model(self, model_path):
        state_dict = torch.load(model_path)
        self.model.load_state_dict(state_dict['state_dict'])
        self.model.eval()  # 设置为评估模式

    def test_model(self):
        prediction, gold = {}, {}
        output = []  # 存储数字串输出

        test_loader = self.data.test_loader  # 使用 data 属性获取测试集

        with torch.no_grad():
            batch_size = self.args.batch_size
            test_num = len(test_loader)
            total_batch = test_num // batch_size + 1
            
            for batch_id in range(total_batch):
                start = batch_id * batch_size
                end = (batch_id + 1) * batch_size
                if end > test_num:
                    end = test_num
                test_instance = test_loader[start:end]  
                if not test_instance:
                    continue
                
                input_ids, attention_mask, target, info = self.model.batchify(test_instance)
                gold.update(formulate_gold(target, info))

                new_prediction = {}
                gen_triples = self.model.gen_triples(input_ids, attention_mask, info)
                prediction.update(gen_triples)
                new_prediction.update(gen_triples)
                # print("prediction:",new_prediction)
                # input('stop')

                # 生成数字串
                for idx in range(len(info['seq_len'])):
                    seq_len = info['seq_len'][idx]
                    result = ['0'] * (seq_len - 2)  # Initialize with '0'

                    new_idx = batch_id * batch_size + idx
                    
                    if new_idx in new_prediction:
                        for pred in new_prediction[new_idx]:
                            for i in range(pred.head_start_index, pred.head_end_index + 1):
                                if i < len(result):
                                    result[i] = '1'  # Set '1' for the range
                    else:
                        print(f"No prediction for new_idx {new_idx}")
                    output.append(''.join(result))

        # 将数字串写入 test.txt 文件
        with open('test_ex101.txt', 'w') as f:
            for line in output:
                f.write(line + '\n')

        # 计算评估指标
        results = metric(prediction, gold)
        return results

# # 实例化Tester类
# tester = Tester(model, data, args)
# tester.load_model("/data2024/yyyl/code/ProtBert/data/generated_data/ex2/Set-Prediction-Networks_protein_epoch_45_f1_0.9856.pth")  # 替换为最佳模型权重的路径
# results = tester.test_model()  # 不传递参数，直接调用
# print("Test Results:", results)
