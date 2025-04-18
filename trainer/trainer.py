import torch, random, gc
from torch import nn, optim
from tqdm import tqdm
from transformers import AdamW
from utils.average_meter import AverageMeter
from utils.functions import formulate_gold
# from utils.metric import metric, num_metric, overlap_metric
from utils.metric import metric
import matplotlib.pyplot as plt

class Trainer(nn.Module):
    def __init__(self, model, data, args):
        super().__init__()
        self.args = args
        self.model = model
        self.data = data

        no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
        component = ['encoder', 'decoder']
        grouped_params = [
            {
                'params': [p for n, p in self.model.named_parameters() if not any(nd in n for nd in no_decay) and component[0] in n],
                'weight_decay': args.weight_decay,
                'lr': args.encoder_lr
            },
            {
                'params': [p for n, p in self.model.named_parameters() if any(nd in n for nd in no_decay) and component[0] in n],
                'weight_decay': 0.0,
                'lr': args.encoder_lr
            },
            {
                'params': [p for n, p in self.model.named_parameters() if
                           not any(nd in n for nd in no_decay) and component[1] in n],
                'weight_decay': args.weight_decay,
                'lr': args.decoder_lr
            },
            {
                'params': [p for n, p in self.model.named_parameters() if
                           any(nd in n for nd in no_decay) and component[1] in n],
                'weight_decay': 0.0,
                'lr': args.decoder_lr
            }
        ]
        if args.optimizer == 'Adam':
            self.optimizer = optim.Adam(grouped_params)
        elif args.optimizer == 'AdamW':
            self.optimizer = AdamW(grouped_params)
        else:
            raise Exception("Invalid optimizer.")
        if args.use_gpu:
            self.cuda()

    def train_model(self):
        best_f1 = 0
        best_precision = 0
        best_recall = 0 
        best_mcc = 0
        best_specificity = 0
        train_loader = self.data.train_loader
        train_num = len(train_loader)
        batch_size = self.args.batch_size
        total_batch = train_num // batch_size + 1

        losses = []
        f1_list = []
        precision_list = []
        recall_list = []

        # result = self.eval_model(self.data.test_loader)
        for epoch in range(self.args.max_epoch):
            # Train
            self.model.train()
            self.model.zero_grad()
            self.optimizer = self.lr_decay(self.optimizer, epoch, self.args.lr_decay)
            print("=== Epoch %d train ===" % epoch, flush=True)
            avg_loss = AverageMeter()
            # random.shuffle(train_loader)

            for batch_id in range(total_batch):
                start = batch_id * batch_size
                end = (batch_id + 1) * batch_size
                if end > train_num:
                    end = train_num
                train_instance = train_loader[start:end]

                # # 打印 train_loader 和 train_instance 进行检查


                # print([ele[0] for ele in train_instance])
                if not train_instance:
                    continue

                input_ids, attention_mask, targets, _ = self.model.batchify(train_instance)
                loss, _ = self.model(input_ids, attention_mask, targets)
                # loss, output = self.model(input_ids, attention_mask, targets)
                # print("Model Output:", output)
                # input('stop')
                avg_loss.update(loss.item(), 1)
                # Optimize
                loss.backward()
                if self.args.max_grad_norm != 0:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.args.max_grad_norm)
                if (batch_id + 1) % self.args.gradient_accumulation_steps == 0:
                    self.optimizer.step()
                    self.model.zero_grad()
                if batch_id % 100 == 0 and batch_id != 0:
                    print("     Instance: %d; loss: %.4f" % (start, avg_loss.avg), flush=True)
            
            losses.append(avg_loss.avg)
            
            gc.collect()
            torch.cuda.empty_cache()

            # Validation
            print("=== Epoch %d Validation ===" % epoch)
            result = self.eval_model(self.data.valid_loader)
            
            precision = result['precision']
            precision_list.append(result['precision'])

            f1 = result['f1']
            f1_list.append(result['f1'])

            recall = result['recall']
            recall_list.append(result['recall'])

            specificity = result['specificity']
            mcc = result['mcc']

            if f1 > best_f1:
                print("Achieving Best Result on Validation Set.", flush=True)
                torch.save({'state_dict': self.model.state_dict()}, self.args.generated_param_directory + "%s_%s_epoch_%d_f1_%.4f.pth" %(self.args.model_name, self.args.dataset_name, epoch, result['f1']))
                best_f1 = f1
                best_result_epoch = epoch

            if precision > best_precision:
                best_precision = precision
            if recall > best_recall:
                best_recall = recall
            if specificity > best_specificity:
                best_specificity = specificity
            if mcc > best_mcc:
                best_mcc = mcc
            gc.collect()
            torch.cuda.empty_cache()
        print("Best result on test set is %f , %f, %f, %f, %f achieving at epoch %d." % (precision, recall, best_f1, specificity, mcc, best_result_epoch), flush=True)
        print("best_precision = ", best_precision, " best_recall = ", best_recall, " best_f1 = ", best_f1, "best_specificity = ",best_specificity, "best_mcc = ", best_mcc)
    

        # Plotting the loss and precision curves
        plt.figure(figsize=(12, 10))

        plt.subplot(2, 2, 1)
        plt.plot(losses, label='Training Loss')
        plt.title('Training Loss Over Epochs')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')

        plt.subplot(2, 2, 2)
        plt.plot(precision_list, label='Test Precision')
        plt.title('Test Precision Over Epochs')
        plt.xlabel('Epochs')
        plt.ylabel('Precision')

        plt.subplot(2, 2, 3)
        plt.plot(recall_list, label='Test Recall')  
        plt.title('Test Recall Over Epochs')
        plt.xlabel('Epochs')
        plt.ylabel('Recall')

        plt.subplot(2, 2, 4)
        plt.plot(f1_list, label='Test F1')  
        plt.title('Test F1 Over Epochs')
        plt.xlabel('Epochs')
        plt.ylabel('F1')

        plt.tight_layout()
        plt.savefig('ex1.png')
        plt.close()
        print("Test results saved to 'ex1.png'.")


    def eval_model(self, eval_loader):
        self.model.eval()
        # print(self.model.decoder.query_embed.weight)
        prediction, gold = {}, {}
        with torch.no_grad():
            batch_size = self.args.batch_size
            eval_num = len(eval_loader)
            total_batch = eval_num // batch_size + 1
            for batch_id in range(total_batch):
                start = batch_id * batch_size
                end = (batch_id + 1) * batch_size
                if end > eval_num:
                    end = eval_num
                eval_instance = eval_loader[start:end]
                if not eval_instance:
                    continue
                input_ids, attention_mask, target, info = self.model.batchify(eval_instance)
                gold.update(formulate_gold(target, info))
                # print(target)
                gen_triples = self.model.gen_triples(input_ids, attention_mask, info)
                prediction.update(gen_triples)
                # print("prediction:",prediction)
                # print("gold:",gold)
                # input('stop')
        # num_metric(prediction, gold)
        # overlap_metric(prediction, gold)
        return metric(prediction, gold)

    def load_state_dict(self, state_dict):
        self.model.load_state_dict(state_dict)

    @staticmethod
    def lr_decay(optimizer, epoch, decay_rate):
        # lr = init_lr * ((1 - decay_rate) ** epoch)
        if epoch != 0:
            for param_group in optimizer.param_groups:
                param_group['lr'] = param_group['lr'] * (1 - decay_rate)
                # print(param_group['lr'])
        return optimizer
