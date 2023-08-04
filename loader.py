# -*- coding: utf-8 -*-#
import time
import os
import numpy as np
from copy import deepcopy
from collections import Counter
from collections import OrderedDict
from ordered_set import OrderedSet
import torch
import torch.nn as nn
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from tqdm import tqdm
import random
from models.module import ModelManager
from utils.miulab import computeF1Score
class Args:
    alpha = 0.2
    attention_hidden_dim = 1024
    attention_output_dim = 128
    batch_size = 64
    data_dir = './data/MixATIS'
    decoder_gat_hidden_dim = 64
    dropout_rate = 0.4
    early_stop = False
    encoder_hidden_dim = 256
    gat_dropout_rate = 0.4
    gpu = False
    intent_embedding_dim = 64
    l2_penalty = 1e-06
    learning_rate = 0.001
    load_dir = None
    log_dir = './log/MixATIS'
    log_name = 'MixATIS.txt'
    n_heads=4
    n_layers_decoder=2
    num_epoch=10
    patience=10
    random_state=72
    row_normalized=True
    save_dir='./save/MixATIS'
    slot_decoder_hidden_dim=64
    slot_embedding_dim=128
    slot_forcing_rate = 0.9
    threshold = 0.5
    word_embedding_dim = 32
args=Args

train_path = os.path.join(args.data_dir, 'train.txt')
dev_path = os.path.join(args.data_dir, 'dev.txt')
test_path = os.path.join(args.data_dir, 'test.txt')
print(train_path)


def read_file(file_path):
    """ Read data file of given path.
    :param file_path: path of data file.
    :return: list of sentence, list of slot and list of intent.
    """
    texts, slots, intents = [], [], []
    text, slot = [], []
    with open(file_path, 'r', encoding="utf8") as fr:
        for line in fr.readlines():
            items = line.strip().split()
            if len(items) == 1:
                texts.append(text)
                slots.append(slot)
                if "/" not in items[0]:
                    intents.append(items)
                else:
                    new = items[0].split("/")
                    intents.append([new[1]])
                # clear buffer lists.
                text, slot = [], []
            elif len(items) == 2:
                text.append(items[0].strip())
                slot.append(items[1].strip())
    return texts, slots, intents
train_texts, train_slots, train_intents=read_file(file_path=train_path)
test_texts, test_slots, test_intents=read_file(file_path=test_path)
dev_texts, dev_slots, dev_intents=read_file(file_path=dev_path)
def get_counter(instances,multi_intent=False):
    from collections import Counter
    freq = Counter()
    if multi_intent:
        for instance in instances:
            for i in instance[0].split('#'):
                freq[i]+=1
    else:
        for instance in instances:
            for i in instance:
                freq[i] += 1
    return freq

def save_content( dir_path,name,counter):
    if not os.path.exists(dir_path):
        os.mkdir(dir_path)
    list_path = os.path.join(dir_path,name +"_list.txt")
    with open(list_path, 'w', encoding="utf8") as fw:
        for element, frequency in counter.most_common():
            fw.write(element + '\t' + str(frequency) + '\n')
texts_counter = get_counter(train_texts,multi_intent=False)
save_content( dir_path=args.save_dir,name='word',counter=texts_counter)
slot_counter = get_counter(train_slots,multi_intent=False)
save_content( dir_path=args.save_dir,name='slot',counter=slot_counter)
intent_counter =get_counter(train_intents,multi_intent=True)
save_content( dir_path=args.save_dir,name='intent',counter=intent_counter)

def get_dict( dir_path,texts_counter, intent_counter, slot_counter):
    #保存为文件
    word_dict,intent_dict,slot_dict ={},{},{}
    word_dict['<PAD>']=0
    word_dict['<UNK>'] = 1
    for key,values in texts_counter.items():
        if key not in word_dict:
            word_dict[key] = len(word_dict)
    for key,value in intent_counter.items():
        if key not in intent_dict:
            intent_dict[key]=len(intent_dict)
    for key, value in slot_counter.items():
        if key not in slot_dict:
            slot_dict[key] = len(slot_dict)
    word_dict_path = os.path.join(dir_path, 'word' + "_dict.txt")
    intent_dict_path = os.path.join(dir_path, 'intent' + "_dict.txt")
    slot_dict_path = os.path.join(dir_path, 'slot' + "_dict.txt")
    with open(word_dict_path, 'w', encoding="utf8") as fw:
        for index, element in enumerate(word_dict):
            fw.write(element + '\t' + str(index) + '\n')
    with open(intent_dict_path, 'w', encoding="utf8") as fw:
        for index, element in enumerate(intent_dict):
            fw.write(element + '\t' + str(index) + '\n')
    with open(slot_dict_path, 'w', encoding="utf8") as fw:
        for index, element in enumerate(slot_dict):
            fw.write(element + '\t' + str(index) + '\n')
    return word_dict, intent_dict, slot_dict
word_dict,intent_dict,slot_dict =get_dict(args.save_dir,texts_counter, intent_counter, slot_counter)


def get_idx(texts,slots,intents,max_len):
    words_idx, slots_idx, intents_idx = [], [], []
    for text in texts:
        word_idx = [word_dict.get(i,0) for i in text]

        words_idx.append(word_idx)
    for slot in slots:
        ss =[slot_dict.get(i,0) for i in slot]
        slots_idx.append(ss)
    for intent in train_intents:
        abc =[]
        target = intent[0].split('#')
        abc.append([intent_dict.get(i,0) for i in target])
        intents_idx.append(abc)
        # label = [0] * 17
        # for i, intent in enumerate(list(intent_dict.keys())):
        #     if intent in target:
        #         label[i] = 1
        # assert len(label)==len(list(intent_dict.keys()))
        # intents_idx.append(label)
    return words_idx, slots_idx, intents_idx
train_words_idx, train_slots_idx, train_intents_idx =get_idx(train_texts, train_slots, train_intents,32)
print(train_words_idx[0])
print(train_slots_idx[0])
print(train_intents_idx[0])

class TorchDataset(Dataset):
    """
    Helper class implementing torch.utils.data.Dataset to
    instantiate DataLoader which deliveries data batch.
    """
    def __init__(self, text, slot, intent):
        self.text = text
        self.slot = slot
        self.intent = intent
    def __getitem__(self, index):
        return self.text[index], self.slot[index], self.intent[index]
    def __len__(self):
        # Pre-check to avoid bug.
        assert len(self.text) == len(self.slot)
        return len(self.text)

train_set =TorchDataset(train_words_idx, train_slots_idx, train_intents_idx )


def collate_fn(batch):
    texts =[]
    slots = []
    intents = []
    for text, slot,intent in batch:
        texts.append(text)
        slots.append(slot)
        intents.append(intent)
    return  texts,slots,intents

train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True,collate_fn=collate_fn)


def add_padding(texts,items=None, digital=True):
    res= []
    len_list = [len(text) for text in texts]
    max_len = max(len_list)
    # print('max_len',max_len)
    # Get sorted index of len_list.
    sorted_index = np.argsort(len_list)[::-1]
    trans_texts, seq_lens, trans_items = [], [], None
    if items is not None:
        trans_items = [[] for _ in range(0, len(items))]
    for index in sorted_index:
        seq_lens.append(deepcopy(len_list[index]))
        trans_texts.append(deepcopy(texts[index]))
        if digital:
            trans_texts[-1].extend([0] * (max_len - len_list[index]))
        else:
            trans_texts[-1].extend(['<PAD>'] * (max_len - len_list[index]))

        if items is not None:
            for item, (o_item, required) in zip(trans_items, items):
                item.append(deepcopy(o_item[index]))
                if required:
                    if digital:
                        item[-1].extend([0] * (max_len - len_list[index]))
                    else:
                        item[-1].extend(['<PAD>'] * (max_len - len_list[index]))
    if items is not None:
        return trans_texts, trans_items, seq_lens
    else:
        return trans_texts, seq_lens



def normalize_adj(mx):
    """
    Row-normalize matrix  D^{-1}A
    torch.diag_embed: https://github.com/pytorch/pytorch/pull/12447
    """
    mx = mx.float()
    rowsum = mx.sum(2)
    r_inv = torch.pow(rowsum, -1)
    r_inv[torch.isinf(r_inv)] = 0.
    r_mat_inv = torch.diag_embed(r_inv, 0)
    mx = r_mat_inv.matmul(mx)
    return mx

def get_dataloader(args):
    train_path = os.path.join(args.data_dir, 'train.txt')
    dev_path = os.path.join(args.data_dir, 'dev.txt')
    test_path = os.path.join(args.data_dir, 'test.txt')
    train_texts, train_slots, train_intents = read_file(file_path=train_path)
    test_texts, test_slots, test_intents = read_file(file_path=test_path)
    dev_texts, dev_slots, dev_intents = read_file(file_path=dev_path)
    texts_counter = get_counter(train_texts, multi_intent=False)
    save_content(dir_path=args.save_dir, name='word', counter=texts_counter)
    slot_counter = get_counter(train_slots, multi_intent=False)
    save_content(dir_path=args.save_dir, name='slot', counter=slot_counter)
    intent_counter = get_counter(train_intents, multi_intent=True)
    save_content(dir_path=args.save_dir, name='intent', counter=intent_counter)
    word_dict, intent_dict, slot_dict = get_dict(args.save_dir, texts_counter, intent_counter, slot_counter)
    train_words_idx, train_slots_idx, train_intents_idx = get_idx(train_texts, train_slots, train_intents, 32)
    test_words_idx, test_slots_idx, test_intents_idx = get_idx(test_texts, test_slots, test_intents, 32)
    dev_words_idx, dev_slots_idx, dev_intents_idx = get_idx(dev_texts, dev_slots, dev_intents, 32)
    train_set = TorchDataset(train_words_idx, train_slots_idx, train_intents_idx)
    test_set = TorchDataset(test_words_idx, test_slots_idx, test_intents_idx)
    dev_set = TorchDataset(dev_words_idx, dev_slots_idx, dev_intents_idx)
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn)
    test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn)
    dev_loader = DataLoader(dev_set, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn)
    return train_loader,test_loader, dev_loader

class Processor(args):
    def __init__(self,args):

        self.args = args
        self.batch_size = args.batch_size
        self.load_dir = args.load_dir
        self.device ="cuda" if torch.cuda.is_available() else "cpu"
        self.criterion = nn.NLLLoss()
        self.criterion_intent = nn.BCEWithLogitsLoss()
    # optimizer = torch.optim.Adam(
    #         model.parameters(), lr=args.learning_rate, weight_decay=args.l2_penalty
    #     )
    # gpu_model_path = torch.load(os.path.join(load_dir, 'model/model.pkl'))
    # cpu_model = torch.load(os.path.join(load_dir, 'model/model.pkl'),map_location=torch.device('cpu'))

def multilabel2one_hot(labels, nums):
    res = [0.] * nums
    if len(labels) == 0:
        return res
    if isinstance(labels[0], list):
        for label in labels[0]:
            res[label] = 1.
        return res
    for label in labels:
        res[label] = 1.
    return res




def train(model,args):
    processor =Processor(args)
    optimizer=torch.optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.l2_penalty)
    best_dev_sent = 0.0
    best_epoch = 0
    no_improve = 0
    train_loader,test_loader, dev_loader =get_dataloader(args)
    for epoch in range(0, args.num_epoch):
        total_slot_loss, total_intent_loss = 0.0, 0.0
        time_start = time.time()
        model.train()
        for i, data in enumerate(tqdm(train_loader, ncols=50)):
            text_batch, slot_batch, intent_batch =data[0],data[1],data[2]
            padded_text, [sorted_slot, sorted_intent], seq_lens = \
                add_padding(text_batch,[(slot_batch, True), (intent_batch, False)])
            sorted_intent = [multilabel2one_hot(intents, len(intent_dict)) for intents in sorted_intent]
            texts_var = torch.LongTensor(padded_text)
            slots_var = torch.LongTensor(sorted_slot)
            intents_var = torch.Tensor(sorted_intent)
            # print(texts_var.shape)
            # print(slots_var.shape )
            # print(intents_var.shape)
            max_len = np.max(seq_lens)
            if args.gpu:
                texts_var = texts_var.cuda()
                slots_var = slots_var.cuda()
                intents_var = intents_var.cuda()
            random_slot, random_intent = random.random(), random.random()
            if random_slot < args.slot_forcing_rate:

                slot_out, intent_out = model(texts_var, seq_lens, forced_slot=slots_var)
            else:
                slot_out, intent_out = model(texts_var, seq_lens)
            slots_var = torch.cat([slots_var[i][:seq_lens[i]] for i in range(0, len(seq_lens))], dim=0)
            slot_loss = processor.criterion(slot_out, slots_var)
            intent_loss = processor.criterion_intent(intent_out, intents_var)
            batch_loss = slot_loss + intent_loss
            optimizer.zero_grad()
            batch_loss.backward()
            optimizer.step()
            try:
                total_slot_loss += slot_loss.cpu().item()
                total_intent_loss += intent_loss.cpu().item()
            except AttributeError:
                total_slot_loss += slot_loss.cpu().data.numpy()[0]
                total_intent_loss += intent_loss.cpu().data.numpy()[0]
        time_con = time.time() - time_start
        print(
            '[Epoch {:2d}]: The total slot loss on train data is {:2.6f}, intent data is {:2.6f}, cost ' \
            'about {:2.6} seconds.'.format(epoch, total_slot_loss, total_intent_loss, time_con))
        change, time_start = False, time.time()
        dev_slot_f1_score, dev_intent_f1_score, dev_intent_acc_score, dev_sent_acc_score =\
            estimate(if_dev=True,test_batch=args.batch_size,args=args)
        if dev_sent_acc_score > best_dev_sent:
            no_improve = 0
            best_epoch = epoch
            best_dev_sent = dev_sent_acc_score
            test_slot_f1, test_intent_f1, test_intent_acc, test_sent_acc = estimate(
                if_dev=False,model=model, test_loader=test_loader,dev_loader=dev_loader,
                args=args,test_batch=args.batch_size)
            print('\nTest result: epoch: {}, slot f1 score: {:.4f}, intent f1 score: {:.4f},'
                  ' intent acc score:'
                  ' {:.4f}, semantic accuracy score: {:.4f}.'.
                  format(epoch, test_slot_f1, test_intent_f1, test_intent_acc, test_sent_acc))
            model_save_dir = os.path.join(args.save_dir, "model")
            if not os.path.exists(model_save_dir):
                os.mkdir(model_save_dir)
            torch.save(model, os.path.join(model_save_dir, "model.pkl"))
            # torch.save(dataset, os.path.join(model_save_dir, 'dataset.pkl'))
            time_con = time.time() - time_start
            print('[Epoch {:2d}]: In validation process, the slot f1 score is {:2.6f}, ' \
                  'the intent f1 score is {:2.6f}, the intent acc score is {:2.6f}, '
                  'the semantic acc is {:.2f}, cost about {:2.6f} seconds.\n'.format(
                epoch, dev_slot_f1_score, dev_intent_f1_score, dev_intent_acc_score,
                dev_sent_acc_score, time_con))
        else:
            no_improve += 1
        if args.early_stop == True:
            if no_improve > args.patience:
                print('early stop at epoch {}'.format(epoch))
                break
        print('Best epoch is {}'.format(best_epoch))
        return best_epoch
def instance2onehot(func, num_intent, data):
    res = []
    for intents in func(data):
        res.append(multilabel2one_hot(intents, num_intent))
    return np.array(res)

model = ModelManager(
    args, len(word_dict),
    len(slot_dict),
    len(intent_dict)
)
# train(model,args)

def estimate(if_dev,  model,test_loader,dev_loader,args, test_batch=100):
    """
    Estimate the performance of model on dev or test dataset.
    """
    if if_dev:
        ss, pred_slot, real_slot, pred_intent, real_intent = prediction(
            model, dev_loader,"dev", test_batch, args)
    else:
        ss, pred_slot, real_slot, pred_intent, real_intent = prediction(
            model, test_loader,dev_loader,"test", args)
    num_intent = len(intent_dict)
    slot_f1_score =computeF1Score(ss, real_slot, pred_slot, args)[0]
    intent_f1_score = f1_score(
        instance2onehot(get_index, num_intent, real_intent),
        instance2onehot(get_index, num_intent, pred_intent),
        average='macro')
    intent_acc_score = intent_acc(pred_intent, real_intent)
    sent_acc = semantic_acc(pred_slot, real_slot, pred_intent, real_intent)
    print("slot f1: {}, intent f1: {}, intent acc: {}, exact acc: {}".format(
        slot_f1_score, intent_f1_score,intent_acc_score, sent_acc))
    # Write those sample both have intent and slot errors.
    with open(os.path.join(args.save_dir, 'error.txt'), 'w', encoding="utf8") as fw:
        for p_slot_list, r_slot_list, p_intent_list, r_intent in \
                zip(pred_slot, real_slot, pred_intent, real_intent):
            fw.write(','.join(p_intent_list) + '\t' + ','.join(r_intent) + '\n')
            for w, r_slot, in zip(p_slot_list, r_slot_list):
                fw.write(w + '\t' + r_slot + '\t''\n')
            fw.write('\n\n')
    return slot_f1_score, intent_f1_score, intent_acc_score, sent_acc


def prediction(model, test_loader,dev_loader,mode, batch_size, args):
    model.eval()
    if mode == "dev":
        dataloader = dev_loader
    elif mode == "test":
        dataloader = test_loader
    else:
        raise Exception("Argument error! mode belongs to {\"dev\", \"test\"}.")
    pred_slot, real_slot = [], []
    pred_intent, real_intent = [], []
    all_token = []
    for i, data in enumerate(tqdm(train_loader, ncols=50)):
        text_batch, slot_batch, intent_batch = data[0], data[1], data[2]
        padded_text, [sorted_slot, sorted_intent], seq_lens = \
            add_padding(text_batch, [(slot_batch, False), (intent_batch, False)],digital=False)

        real_slot.extend(sorted_slot)
        all_token.extend([pt[:seq_lens[idx]] for idx, pt in enumerate(padded_text)])
        for intents in list(expand_list(sorted_intent)):
            if '#' in intents:
                real_intent.append(intents.split('#'))
            else:
                real_intent.append([intents])

        digit_text = get_index(padded_text)
        var_text = torch.LongTensor(digit_text)
        max_len = np.max(seq_lens)
        if args.gpu:
            var_text = var_text.cuda()
        slot_idx, intent_idx = model(var_text, seq_lens, n_predicts=1)
        nested_slot = nested_list([list(expand_list(slot_idx))], seq_lens)[0]
        pred_slot.extend(get_instance(nested_slot))
        intent_idx_ = [[] for i in range(len(digit_text))]
        for item in intent_idx:
            intent_idx_[item[0]].append(item[1])
        intent_idx = intent_idx_
        pred_intent.extend(intent_alphabet.get_instance(intent_idx))
    # if 'MixSNIPS' in args.data_dir or 'MixATIS' in args.data_dir or 'DSTC' in args.data_dir:
    [p_intent.sort() for p_intent in pred_intent]
    [r_intent.sort() for r_intent in real_intent]
    with open(os.path.join(args.save_dir, 'token.txt'), "w", encoding="utf8") as writer:
        idx = 0
        for line, slots, rss in zip(all_token, pred_slot, real_slot):
            for c, sl, rsl in zip(line, slots, rss):
                writer.writelines(
                    str(sl == rsl) + " " + c + " " + sl + " " + rsl + "\n")
            idx = idx + len(line)
            writer.writelines("\n")

    return all_token, pred_slot, real_slot, pred_intent, real_intent

def get_index(instance, multi_intent=False):
    if isinstance(instance, (list, tuple)):
        return [get_index(elem, multi_intent=multi_intent) for elem in instance]

    assert isinstance(instance, str)
    if multi_intent and '#' in instance:
        return [get_index(element, multi_intent=multi_intent) for element in instance.split('#')]

    try:
        return self.__instance2index[instance]
    except KeyError:
        if if_use_unk:
            return self.__instance2index[self.__sign_unk]
        else:
            max_freq_item = self.__counter.most_common(1)[0][0]
            return self.__instance2index[max_freq_item]


def f1_score(pred_list, real_list):
    """
    Get F1 score measured by predictions and ground-trues.
    """
    tp, fp, fn = 0.0, 0.0, 0.0
    for i in range(len(pred_list)):
        seg = set()
        result = [elem.strip() for elem in pred_list[i]]
        target = [elem.strip() for elem in real_list[i]]

        j = 0
        while j < len(target):
            cur = target[j]
            if cur[0] == 'B':
                k = j + 1
                while k < len(target):
                    str_ = target[k]
                    if not (str_[0] == 'I' and cur[1:] == str_[1:]):
                        break
                    k = k + 1
                seg.add((cur, j, k - 1))
                j = k - 1
            j = j + 1

        tp_ = 0
        j = 0
        while j < len(result):
            cur = result[j]
            if cur[0] == 'B':
                k = j + 1
                while k < len(result):
                    str_ = result[k]
                    if not (str_[0] == 'I' and cur[1:] == str_[1:]):
                        break
                    k = k + 1
                if (cur, j, k - 1) in seg:
                    tp_ += 1
                else:
                    fp += 1
                j = k - 1
            j = j + 1

        fn += len(seg) - tp_
        tp += tp_

    p = tp / (tp + fp) if tp + fp != 0 else 0
    r = tp / (tp + fn) if tp + fn != 0 else 0
    return 2 * p * r / (p + r) if p + r != 0 else 0

def max_freq_predict(sample):
    predict = []
    for items in sample:
        predict.append(Counter(items).most_common(1)[0][0])
    return predict

def exp_decay_predict(sample, decay_rate=0.8):
    predict = []
    for items in sample:
        item_dict = {}
        curr_weight = 1.0
        for item in items[::-1]:
            item_dict[item] = item_dict.get(item, 0) + curr_weight
            curr_weight *= decay_rate
        predict.append(sorted(item_dict.items(), key=lambda x_: x_[1])[-1][0])
    return predict

def expand_list(nested_list):
    for item in nested_list:
        if isinstance(item, (list, tuple)):
            for sub_item in expand_list(item):
                yield sub_item
        else:
            yield item

def intent_acc(pred_intent, real_intent):
    total_count, correct_count = 0.0, 0.0
    for p_intent, r_intent in zip(pred_intent, real_intent):

        if p_intent == r_intent:
            correct_count += 1.0
        total_count += 1.0

    return 1.0 * correct_count / total_count

def semantic_acc(pred_slot, real_slot, pred_intent, real_intent):
    """
    Compute the accuracy based on the whole predictions of
    given sentence, including slot and intent.
    """
    total_count, correct_count = 0.0, 0.0
    for p_slot, r_slot, p_intent, r_intent in zip(pred_slot, real_slot, pred_intent, real_intent):

        if p_slot == r_slot and p_intent == r_intent:
            correct_count += 1.0
        total_count += 1.0

    return 1.0 * correct_count / total_count

def accuracy(pred_list, real_list):
    """
    Get accuracy measured by predictions and ground-trues.
    """

    pred_array = np.array(list(expand_list(pred_list)))
    real_array = np.array(list(expand_list(real_list)))
    return (pred_array == real_array).sum() * 1.0 / len(pred_array)

def f1_score_intents(pred_array, real_array):
    pred_array = pred_array.transpose()
    real_array = real_array.transpose()
    P, R, F1 = 0, 0, 0
    for i in range(pred_array.shape[0]):
        TP, FP, FN = 0, 0, 0
        for j in range(pred_array.shape[1]):
            if (pred_array[i][j] + real_array[i][j]) == 2:
                TP += 1
            elif real_array[i][j] == 1 and pred_array[i][j] == 0:
                FN += 1
            elif pred_array[i][j] == 1 and real_array[i][j] == 0:
                FP += 1
        precision = TP / float(TP + FP) if (TP + FP) != 0 else 0
        recall = TP / float(TP + FN) if (TP + FN) != 0 else 0
        F1 += 2 * precision * recall / float(precision + recall) if (precision + recall) != 0 else 0
        P += precision
        R += recall
    P /= pred_array.shape[0]
    R /= pred_array.shape[0]
    F1 /= pred_array.shape[0]
    return F1

def nested_list(items, seq_lens):
    num_items = len(items)
    trans_items = [[] for _ in range(0, num_items)]

    count = 0
    for jdx in range(0, len(seq_lens)):
        for idx in range(0, num_items):
            trans_items[idx].append(items[idx][count:count + seq_lens[jdx]])
        count += seq_lens[jdx]
    return trans_items


print(len(word_dict))
model = ModelManager(
    args, len(word_dict),
    len(intent_dict),
    len(slot_dict)
)



