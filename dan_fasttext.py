import torch
import argparse
from numpy import *
from sklearn.utils import class_weight
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
import time
import logging
from sklearn.metrics import f1_score, classification_report
from sklearn import metrics
from transformers import AdamW, get_cosine_schedule_with_warmup
import sklearn.utils.class_weight
import numpy as np
import numpy
import collections
from collections import Counter
import fasttext.util
from nltk import ngrams
import fasttext

logging.basicConfig(filename='log/dan_tncc_mlp.log', filemode="w",
                    format="%(asctime)s %(name)s:%(levelname)s:%(message)s",
                    datefmt="%d-%m-%Y %H:%M:%S",
                    level=logging.DEBUG)
logger = logging.getLogger("DAN_TNCC")

fasttext_model = fasttext.load_model("model/tncc_title_200.bin")

lmd = 0.0


class MlpModel(torch.nn.Module):
    def __init__(self, input_num, hidden_num, out_num):
        super(MlpModel, self).__init__()
        self.linear1 = torch.nn.Linear(input_num, hidden_num)
        self.linear2 = torch.nn.Linear(2 * hidden_num, hidden_num)
        self.linear3 = torch.nn.Linear(hidden_num, hidden_num)
        self.linear4 = torch.nn.Linear(hidden_num, out_num)
        self.relu = torch.nn.ReLU()

    def forward(self, x, text_avs):
        x = self.linear1(x)
        x = self.relu(x)
        x = torch.concat((x, text_avs), 1)
        x = self.linear2(x)
        x = self.relu(x)
        x = self.linear3(x)
        x = self.relu(x)
        x = lmd * x + (1 - lmd) * text_avs
        x = self.linear4(x)
        return x


class DAN_Trainer():
    def __init__(self, args, dan_preprocess):
        self.args = args
        self.dan_preprocess = dan_preprocess
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.vocab = self.dan_preprocess.wordMapAll
        self.revocab = self.reVocab()
        self.pading_num = len(self.vocab)
        self.pre_embeddings_params = self.pretraine_word_embeddings()
        self.class_names = ["Politics", "Economics", "Education", "Tourism", "Environment", "Language", "Literature",
                            "Religion", "Arts", "Medicine", "Customs", "Instruments"]

    def pretraine_word_embeddings(self):
        if self.args['rand_We']:
            print('randomly initializing word embeddings...')
            orig_We = (random.rand(self.args['d'], len(self.vocab)) * 2 - 1) * 0.08
        else:
            print('loading pretrained word embeddings...')
            orig_We = self.dan_preprocess.We
        return orig_We

    def reVocab(self):
        revMap = dict()
        for k, v in self.vocab.items():
            revMap[v] = k
        return revMap

    def dataset(self, data_path):
        input_ids, input_ids_lengths, labels = [], [], []
        if data_path == 'train':
            rootfines = self.dan_preprocess.train_sents
        elif data_path == 'dev':
            rootfines = self.dan_preprocess.dev_sents
        elif data_path == 'val':
            rootfines = self.dan_preprocess.test_sents
        else:
            raise Exception(f"data_path error，data_path={data_path}")
        c = Counter()
        tot = 0
        max_length = max([len(item[0]) for item in rootfines])
        for rootfine in rootfines:
            sent = rootfine[0]
            padding_sent = [self.pading_num] * max_length
            input_ids_length = len(sent)
            sent = sent + padding_sent[input_ids_length:]
            label = int(self.class_names.index(rootfine[1]))
            input_ids.append(sent)
            input_ids_lengths.append(input_ids_length)
            labels.append(label)
            c[label] += 1
            tot += 1
        print(data_path, c, tot)
        print(sorted(c.keys()), len(c.keys()))
        if data_path == 'train':
            self.y_train = np.array(labels)
        return torch.tensor(input_ids), torch.tensor(input_ids_lengths), torch.tensor(labels)

    def data_loader(self, input_ids, input_ids_lengths, labels):
        data = TensorDataset(input_ids, input_ids_lengths, labels)
        loader = DataLoader(data, batch_size=self.args['batch_size'], shuffle=True)
        return loader

    def predict(self, model, test_loader):
        test_pred, test_true = [], []
        with torch.no_grad():
            for idx, (ids, ids_length, y_true) in enumerate(test_loader):
                ids, ids_length = ids.numpy().tolist(), ids_length.numpy().tolist()
                ids_avs = []
                text_avs = []
                for sent_idx in range(len(ids)):
                    sent = ids[sent_idx][:ids_length[sent_idx]]
                    curr_sent = []
                    mask = random.rand(len(sent)) > self.args['word_drop']
                    for index, keep in enumerate(mask):
                        if keep:
                            curr_sent.append(sent[index])
                    # all examples must have at least one word
                    if len(curr_sent) == 0:
                        curr_sent = sent
                    curr_sent_text = ' '.join([self.dan_preprocess.revMap[item] for item in curr_sent])
                    text_avs.append(fasttext_model.get_sentence_vector(curr_sent_text))
                    av = average(self.pre_embeddings_params[:, curr_sent], axis=1)
                    ids_avs.append(av)
                ids = torch.Tensor(numpy.array(ids_avs))
                text_avs = torch.Tensor(numpy.array(text_avs))
                ids, y_true = ids.to(self.device), y_true.to(self.device)
                text_avs = text_avs.to(self.device)
                y_pred = model(ids, text_avs)
                y_pred = torch.argmax(y_pred, dim=1).detach().cpu().numpy().tolist()
                test_pred.extend(y_pred if isinstance(y_pred, list) else [y_pred])

                y_true = y_true.squeeze().cpu().numpy().tolist()
                test_true.extend(y_true if isinstance(y_true, list) else [y_true])
        return test_true, test_pred

    def train(self, model, train_loader, dev_loader, optimizer, schedule):
        class_weights = class_weight.compute_class_weight(class_weight='balanced', classes=np.unique(self.y_train),
                                                          y=self.y_train)
        class_weights = torch.from_numpy(class_weights).float()
        criterion = torch.nn.CrossEntropyLoss(class_weights)
        best_dev_result = 0.0
        for epoch in range(0, self.args['num_epochs']):
            print(f"----------------------------epoch={epoch}------------------")
            epoch_loss = 0.0
            start_time = time.time()
            model.train()
            for idx, (ids, ids_length, y_true) in enumerate(train_loader):
                ids, ids_length = ids.numpy().tolist(), ids_length.numpy().tolist()
                ids_avs = []
                text_avs = []
                for sent_idx in range(len(ids)):
                    sent = ids[sent_idx][:ids_length[sent_idx]]
                    curr_sent = []
                    mask = random.rand(len(sent)) > self.args['word_drop']
                    for index, keep in enumerate(mask):
                        if keep:
                            curr_sent.append(sent[index])
                    # all examples must have at least one word
                    if len(curr_sent) == 0:
                        curr_sent = sent
                    curr_sent_text = ' '.join([self.dan_preprocess.revMap[item] for item in curr_sent])
                    text_avs.append(fasttext_model.get_sentence_vector(curr_sent_text))
                    av = average(self.pre_embeddings_params[:, curr_sent], axis=1)
                    ids_avs.append(av)
                ids = torch.Tensor(numpy.array(ids_avs))
                text_avs = torch.Tensor(numpy.array(text_avs))
                ids, y_true = ids.to(self.device), y_true.to(self.device)
                text_avs = text_avs.to(self.device)
                y_pred = model(ids, text_avs)
                loss = criterion(y_pred, y_true)
                loss /= self.args['gradient_acc']
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                schedule.step()
                epoch_loss += loss.item()
                if (idx + 1) % (len(train_loader) // 10) == 0:
                    logger.info("Epoch {:02d} | Step {:03d}/{:03d} | Loss {:.4f} | Time {:.2f}".format(
                        epoch + 1, idx + 1, len(train_loader), epoch_loss / (idx + 1), time.time() - start_time))
                    # print("Epoch {:02d} | Step {:03d}/{:03d} | Loss {:.4f} | Time {:.2f}".format(
                    #     epoch + 1, idx + 1, len(train_loader), epoch_loss / (idx + 1), time.time() - start_time))
            model.eval()
            dev_true, dev_pred = self.predict(model, dev_loader)
            micro_f1 = f1_score(dev_true, dev_pred, average='macro')
            # acc = metrics.accuracy_score(dev_true, dev_pred)
            current_dev_result = micro_f1
            if current_dev_result > best_dev_result:
                best_dev_result = current_dev_result
                torch.save(model.state_dict(), self.args['model_save_dir'])
            logger.info(
                "current_dev_result is {:.4f}, best_dev_result is {:.4f}".format(current_dev_result, best_dev_result))
            logger.info("Time costed : {}s \n".format(round(time.time() - start_time, 3)))
            print("current_dev_result is {:.4f}, best_dev_result is {:.4f}".format(current_dev_result, best_dev_result))
            print("Time costed : {}s \n".format(round(time.time() - start_time, 3)))

    def run_finetune(self):
        train_loader = self.data_loader(*self.dataset('train'))
        dev_loader = self.data_loader(*self.dataset('dev'))
        test_loader = self.data_loader(*self.dataset('val'))
        temp_loader = dev_loader
        dev_loader = test_loader
        test_loader = temp_loader
        model = MlpModel(self.args['d'], self.args['dh'], self.args['labels']).to(self.device)
        total_steps = len(train_loader) * self.args['num_epochs']
        optimizer = AdamW(params=model.parameters(),
                          lr=self.args['lr'],
                          weight_decay=self.args['weight_decay'])
        schedule = get_cosine_schedule_with_warmup(optimizer=optimizer,
                                                   num_warmup_steps=self.args['warmup_rate'],
                                                   num_training_steps=total_steps)
        self.train(model, train_loader, dev_loader, optimizer, schedule)
        model.load_state_dict(torch.load(self.args['model_save_dir']))
        test_true, test_pred = self.predict(model, test_loader)
        logger.info('\n' + classification_report(test_true, test_pred,
                                                 target_names=self.class_names,
                                                 digits=6))
        print('\n' + classification_report(test_true, test_pred,
                                           target_names=self.class_names,
                                           digits=6))
        print(f1_score(test_true, test_pred, average="macro"))
        with open("../log/lmd.txt", encoding="utf-8", mode="a") as f:
            print(f"---------------------lmd={lmd}\n")
            f.writelines(f"lmd={lmd}\n")
            f.writelines('\n' + classification_report(test_true, test_pred,
                                                      target_names=self.class_names,
                                                      digits=6))
            f.writelines(f"\n\n\n")


class DAN_preprocess():
    def __init__(self, args):
        self.args = args
        self.textNumConverter()
        self.load_embeddings()

    def textNumConverter(self):
        wmap = self.buildWordMap()
        ddir = self.args['data']
        print('num words: ', len(wmap))
        with open(f'{ddir}train.txt', mode='r', encoding='utf-8') as f:
            train = f.readlines()
        with open(f'{ddir}dev.txt', mode='r', encoding='utf-8') as f:
            dev = f.readlines()
        with open(f'{ddir}test.txt', mode='r', encoding='utf-8') as f:
            test = f.readlines()

        revMap = {}
        for k, v in wmap.items():
            revMap[v] = k
        self.revMap = revMap
        print(len(train), len(dev), len(test))

        # store train root labels
        t_sents = []
        for line in train:
            lines = line.split('\t')
            label = lines[0].strip()
            line_list = lines[1].strip().split()
            line_list2 = [f"{item[0]}་{item[1]}" for item in ngrams(line_list, 2)]
            line_list = line_list + line_list2
            sent = []
            for line_item in line_list:
                sent.append(wmap[line_item])
            t_sents.append([sent, label])
        print(t_sents[0][0])
        print([revMap[x] for x in t_sents[0][0]])
        print('num train instances ', len(t_sents))
        c = Counter()
        for sent, label in t_sents:
            c[label] += 1
        print(c)
        self.train_sents = t_sents

        # store both phrases and roots for dev / val
        dev_sents = []
        for line in dev:
            lines = line.split('\t')
            label = lines[0].strip()
            line_list = lines[1].strip().split()
            line_list2 = [f"{item[0]}་{item[1]}" for item in ngrams(line_list, 2)]
            line_list = line_list + line_list2
            sent = []
            for line_item in line_list:
                sent.append(wmap[line_item])
            dev_sents.append([sent, label])
        print(dev_sents[0][0])
        print([revMap[x] for x in dev_sents[0][0]])
        print('dev phrase length ', len(dev_sents))
        c = Counter()
        for sent, label in dev_sents:
            c[label] += 1
        print(c)
        self.dev_sents = dev_sents

        test_sents = []
        for line in test:
            lines = line.split('\t')
            label = lines[0].strip()
            line_list = lines[1].strip().split()
            line_list2 = [f"{item[0]}་{item[1]}" for item in ngrams(line_list, 2)]
            line_list = line_list + line_list2
            sent = []
            for line_item in line_list:
                sent.append(wmap[line_item])
            test_sents.append([sent, label])
        print(test_sents[0][0])

        print([revMap[x] for x in test_sents[0][0]])
        print('val phrase length ', len(test_sents))
        c = Counter()
        for sent, label in test_sents:
            c[label] += 1
        print(c)
        self.test_sents = test_sents

    def buildWordMap(self):
        """
        Builds map of all words in training set
        to integer values.
        """
        words = collections.defaultdict(int)
        ddir = self.args['data']

        for file in [ddir + 'train.txt', ddir + 'dev.txt', ddir + 'test.txt']:
            with open(file, 'r', encoding='utf-8') as fid:
                lines = fid.readlines()
                for line in lines:
                    line_list = line.split('\t')
                    assert len(line_list) == 2, print(line)
                    line1_list = line_list[1].split()
                    line_list2 = [f"{item[0]}་{item[1]}" for item in ngrams(line1_list, 2)]
                    line1_list = line1_list + line_list2
                    for word in line1_list:
                        words[word.strip()] += 1

            print("Counting words..")
            print(len(words))

        words = sorted(words.items(), key=lambda x: x[1], reverse=True)
        words_list = [item[0] for item in words]
        wordMap = dict(zip(words_list, range(len(words))))

        self.wordMapAll = wordMap

        return wordMap

    def load_embeddings(self):
        ft_ext = fasttext.load_model(self.args['model_pretrain_dir'])
        wmap = self.wordMapAll
        revMap = {}
        for word in wmap:
            revMap[wmap[word]] = word
        d = ft_ext.get_dimension()
        all_vocab = {}
        ti_words = wmap.keys()
        for ti_word in ti_words:
            all_vocab[ti_word] = ft_ext.get_word_vector(ti_word)
        print(len(wmap), len(all_vocab), float(len(all_vocab)) / float(len(wmap)))
        We = empty((d, len(wmap)))
        print('creating We for ', len(wmap), ' words')
        unknown = []
        for i in range(0, len(wmap)):
            word = revMap[i]
            try:
                We[:, i] = all_vocab[word]
            except KeyError:
                unknown.append(word)
                # print('unknown: ', word)
                We[:, i] = ["0.0"] * d
        print('num unknowns: ', len(unknown))
        print(unknown[0:10])
        print(We.shape)
        self.We = We


def main():
    parser = argparse.ArgumentParser(description='TNCC DAN')
    parser.add_argument('-model_pretrain_dir', help="model_pretrain_dir", type=str,
                        default="model\cc.bo.300.ext.bin")
    parser.add_argument('-model_save_dir', help="model_save_dir", default="../save_models/best_dan_tncc.pth")
    parser.add_argument('-weight_decay', help="weight_decay", default=1e-4)
    parser.add_argument('-warmup_rate', help="warmup_rate", default=0.01)
    parser.add_argument('-gradient_acc', help="gradient_acc", default=1)
    parser.add_argument('-word_drop', help="word_drop", default=0.1)
    parser.add_argument('-data', help='location of dataset', default='data/')
    parser.add_argument('-rand_We', help='randomly init word embeddings', type=int, default=0)
    parser.add_argument('-d', help='word embedding dimension', type=int, default=300)
    parser.add_argument('-dh', help='hidden dimension', type=int, default=200)
    parser.add_argument('-rho', help='regularization weight', type=float, default=1e-4)
    parser.add_argument('-labels', help='number of labels', type=int, default=12)
    parser.add_argument('-ft', help='fine tune word vectors', type=int, default=1)
    parser.add_argument('-b', '--batch_size', help='adagrad minibatch size (ideal: 25 minibatches \
                        per epoch). for provided datasets, x for history and y for lit', type=int, \
                        default=30)
    parser.add_argument('-ep', '--num_epochs', help='number of training epochs, can also determine \
                         dynamically via validate method', type=int, default=30)
    parser.add_argument('-lr', help='adagrad initial learning rate', type=float, default=2e-3)
    args = vars(parser.parse_args())
    dan_preprocess = DAN_preprocess(args)
    trainer = DAN_Trainer(args, dan_preprocess)
    trainer.run_finetune()


def split_by_ratio(data, ratio):
    train_index = int((ratio[0] / sum(ratio)) * len(data))
    dev_index = int(((ratio[0] + ratio[1]) / sum(ratio)) * len(data))
    return [data[:train_index], data[train_index:dev_index], data[dev_index:]]


def write_lines_file(lines, file_name):
    with open(file_name, mode='w', encoding='utf-8') as f:
        f.writelines(lines)


def preprocess_data():
    import os
    path = "data/text/title-tibetan.txt"
    os.remove('data/train.txt')
    os.remove('data/dev.txt')
    os.remove('data/test.txt')
    with open(path, mode='r', encoding='utf-8') as f:
        lines = f.readlines()
    random.shuffle(lines)
    ratio = [8, 1, 1]
    train, dev, test = split_by_ratio(lines, ratio)
    write_lines_file(train, 'data/train.txt')
    write_lines_file(dev, 'data/dev.txt')
    write_lines_file(test, 'data/test.txt')
    print(f"train:{len(train)},dev:{len(dev)},val:{len(test)},total:{len(lines)}")


def embeding_fasttext():
    data_paths = ["data/train.txt", "data/dev.txt", "data/test.txt"]
    for data_path in data_paths:
        with open(data_path, encoding="utf-8", mode="r") as f:
            contents = f.readlines()
        new_data_path = data_path.replace("data", "data/fasttext")
        with open(new_data_path, encoding="utf-8", mode="w") as fw:
            for line in contents:
                label, text = line.split('\t')
                label = f"__label__{label}"
                newline = label + " " + text
                fw.writelines(newline)
    import fasttext
    model = fasttext.train_supervised(input="data/fasttext/train.txt",
                                      autotuneValidationFile="data/fasttext/dev.txt", epoch=30, lr=1,
                                      wordNgrams=2, dim=200)
    model.save_model("model/tncc_title_200.bin")
    model = fasttext.load_model("model/tncc_title_200.bin")
    scores = model.test("data/fasttext/test.txt")
    print(scores)
    global fasttext_model
    fasttext_model = fasttext.load_model("model/tncc_title_200.bin")


def main1():
    global lmd
    for i in range(11):
        lmd = i / 10
        print(lmd)
        main()


if __name__ == '__main__':
    # preprocess_data()
    embeding_fasttext()
    # global lmd
    # for i in range(10):
    #     print(i)
    # main()
    # for i in range(7):
    #     main1()
    main()
