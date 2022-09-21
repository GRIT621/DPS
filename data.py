import logging
import math

import numpy as np
from torch.utils.data import Dataset, DataLoader
import torch
import pandas as pd
from transformers import AutoTokenizer, AutoModel, AutoConfig, BertTokenizer, BertModel, BertConfig, BertForSequenceClassification
from sklearn.model_selection import train_test_split


logger = logging.getLogger(__name__)

class Base_dataset(Dataset):

    def __init__(self,args,model,mode,train_data= None,train_label = None,test_data = None,test_label= None):
        self.dataset = args.dataset
        self.model = model
        self.mode = mode
        self.max_len = args.max_len

        if self.dataset == 'AGNews':
            args.num_classes = 4
        elif self.dataset == 'Yelp':
            args.num_classes = 5
        if self.mode == "test":
            self.test_data = test_data
            self.test_label = test_label
            print("test data %d" ,len(test_data))
        else:
            n = len(train_label)
            if self.model == "bert":
                self.train_data_l = train_data[:args.num_labeled]
                self.train_data_u = train_data[args.num_labeled:]
                self.train_label_l = train_label[:args.num_labeled]
                self.train_label_u = train_label[args.num_labeled:]
                self.dev_data = test_data
                self.dev_label = test_label
            print("labeled data %d, unlabeled data %d" ,args.num_labeled, n - args.num_labeled)

    def __getitem__(self, index):

        if self.mode == 'labeled':
            text, target = self.train_data_l[index], self.train_label_l[index]

            text1 = torch.tensor(text)
            return text1, target
        elif self.mode == 'unlabeled':
            text, target = self.train_data_u[index], self.train_label_u[index]
            text1 = torch.tensor(text)
            text2 = torch.tensor(text)

            return text1, text2, target
        elif self.mode == 'test' or self.mode == "dev":
            text, target = self.test_data[index], self.test_label[index]
            text1 = torch.tensor(text)
            return text1, target

        def __len__(self):
            if self.mode == 'labeled':
                return len(self.train_data_l)
            elif self.mode == 'unlabeled':
                return len(self.train_data_u)
            else:
                return len(self.test_data)



    def __len__(self):
        if self.mode == 'labeled':
            return len(self.train_data_l)
        elif self.mode == 'unlabeled':
            return len(self.train_data_u)
        else:
            return len(self.test_data)

def get_AGNews(args):
    pretrained_weights = 'bert-base-cased'

    model_class = BertForSequenceClassification
    tokenizer_class = BertTokenizer

    tokenizer = tokenizer_class.from_pretrained(pretrained_weights, do_lower_case=False)

    def tokenizen(sent):

        # sent = tokenizer.encode(sent)
        #
        # return  sent[:lg - 2]  + [0] * (lg - len(sent[:lg - 2]) - 2)
        encoded_dict = tokenizer(sent, max_length=args.max_len, padding='max_length', truncation=True, )
        return encoded_dict['input_ids'],  encoded_dict['attention_mask']


    data = pd.read_csv('./dataset/AG_NEWS/train_aug.csv',names = ["label","title","text"])
    test_data = pd.read_csv('./dataset/AG_NEWS/test.csv',names = ["label","title","text"])
    if args.num_labeled == 100:
        data2 = data.sample(frac= 0.3, random_state=2333)
    if args.num_labeled == 1000:
        data2 = data.sample(frac= 0.3, random_state=2333)
    if args.num_labeled == 10000:
        data2 = data.sample(frac= 0.4, random_state=2333)


    data2['train_id'] = data2['title'] + data2['text']
    data2['train_idaug'] = data2['title'] + data2['train_idaug']

    data2['train_id'] = data2['train_id'].apply(tokenizen)
    data2['train_idaug'] = data2['train_idaug'].apply(tokenizen)
    #
#    data2["new"] = [[data2['train_id'][i]] + [data2["train_idaug"][i]] for i in range(360)]

    data2["train_orau"] = data2['train_id'] + data2["train_idaug"]

    test_data['train_id'] = test_data['title'] + " " + test_data["text"]
    test_data['train_id'] = test_data['train_id'].apply(tokenizen)




    def modifylabel(row):
        return row - 1

    data2["label"] = data2["label"].apply(modifylabel)
    test_data["label"] = test_data["label"].apply(modifylabel)

    # train_dataaug = data2['train_idaug'].sample(20000,random_state = 2333).values.tolist()
    # train_data_aug = torch.tensor(train_dataaug, dtype=torch.long)

    train_data, dev_data, train_label, dev_label = train_test_split(data2['train_id'].values,
                                                                          data2['label'].values,
                                                                          test_size=int(0.8 * args.num_labeled),
                                                                          train_size=args.num_labeled + args.num_unlabeled,
                                                                          random_state=2333)


    testdata, testlabel = torch.tensor(test_data['train_id'] , dtype=torch.long), torch.tensor(test_data['label'] , dtype=torch.long)

    #
    # # (self, dataset, args, model, mode, train_data, train_label, test_data, test_label):

    train_labeled_dataset = Base_dataset(args,model =args.model, mode="labeled",
                                                  train_data=train_data,test_data=dev_data, train_label=train_label, test_label=dev_label)
    train_unlabeled_dataset =Base_dataset(args,model =args.model, mode="unlabeled",
                                                   train_data=train_data,test_data=dev_data, train_label=train_label, test_label=dev_label)
    dev_dataset =Base_dataset(args,model = args.model,mode="dev",
                                        train_data=train_data,test_data=test_data, train_label=train_label, test_label=dev_label)
    test_dataset = Base_dataset(args, model=args.model, mode="test",
                                train_data=testdata, test_data=testdata, train_label=testlabel,
                                test_label=testlabel)

    return train_labeled_dataset, train_unlabeled_dataset, dev_dataset, test_dataset


def get_Yelp(args):
    pretrained_weights = 'bert-base-cased'

    tokenizer_class = BertTokenizer

    tokenizer = tokenizer_class.from_pretrained(pretrained_weights, do_lower_case=False)

    def tokenizen(sent):

        # sent = tokenizer.encode(sent)
        #
        # return  sent[:lg - 2]  + [0] * (lg - len(sent[:lg - 2]) - 2)
        encoded_dict = tokenizer(sent, max_length=args.max_len, padding='max_length', truncation=True, )
        return encoded_dict['input_ids'],  encoded_dict['attention_mask']



    data = pd.read_csv('./dataset/Yelp/train.csv',names = ["label","text"])
    test_data = pd.read_csv('./dataset/Yelp/test.csv',names = ["label","text"])
    if args.num_labeled == 100:
        data2 = data.sample(frac= 0.04, random_state=args.seed)
        logger.info(f"frac:0.04")
    if args.num_labeled == 1000:
        data2 = data.sample(frac= 0.08, random_state=args.seed)
        logger.info(f"frac:0.04")
    if args.num_labeled == 10000:
        data2 = data.sample(frac= 0.2, random_state=args.seed)
        logger.info(f"frac:0.2")

    data2['train_id'] = data2['text'].apply(tokenizen)
    test_data['train_id'] = test_data['text'].apply(tokenizen)



    def modifylabel(row):
        return row - 1

    data2["label"] = data2["label"].apply(modifylabel)
    test_data["label"] = test_data["label"].apply(modifylabel)


    train_data, dev_data, train_label, dev_label = train_test_split(data2['train_id'].values,
                                                                              data2['label'].values, test_size=int(0.8 * args.num_labeled),train_size=args.num_labeled+args.num_unlabeled,
                                                                              random_state=args.seed)
    testdata, testlabel = torch.tensor(test_data['train_id'] , dtype=torch.long), torch.tensor(test_data['label'] , dtype=torch.long)

    #
    # # (self, dataset, args, model, mode, train_data, train_label, test_data, test_label):

    train_labeled_dataset = Base_dataset(args,model =args.model, mode="labeled",
                                                  train_data=train_data,test_data=dev_data, train_label=train_label, test_label=dev_label)
    train_unlabeled_dataset =Base_dataset(args,model =args.model, mode="unlabeled",
                                                   train_data=train_data,test_data=dev_data, train_label=train_label, test_label=dev_label)
    dev_dataset =Base_dataset(args,model = args.model,mode="dev",
                                        train_data=train_data,test_data=dev_data, train_label=train_label, test_label=dev_label)
    test_dataset = Base_dataset(args, model=args.model, mode="test",
                                train_data=testdata, test_data=testdata, train_label=testlabel,
                                test_label=testlabel)

    return train_labeled_dataset, train_unlabeled_dataset, dev_dataset, test_dataset



DATASET_GETTERS = {'AGNews':get_AGNews,
                   "Yelp":get_Yelp
                   }
