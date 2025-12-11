import numpy as np
import os
import json
import argparse
import torch

# ----------------------------
# Device (GPU nếu có)
# ----------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("🔥 Using device:", device)

parser = argparse.ArgumentParser()
parser.add_argument('--in_path', type=str, default="../data")
parser.add_argument('--out_path', type=str, default="prepro_data")
args = parser.parse_args()

in_path = args.in_path
out_path = args.out_path
case_sensitive = False

char_limit = 16
sent_limit = 25
word_size = 100

train_distant_file_name = os.path.join(in_path, 'train_distant.json')
train_annotated_file_name = os.path.join(in_path, 'train_annotated.json')
dev_file_name = os.path.join(in_path, 'dev.json')
test_file_name = os.path.join(in_path, 'test.json')

rel2id = json.load(open(os.path.join(out_path, 'rel2id.json'), "r"))
id2rel = {v: u for u, v in rel2id.items()}
json.dump(id2rel, open(os.path.join(out_path, 'id2rel.json'), "w"))

fact_in_train = set([])
fact_in_dev_train = set([])

word2id_name = "word2id.json"


def sents_2_idx(sents, word2id):
    sents_idx = []
    start_idx = 0
    for i, sent in enumerate(sents[:sent_limit]):
        sents_idx.append(list(range(start_idx, start_idx + len(sent))))
        start_idx += len(sent)
    return sents_idx


def init(data_file_name, rel2id, max_length=512, is_training=True, suffix=''):
    ori_data = json.load(open(data_file_name))

    data = []
    word2id = json.load(open(os.path.join(out_path, word2id_name)))
    ner2id = json.load(open(os.path.join(out_path, "ner2id.json")))

    for i in range(len(ori_data)):
        Ls = [0]
        L = 0
        for x in ori_data[i]['sents']:
            L += len(x)
            Ls.append(L)

        vertexSet = ori_data[i]['vertexSet']
        for j in range(len(vertexSet)):
            for k in range(len(vertexSet[j])):
                vertexSet[j][k]['sent_id'] = int(vertexSet[j][k]['sent_id'])
                sent_id = vertexSet[j][k]['sent_id']
                dl = Ls[sent_id]
                pos1 = vertexSet[j][k]['pos'][0]
                pos2 = vertexSet[j][k]['pos'][1]
                vertexSet[j][k]['pos'] = (pos1 + dl, pos2 + dl)
                vertexSet[j][k]["type"] = ner2id[vertexSet[j][k]["type"]]

        ori_data[i]['vertexSet'] = vertexSet

        item = {}
        item['vertexSet'] = vertexSet
        labels = ori_data[i].get('labels', [])

        train_triple = set([])
        new_labels = []

        for label in labels:
            rel = label['r']
            assert rel in rel2id
            label['r'] = rel2id[label['r']]

            train_triple.add((label['h'], label['t']))

            if suffix == '_train':
                for n1 in vertexSet[label['h']]:
                    for n2 in vertexSet[label['t']]:
                        fact_in_dev_train.add((n1['name'], n2['name'], rel))

            if is_training:
                for n1 in vertexSet[label['h']]:
                    for n2 in vertexSet[label['t']]:
                        fact_in_train.add((n1['name'], n2['name'], rel))
            else:
                label['intrain'] = False
                label['indev_train'] = False

                for n1 in vertexSet[label['h']]:
                    for n2 in vertexSet[label['t']]:
                        if (n1['name'], n2['name'], rel) in fact_in_train:
                            label['intrain'] = True

                        if suffix in ['_dev', '_test']:
                            if (n1['name'], n2['name'], rel) in fact_in_dev_train:
                                label['indev_train'] = True

            new_labels.append(label)

        item['labels'] = new_labels
        item['title'] = ori_data[i]['title']

        na_triple = []
        for j in range(len(vertexSet)):
            for k in range(len(vertexSet)):
                if j != k and (j, k) not in train_triple:
                    na_triple.append((j, k))

        item['na_triple'] = na_triple
        item['Ls'] = Ls
        item['sents'] = ori_data[i]['sents']
        item['sents_idx'] = sents_2_idx(ori_data[i]['sents'], word2id)
        data.append(item)

    print("\nSaving JSON...")
    prefix = "train" if is_training else "dev"
    json.dump(data, open(os.path.join(out_path, prefix + suffix + '.json'), "w"))

    print("Allocating tensors on:", device)
    sen_tot = len(ori_data)

    sen_word = torch.zeros((sen_tot, max_length), dtype=torch.long, device=device)
    sen_pos = torch.zeros((sen_tot, max_length), dtype=torch.long, device=device)
    sen_ner = torch.zeros((sen_tot, max_length), dtype=torch.long, device=device)
    sen_char = torch.zeros((sen_tot, max_length, char_limit), dtype=torch.long, device=device)

    char2id = json.load(open(os.path.join(out_path, "char2id.json")))
    word2id = json.load(open(os.path.join(out_path, word2id_name)))

    for i in range(len(ori_data)):
        item = ori_data[i]
        words = []
        for sent in item['sents']:
            words += sent

        for j, w in enumerate(words):
            word = w.lower()

            if j < max_length:
                sen_word[i][j] = word2id.get(word, word2id['UNK'])

            for c_idx, ch in enumerate(list(word)):
                if c_idx >= char_limit:
                    break
                sen_char[i][j][c_idx] = char2id.get(ch, char2id['UNK'])

        for j in range(j + 1, max_length):
            sen_word[i][j] = word2id['BLANK']

        vertexSet = item['vertexSet']
        for idx, vertex in enumerate(vertexSet, 1):
            for v in vertex:
                sen_pos[i][v['pos'][0]:v['pos'][1]] = idx
                sen_ner[i][v['pos'][0]:v['pos'][1]] = v['type']

    print("Saving .npy tensors...")
    np.save(os.path.join(out_path, prefix + suffix + '_word.npy'), sen_word.cpu().numpy())
    np.save(os.path.join(out_path, prefix + suffix + '_pos.npy'), sen_pos.cpu().numpy())
    np.save(os.path.join(out_path, prefix + suffix + '_ner.npy'), sen_ner.cpu().numpy())
    np.save(os.path.join(out_path, prefix + suffix + '_char.npy'), sen_char.cpu().numpy())

    print("✔ Finished saving\n")


# ----------------------------
# RUN
# ----------------------------
if os.path.exists(train_annotated_file_name):
    init(train_annotated_file_name, rel2id, max_length=512, is_training=False, suffix='_train')
else:
    print("train annotated file not exist")

if os.path.exists(dev_file_name):
    init(dev_file_name, rel2id, max_length=512, is_training=False, suffix='_dev')
else:
    print("dev file not exist")

if os.path.exists(test_file_name):
    init(test_file_name, rel2id, max_length=512, is_training=False, suffix='_test')
else:
    print("test file not exist")
