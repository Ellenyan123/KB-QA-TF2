import re
from collections import defaultdict
import jieba.posseg
import numpy as np
import tensorflow as tf


def tokenizer(filename, stop_words):
    texts = []
    with open(filename, 'r', encoding='utf-8') as f:
        for line in f.readlines():
            texts.append([token for token, _ in jieba.posseg.cut(line.rstrip())
                          if token not in stop_words])

    # remove words that appear only once
    frequency = defaultdict(int)
    for text in texts:
        for token in text:
            frequency[token] += 1
    texts = [[token for token in text if frequency[token] > 1] for text in texts]
    return texts


def load_embedding(filename):
    embeddings = []
    word2idx = defaultdict(list)
    with open(filename, mode="r", encoding="utf-8") as rf:
        for line in rf:
            arr = line.split(" ")
            embedding = [float(val) for val in arr[1: -1]]
            word2idx[arr[0]] = len(word2idx)
            embeddings.append(embedding)

    return embeddings, word2idx


def words_list2index(words_list, word2idx,  max_len):
    """
    word list to indexes in embeddings.
    """
    unknown = word2idx.get("UNKNOWN", 0)
    num = word2idx.get("NUM", len(word2idx))
    index = [unknown] * max_len
    i = 0
    for word in words_list:
        if word in word2idx:
            index[i] = word2idx[word]
        else:
            if re.match("\d+", word):
                index[i] = num
            else:
                index[i] = unknown
        if i >= max_len - 1:
            break
        i += 1
    return index


def load_data(knowledge_file, filename, word2idx, stop_words, sim_ixs, max_len):
    knowledge_texts = tokenizer(knowledge_file, stop_words)
    train_texts = tokenizer(filename, stop_words)

    question_num = 0
    tmp = []
    questions, answers, labels = [], [], []  #问题、答案（正确答案，错误答案）、标签（正确答案）
    with open(filename, mode="r", encoding="utf-8") as f: #从train文件中抽取出B、Q、R、W、W、W
        for i, line in enumerate(f):
            if i % 6 == 0: #B、Q、R、W、W、W---> B
                question_num += 1
                for j in sim_ixs[i//6]: #6//6=1, sim_ixs[1] =[ ?]--其中为与第i个背景+Q相似knowledge的序号
                    tmp.extend(knowledge_texts[j]) #tmp 先将相似的knowledge_texts加进来
                tmp.extend(train_texts[i]) #然后把B加进来
            elif i % 6 == 1:
                tmp.extend(train_texts[i]) #然后把Q加进来
                #tmp = [knowledge_texts[多个], B, Q]每个Q对应一个一维长向量
                #t = [tmp--id化]
                t = words_list2index(tmp, word2idx, max_len) #word--id
            else:
                if line[0] == 'R':
                    questions.append(t) #问题[t]
                    #答案[train_texts[i]的id化]
                    answers.append(words_list2index(train_texts[i], word2idx, max_len))
                    #labels=[1]
                    labels.append(1)
                elif line[0] == 'W':
                    questions.append(t) #问题[t, t, t, t]
                    #答案[正确train_texts[i]的id化, 错误train_texts[i]的id化，错误train_texts[i]的id化，错误train_texts[i]的id化]
                    answers.append(words_list2index(train_texts[i], word2idx, max_len))
                     #labels=[1, 0, 0, 0]
                    labels.append(0)
                if i % 6 == 5:
                    tmp.clear()
    return questions, answers, labels, question_num


def training_batch_iter(questions, answers, labels, question_num, batch_size):
    """
    :return q + -
    """
    batch_num = int(question_num / batch_size) + 1
    for batch in range(batch_num):
        # for each batch
        ret_questions, true_answers, false_answers = [], [], []
        for i in range(batch * batch_size, min((batch + 1) * batch_size, question_num)):
            # for each question(4 line)
            ix = i * 4
            ret_questions.extend([questions[ix]] * 3) #三个相同的
            for j in range(ix, ix + 4):
                if labels[j]:
                    true_answers.extend([answers[j]] * 3) #[[],[],[]] 三个相同的向量
                else:
                    false_answers.append(answers[j]) #[[],[],[]] 3个不同的
                    #print(tf.constant(false_answers).shape)
        yield np.array(ret_questions), np.array(true_answers), np.array(false_answers)


def testing_batch_iter(questions, answers, question_num, batch_size):
    batch_num = int(question_num / batch_size) + 1
    questions, answers = np.array(questions), np.array(answers)
    for batch in range(batch_num):
        start_ix = batch * batch_size * 4
        end_ix = min((batch + 1) * batch_size * 4, len(questions))
        yield questions[start_ix:end_ix], answers[start_ix:end_ix]


