import jieba.posseg
import os
import codecs
import pickle
from gensim import corpora, models, similarities
from data_util import tokenizer


def generate_dic_and_corpus(knowledge_file, file_name, stop_words):
    knowledge_texts = tokenizer(knowledge_file, stop_words)
    train_texts = tokenizer(file_name, stop_words)

    dictionary = corpora.Dictionary(knowledge_texts + train_texts)  # dictionary of knowledge and train data
    dictionary.save(os.path.join('tmp/dictionary.dict'))

    corpus = [dictionary.doc2bow(text) for text in knowledge_texts]  # corpus of knowledge
    corpora.MmCorpus.serialize('tmp/knowledge_corpus.mm', corpus)


def topk_sim_ix(file_name, stop_words, k):
    sim_path = "tmp/" + file_name[5:-4]
    if os.path.exists(sim_path):
        with open(sim_path, "rb") as f:
            sim_ixs = pickle.load(f)
        return sim_ixs

    # load dictionary and corpus
    dictionary = corpora.Dictionary.load("tmp/dictionary.dict")  # dictionary of knowledge and train data
    corpus = corpora.MmCorpus("tmp/knowledge_corpus.mm")  # corpus of knowledge

    # build Latent Semantic Indexing model
    lsi = models.LsiModel(corpus, id2word=dictionary, num_topics=10)  # initialize an LSI transformation

    # similarity
    index = similarities.MatrixSimilarity(lsi[corpus])  # transform corpus to LSI space and index it
    sim_ixs = []  # topk related knowledge index of each question
    with open(file_name, 'r', encoding='utf-8') as f:
        tmp = []  # background and question
        for i, line in enumerate(f):
            if i % 6 == 0:
                tmp.extend([token for token, _ in jieba.posseg.cut(line.rstrip()) if token not in stop_words])
            if i % 6 == 1:
                tmp.extend([token for token, _ in jieba.posseg.cut(line.rstrip()) if token not in stop_words])
                vec_lsi = lsi[dictionary.doc2bow(tmp)]  # convert the query to LSI space
                sim_ix = index[vec_lsi]  # perform a similarity query against the corpus
                sim_ix = [i for i, j in sorted(enumerate(sim_ix), key=lambda item: -item[1])[:k]]  # topk index
                sim_ixs.append(sim_ix)
                tmp.clear()
    with open(sim_path, "wb") as f:
        pickle.dump(sim_ixs, f)
    return sim_ixs


# module test
if __name__ == '__main__':
    stop_words_ = codecs.open("data/stop_words.txt", 'r', encoding='utf8').readlines()
    stop_words_ = [w.strip() for w in stop_words_]
    generate_dic_and_corpus("data/knowledge.txt", "data/train.txt", stop_words_)
    res = topk_sim_ix("data/train.txt", stop_words_, 5)
    print(len(res))

    knowledge_file = "data/knowledge.txt"
    file_name = "data/train.txt",
    knowledge_texts = tokenizer(knowledge_file, stop_words_)
    knowledge_texts[0]
    #['地球', '宇宙', '中', '一颗', '行星', '运动', '规律']
    dictionary.doc2bow(knowledge_texts[0])
    #"""Convert `document` into the bag-of-words (BoW) format = list of `(token_id, token_count)` tuples.
    #[(0, 1), (1, 1), (2, 1), (3, 1), (4, 1), (5, 1), (6, 1)]
    dictionary.doc2bow(knowledge_texts[10])
    #['蟹', '状', '星云', '金牛座', '一团', '膨胀', '气体']
    #[(27, 1), (51, 1), (53, 1), (54, 1), (55, 1), (56, 1), (57, 1)]



