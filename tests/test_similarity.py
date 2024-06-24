import unittest
import sys

sys.path.append('..')

from text2vec2onnx import SentenceModel, cos_sim, semantic_search

embedder = SentenceModel(model_dir_path='/data/text2vec-base-chinese-onnx',device_id=0)

case_same_keywords = [['飞行员没钱买房怎么办？', '父母没钱买房子', False],
                      ['聊天室都有哪些好的', '聊天室哪个好', True],
                      ['不锈钢上贴的膜怎么去除', '不锈钢上的胶怎么去除', True],
                      ['动漫人物的口头禅', '白羊座的动漫人物', False]]

case_categories_corresponding_pairs = [['从广州到长沙在哪里定高铁票', '在长沙哪里坐高铁回广州？', False],
                                       ['请问现在最好用的听音乐软件是什么啊', '听歌用什么软件比较好', True],
                                       ['谁有吃过完美的产品吗？如何？', '完美产品好不好', True],
                                       ['朱熹是哪个朝代的诗人', '朱熹是明理学的集大成者，他生活在哪个朝代', True],
                                       ['这是哪个奥特曼？', '这是什么奥特曼...', True],
                                       ['网上找工作可靠吗', '网上找工作靠谱吗', True],
                                       ['你们都喜欢火影忍者里的谁啊', '火影忍者里你最喜欢谁', True]]

long_a = '你们都喜欢火影忍者里的谁啊，你说的到底是谁？看Bert里面extract_features.py这个文件，可以得到类似预训练的词向量组成的句子表示，' \
         '类似于Keras里面第一步Embedding层。以题主所说的句子相似度计算为例，只需要把两个句子用分隔符隔开送到bert的输入（首位加特殊标记符' \
         'CLS的embedding），然后取bert输出中和CLS对应的那个vector（记为c）进行变换就可以了。原文中提到的是多分类任务，给出的输出变换是' \
         '）就可以了。至于题主提到的句向量表示，上文中提到的向量c即可一定程度表' \
         '示整个句子的语义，原文中有提到“ The final hidden state (i.e., output of Transformer) corresponding to this token ' \
         'is used as the aggregate sequence representation for classification tasks.”' \
         '这句话中的“this token”就是CLS位。补充：除了直接使用bert的句对匹配之外，还可以只用bert来对每个句子求embedding。之后再通过向' \
         'Siamese Network这样的经典模式去求相似度也可以'

long_b = '你说的到底是谁？看Bert里面extract_features.py这个文件，可以得到类似预训练的词向量组成的句子表示，' \
         '类似于Keras里面第一步Embedding层。以题主所说的句子相似度计算为例，只需要把两个句子用分隔符隔开送到bert的输入（首位加特殊标记符' \
         'CLS的embedding），然后取bert输出中和CLS对应的那个vector（记为c）进行变换就可以了。原文中提到的是多分类任务，给出的输出变换是' \
         '）就可以了。至于题主提到的句向量表示，上文中提到的向量c即可一定程度表'

corpus = [
    "谢谢观看 下集再见",
    "感谢您的观看",
    "请勿模仿",
    "记得订阅我们的频道哦",
    "The following are sentences in English.",
    "Thank you. Bye-bye.",
    "It's true",
    "I don't know.",
    "Thank you for watching!",
]


def sbert_sim_score(str_a, str_b):
    a_emb = embedder.encode(str_a)
    b_emb = embedder.encode(str_b)
    return cos_sim(a_emb, b_emb).item()


def apply_sbert_case(cases):
    for line in cases:
        q1 = line[0]
        q2 = line[1]
        a = line[2]

        s = sbert_sim_score(q1, q2)
        print(f'q1:{q1}, q2:{q2}, expect:{a}, actual:{s:.4f}')


class SimTextCase(unittest.TestCase):
    def test_bert_sim_batch(self):
        apply_sbert_case(case_same_keywords)
        apply_sbert_case(case_categories_corresponding_pairs)

    def test_longtext(self):
        r = sbert_sim_score(long_a, long_b)
        print(r)
        self.assertEqual(abs(r - 0.872) < 0.2, True)

    def test_query(self):
        corpus_embeddings = embedder.encode(corpus)
        query_embedding = embedder.encode("Thank you. Bye.")
        hits = semantic_search(query_embedding, corpus_embeddings, top_k=1)
        print(hits)
        self.assertEqual(hits[0][0]["corpus_id"], 5)
        self.assertEqual(hits[0][0]["score"] > 0.9, True)
        query_embedding = embedder.encode("你干啥呢")
        hits = semantic_search(query_embedding, corpus_embeddings, top_k=1)
        print(hits)
        self.assertEqual(hits[0][0]["score"] <0.3, True)
        query_embedding = embedder.encode("感谢您的收听")
        hits = semantic_search(query_embedding, corpus_embeddings, top_k=1)
        print(hits)
        self.assertEqual(hits[0][0]["corpus_id"], 1)
        self.assertEqual(hits[0][0]["score"] > 0.8, True)

    def test_query_batch(self):
        queries = [
            'Thank you. Bye.',
            '你干啥呢',
            '感谢您的收听']

        for query in queries:
            query_embedding = embedder.encode(query)
            corpus_embeddings = embedder.encode(corpus)

            hits = semantic_search(query_embedding, corpus_embeddings, top_k=1)
            print("\n\n======================\n\n")
            print("Query:", query)
            print("\nTop 5 most similar sentences in corpus:")
            hits = hits[0]  # Get the hits for the first query
            for hit in hits:
                print(corpus[hit['corpus_id']], "(Score: {:.4f})".format(hit['score']))


if __name__ == '__main__':
    unittest.main()
