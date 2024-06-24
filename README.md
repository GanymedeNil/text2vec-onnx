# text2vec-onnx

本项目是 [text2vec](https://github.com/shibing624/text2vec) 项目的 onnxruntime 推理版本，实现了向量获取和文本匹配搜索。为了保证项目的轻量，只使用了 `onnxruntime` 、 `tokenizers` 和 `numpy` 三个库。

主要在 [GanymedeNil/text2vec-base-chinese-onnx](https://huggingface.co/GanymedeNil/text2vec-base-chinese-onnx) 模型上进行测试，理论上支持 BERT 系列模型。

## 安装

### CPU 版本
```bash
pip install text2vec2onnx[cpu]
```
### GPU 版本
```bash
pip install text2vec2onnx[gpu]
```

## 使用

### 模型下载
以下载 GanymedeNil/text2vec-base-chinese-onnx 为例，下载模型到本地。

- huggingface 模型下载
```bash
huggingface-cli download --resume-download GanymedeNil/text2vec-base-chinese-onnx --local-dir text2vec-base-chinese-onnx
```

### 向量获取

```python
from text2vec2onnx import SentenceModel
embedder = SentenceModel(model_dir_path='local-dir')
emb = embedder.encode("你好")
```

### 文本匹配搜索

```python
from text2vec2onnx import SentenceModel, semantic_search

embedder = SentenceModel(model_dir_path='local-dir')

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
corpus_embeddings = embedder.encode(corpus)

queries = [
    'Thank you. Bye.',
    '你干啥呢',
    '感谢您的收听']

for query in queries:
    query_embedding = embedder.encode(query)
    hits = semantic_search(query_embedding, corpus_embeddings, top_k=1)
    print("\n\n======================\n\n")
    print("Query:", query)
    print("\nTop 5 most similar sentences in corpus:")
    hits = hits[0]  # Get the hits for the first query
    for hit in hits:
        print(corpus[hit['corpus_id']], "(Score: {:.4f})".format(hit['score']))


```

## License
[Appache License 2.0](LICENSE)

## References
- [text2vec](https://github.com/shibing624/text2vec)


## Buy me a coffee
<div align="center">
<a href="https://www.buymeacoffee.com/ganymedenil" target="_blank"><img src="https://cdn.buymeacoffee.com/buttons/v2/default-yellow.png" alt="Buy Me A Coffee" style="height: 60px !important;width: 217px !important;" ></a>
</div>
<div align="center">
<img height="360" src="https://user-images.githubusercontent.com/9687786/224522468-eafb7042-d000-4799-9d16-450489e8efa4.png"/>
<img height="360" src="https://user-images.githubusercontent.com/9687786/224522477-46f3e80b-0733-4be9-a829-37928260038c.png"/>
</div>