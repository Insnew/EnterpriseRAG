"""中文 BM25 稀疏向量生成器（自实现，jieba 分词）。

为什么自实现而不直接用 Qdrant 内置 FastEmbed 的 BM42：BM42 是纯英文模型，
对中文分词无效。自实现代码量小、公式透明，面试可逐行讲清。

BM25 核心公式（面试高频）：
    score(doc, query) = Σ_{t ∈ query∩doc} IDF(t) · tf_norm(t, doc)
    IDF(t)     = ln((N - df + 0.5) / (df + 0.5) + 1)      # 词越稀有权重越高
    tf_norm    = tf / (tf + k1·(1 - b + b·dl/avgdl))      # 词频饱和 + 长度归一

向量化策略（业界常用近似）：
    入库：每个 chunk → {词id: tf_norm·IDF} 稀疏向量
    查询：问题     → {词id: 词频·IDF} 稀疏向量
    点积 ≈ BM25 打分（省略查询侧长度归一）
"""

import json
import math
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import jieba

K1 = 1.5  # 词频饱和参数
B = 0.75  # 长度归一参数


@dataclass
class SparseVector:
    """稀疏向量：仅非零维度存 (词id, 权重)。

    langchain-qdrant 对 sparse_embedding 采用鸭子类型协议：
    只需 embed_query/embed_documents 返回带 indices/values 的对象即可。
    """

    indices: list[int]
    values: list[float]

# 高频无实义词，过滤后降低噪音、缩小词表
STOPWORDS = {
    "的", "了", "是", "在", "和", "与", "或", "及", "等", "对", "中", "为", "上", "下",
    "有", "不", "就", "也", "都", "而", "但", "被", "把", "将", "以", "于", "其", "之",
    "可", "吗", "呢", "啊", "哦", "嗯", "吧", "呀", "这", "那", "要", "会", "能", "还",
    "则", "由", "按", "须", "应", "并", "且", "即", "该", "此", "每", "各", "一", "个",
}


class ChineseBM25SparseEmbeddings:
    """实现 LangChain SparseEmbeddings 协议：embed_documents / embed_query → SparseVector。"""

    def __init__(self, vocab: dict[str, int], idf: dict[str, float], avgdl: float, n_docs: int):
        self.vocab = vocab   # 词 → 词表 id
        self.idf = idf       # 词 → 逆文档频率
        self.avgdl = avgdl   # 语料平均文档长度（词数）
        self.n_docs = n_docs

    # ---------- 分词 ----------

    @staticmethod
    def _tokenize(text: str) -> list[str]:
        tokens = jieba.lcut(text)
        return [t.strip() for t in tokens if t.strip() and t.strip() not in STOPWORDS]

    # ---------- 训练（入库时用全量语料 fit） ----------

    @classmethod
    def fit(cls, corpus: list[str]) -> "ChineseBM25SparseEmbeddings":
        if not corpus:
            raise ValueError("语料为空，无法训练 BM25")
        tokenized = [cls._tokenize(t) for t in corpus]
        df: Counter[str] = Counter()
        for toks in tokenized:
            df.update(set(toks))  # 文档频率：词出现在几篇文档里
        n = len(corpus)
        vocab = {w: i for i, w in enumerate(sorted(df))}
        idf = {w: math.log((n - c + 0.5) / (c + 0.5) + 1) for w, c in df.items()}
        avgdl = sum(len(t) for t in tokenized) / n
        return cls(vocab=vocab, idf=idf, avgdl=avgdl, n_docs=n)

    # ---------- 向量化 ----------

    def _doc_vector(self, tokens: list[str], dl: int) -> SparseVector:
        tf = Counter(tokens)
        indices, values = [], []
        for w, c in tf.items():
            if w not in self.vocab:
                continue
            tf_norm = c / (c + K1 * (1 - B + B * dl / self.avgdl))
            indices.append(self.vocab[w])
            values.append(tf_norm * self.idf[w])
        return SparseVector(indices=indices, values=values)

    def embed_documents(self, texts: list[str]) -> list[SparseVector]:
        return [
            self._doc_vector(self._tokenize(t), max(len(self._tokenize(t)), 1)) for t in texts
        ]

    def embed_query(self, text: str) -> SparseVector:
        tf = Counter(self._tokenize(text))
        indices, values = [], []
        for w, c in tf.items():
            if w not in self.vocab:
                continue
            indices.append(self.vocab[w])
            values.append(c * self.idf[w])
        return SparseVector(indices=indices, values=values)

    # ---------- 持久化（词表/IDF 依赖语料，入库与查询必须同一份） ----------

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(
                {"vocab": self.vocab, "idf": self.idf, "avgdl": self.avgdl, "n_docs": self.n_docs},
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )

    @classmethod
    def load(cls, path: Path) -> "ChineseBM25SparseEmbeddings":
        data = json.loads(path.read_text(encoding="utf-8"))
        return cls(
            vocab=data["vocab"],
            idf=data["idf"],
            avgdl=data["avgdl"],
            n_docs=data["n_docs"],
        )


def sparse_model_path(kb_id: str) -> Path:
    """每个知识库一份词表，与 collection 一一对应。"""
    from app.core.config import PROJECT_ROOT

    return PROJECT_ROOT / "data" / "bm25" / f"{kb_id}.json"
