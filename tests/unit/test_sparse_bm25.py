"""中文 BM25 稀疏向量单元测试（离线，不依赖 API 与 Docker）。"""

from pathlib import Path

from app.retrieval.sparse_bm25 import ChineseBM25SparseEmbeddings, SparseVector

CORPUS = [
    "星火科技考勤管理制度规定，旷工扣款为当日工资的200%。",
    "员工请假须提前一个工作日在OA提交申请。",
    "薪酬管理制度编号为 XH-HR-2026-002，含职级薪酬表。",
    "园区食堂不提供，公司每月发放餐补600元。",
]


def test_fit_and_embed_shapes():
    model = ChineseBM25SparseEmbeddings.fit(CORPUS)
    vecs = model.embed_documents(CORPUS)
    assert len(vecs) == 4
    assert all(isinstance(v, SparseVector) for v in vecs)
    assert all(len(v.indices) == len(v.values) > 0 for v in vecs)
    assert isinstance(model.embed_query("旷工扣款"), SparseVector)


def test_exact_term_scores_higher():
    """含精确编号的文档在查询该编号时点积应显著高于其他文档。"""
    model = ChineseBM25SparseEmbeddings.fit(CORPUS)
    q = model.embed_query("XH-HR-2026-002")

    def dot(v: SparseVector, w: SparseVector) -> float:
        w_map = dict(zip(w.indices, w.values))
        return sum(val * w_map.get(idx, 0.0) for idx, val in zip(v.indices, v.values))

    scores = [dot(model.embed_documents([d])[0], q) for d in CORPUS]
    assert scores[2] > 0, "含编号的文档应得分 > 0"
    assert scores[2] == max(scores), "编号查询应命中含编号的文档"
    assert scores[0] == 0 and scores[1] == 0, "不含编号的文档得分应为 0"


def test_unknown_word_ignored():
    model = ChineseBM25SparseEmbeddings.fit(CORPUS)
    v = model.embed_query("完全没见过的词XYZ")
    assert len(v.indices) == 0  # 词表外的词不产生维度


def test_save_load_roundtrip(tmp_path: Path):
    model = ChineseBM25SparseEmbeddings.fit(CORPUS)
    path = tmp_path / "bm25.json"
    model.save(path)
    loaded = ChineseBM25SparseEmbeddings.load(path)
    assert loaded.vocab == model.vocab
    assert loaded.idf == model.idf
    assert loaded.avgdl == model.avgdl
    q = model.embed_query("旷工")
    q2 = loaded.embed_query("旷工")
    assert q.indices == q2.indices
    assert q.values == q2.values
