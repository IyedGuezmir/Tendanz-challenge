from app.src.retrieval import rag, reranker
from .llm_chain import LLMChain
from .llm_chain_decomp import LLMChainWithDecomposition
from .rag_fusion_chain import RAGFusionChain


chain = LLMChain(
    rag=rag,
    reranker=reranker,
)
decomp_chain = LLMChainWithDecomposition(rag=rag, reranker=reranker)
rag_fusion_chain = RAGFusionChain(rag=rag)


__all__ = ["chain", "LLMChain", "decomp_chain", "rag_fusion_chain"]