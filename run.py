"""
RAG 流水线 CLI 入口
"""

import argparse
import json

from pipeline import RAGPipeline, RAGPipelineConfig


def main():
    parser = argparse.ArgumentParser(description="模块化 RAG 流水线")

    # 配置参数
    parser.add_argument("--config", type=str, help="配置文件路径 (JSON)")
    parser.add_argument("--embedder", type=str, default="sentence_transformer",
                        choices=["openai", "sentence_transformer", "clip"],
                        help="向量化策略")
    parser.add_argument("--embedder-model", type=str, default="all-MiniLM-L6-v2",
                        help="嵌入模型名称")
    parser.add_argument("--vectorstore", type=str, default="chroma",
                        choices=["chroma", "faiss"],
                        help="向量存储策略")
    parser.add_argument("--collection", type=str, default="rag_collection",
                        help="集合名称")
    parser.add_argument("--retriever", type=str, default="similarity",
                        choices=["similarity", "mmr"],
                        help="检索策略")
    parser.add_argument("--top-k", type=int, default=5, help="返回结果数量")
    parser.add_argument("--fetch-k", type=int, default=20, help="MMR 候选数量")
    parser.add_argument("--lambda-mult", type=float, default=0.5,
                        help="MMR 多样性参数 (0=最大多样性, 1=最大相关性)")
    parser.add_argument("--generator", type=str, default="openai",
                        choices=["openai", "anthropic"],
                        help="生成策略")
    parser.add_argument("--model", type=str, default="gpt-4o-mini",
                        help="生成模型")

    # 操作参数
    parser.add_argument("--action", type=str, required=True,
                        choices=["chunk", "embed", "store", "retrieve", "generate", "run"],
                        help="执行的操作")
    parser.add_argument("--files", type=str, nargs="+", help="输入文件列表")
    parser.add_argument("--query", type=str, help="查询文本")
    parser.add_argument("--texts", type=str, nargs="+", help="文本列表 (用于 embed/store)")

    args = parser.parse_args()

    # 构建配置
    config = RAGPipelineConfig(
        embedder_type=args.embedder,
        embedder_model=args.embedder_model,
        vectorstore_type=args.vectorstore,
        collection_name=args.collection,
        retriever_type=args.retriever,
        top_k=args.top_k,
        fetch_k=args.fetch_k,
        lambda_mult=args.lambda_mult,
        generator_type=args.generator,
        generator_model=args.model,
    )

    pipeline = RAGPipeline(config)

    if args.action == "chunk":
        result = pipeline.chunk(files=args.files)
        print(f"分块完成: {len(result.chunks)} 个块, {result.total_nodes} 个节点")
        for i, chunk in enumerate(result.chunks[:3]):
            print(f"  块 {i}: {chunk.content[:50]}...")

    elif args.action == "embed":
        if args.texts:
            result = pipeline.embed(texts=args.texts)
        else:
            result = pipeline.embed()
        print(f"向量化完成: {len(result.embeddings)} 个向量, 维度 {result.dimension}")

    elif args.action == "store":
        pipeline.store(texts=args.texts)
        print("存储完成")

    elif args.action == "retrieve":
        if not args.query:
            print("错误: --query required for retrieve")
            return
        result = pipeline.retrieve(args.query)
        print(f"检索到 {len(result.chunks)} 个结果:")
        for i, chunk in enumerate(result.chunks):
            print(f"  [{i}] {chunk.content[:80]}...")

    elif args.action == "generate":
        if not args.query:
            print("错误: --query required for generate")
            return
        result = pipeline.generate(args.query)
        print(f"生成结果:\n{result.response}")

    elif args.action == "run":
        if not args.query:
            print("错误: --query required for run")
            return
        result = pipeline.run(args.query)
        print(result)


if __name__ == "__main__":
    main()
