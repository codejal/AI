import asyncio

from api.agents.retrieval_generation import rag_pipeline
from langsmith import Client
from qdrant_client import QdrantClient

from langchain_anthropic import ChatAnthropic
from langchain_voyageai import VoyageAIEmbeddings

from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper

from ragas.metrics import IDBasedContextPrecision, IDBasedContextRecall, Faithfulness, ResponseRelevancy
from ragas.dataset_schema import SingleTurnSample

ls_client = Client()
qdrant_client = QdrantClient(url="http://localhost:6333")
ragas_llm = LangchainLLMWrapper(ChatAnthropic(model="claude-haiku-4-5-20251001"))
ragas_embeddings = LangchainEmbeddingsWrapper(VoyageAIEmbeddings(model="voyage-3"))

def ragas_faithfulness(run, example):
        sample = SingleTurnSample(
                user_input=example.inputs["question"],
                response=run.outputs["answer"],
                retrieved_contexts=run.outputs["retrieved_context"],
        )
        scorer = Faithfulness(llm=ragas_llm)
        return asyncio.run(scorer.single_turn_ascore(sample))


def ragas_response_relevancy(run, example):
        sample = SingleTurnSample(
                user_input=example.inputs["question"],
                response=run.outputs["answer"],
                retrieved_contexts=run.outputs["retrieved_context"],
        )
        scorer = ResponseRelevancy(llm=ragas_llm, embeddings=ragas_embeddings)
        return asyncio.run(scorer.single_turn_ascore(sample))


def ragas_context_precision_id_based(run, example):
        if not example.outputs.get("reference_context_ids"):
                return None
        sample = SingleTurnSample(
                retrieved_context_ids=run.outputs["retrieved_context_ids"],
                reference_context_ids=example.outputs["reference_context_ids"]
        )
        scorer = IDBasedContextPrecision()
        return asyncio.run(scorer.single_turn_ascore(sample))


def ragas_context_recall_id_based(run, example):
        if not example.outputs.get("reference_context_ids"):
                return None
        sample = SingleTurnSample(
                retrieved_context_ids=run.outputs["retrieved_context_ids"],
                reference_context_ids=example.outputs["reference_context_ids"]
        )
        scorer = IDBasedContextRecall()
        return asyncio.run(scorer.single_turn_ascore(sample))


"""
Here since we are using Ragas, we need to have 'run' and 'example' as arguments to the method
'run' — a RunTree object wrapping the pipeline execution. run.outputs is the dict your lambda returned
in our case it will be rag_pipeline's output
run.outputs = {
        "answer": answer,
        "question": question,
        "retrieved_context_ids": retrieved_context["retrieved_context_ids"],
        "retrieved_context": retrieved_context["retrieved_context"],
        "similarity_scores": retrieved_context["similarity_scores"]
}
therefore the question -> run.outputs['question']

'example' — the dataset example. example.inputs is the question, example.outputs is the ground_truth
i.e.
inputs  = {"question": "What audio devices do you carry?"}
outputs = {"ground_truth": "...", "reference_context_ids": [...], ...}
Therefore for the metrics that require the example we will use example.outputs and example.inputs


The function 'evaluate' will 
run rag_pipeline against every example in a LangSmith dataset 'rag-evaluation-dataset', 
then runs each evaluator against each result, 
and uploads everything to LangSmith as an "experiment" you can view in the UI.
"""

results = ls_client.evaluate(
        lambda x: rag_pipeline(x["question"], qdrant_client),
        data="rag-evaluation-dataset",
        evaluators=[
                ragas_faithfulness,
                ragas_response_relevancy,
                ragas_context_precision_id_based,
                ragas_context_recall_id_based,
        ],
        experiment_prefix="retriever",
)