import anthropic
from dotenv import load_dotenv
from langsmith import traceable, get_current_run_tree
from qdrant_client import QdrantClient
from qdrant_client.models import Prefetch, FusionQuery, Document
import os

import voyageai


load_dotenv()
VOYAGE_API_KEY = os.environ.get("VOYAGE_API_KEY")
voyageai_client = voyageai.Client(api_key=VOYAGE_API_KEY)
anthropic_client = anthropic.Anthropic()
qdrant_client = qdrant_client = QdrantClient(url='http://qdrant:6333')

@traceable(
        name="embed_query",
        run_type="embedding",
        metadata={"ls_provider": "voyageai", "ls_model_name": "voyage-3"},
)
def get_embedding(voyageai_client, text, model = 'voyage-3'):
        result = voyageai_client.embed(
                [text],
                model=model,
                input_type="document"
        )

        # add metadata to langsmith obervability
        current_run = get_current_run_tree()
        if current_run:
                current_run.metadata["usage_metadata"] = {
                        "total_tokens": result.__dict__['total_tokens'],
                }

        return result.embeddings[0]


@traceable(
        name="retrieve_data",
        run_type="retriever",
)
def retrieve_data(query, k=5):
        query_embedding = get_embedding(voyageai_client, query)
        results = qdrant_client.query_points(
                collection_name="Amazon-items-collection-01-hybrid-search",
                prefetch=[
                        Prefetch(
                                query=query_embedding,
                                using='voyage-3',
                                limit=10,
                        ),
                        Prefetch(
                                query=Document(
                                        text=query,
                                        model="qdrant/bm25"
                                ),
                                using='bm25',
                                limit=10,
                        ),
                ],
                query=FusionQuery(fusion='rrf'), #reciprocal rank fusion
                limit=k,
        )

        retrieved_context_ids = []
        retrieved_context = []
        similarity_scores = []
        retrieved_context_ratings = []

        for item in results.points:
                retrieved_context_ids.append(item.payload['parent_asin'])
                retrieved_context.append(item.payload['description'])
                retrieved_context_ratings.append(item.payload['average_rating'])
                similarity_scores.append(item.score)
        
        return {
                "retrieved_context_ids": retrieved_context_ids,
                "retrieved_context": retrieved_context,
                "retrieved_context_ratings": retrieved_context_ratings,
                "similarity_scores": similarity_scores,
        }


@traceable(
        name="format_retrieved_context",
        run_type="prompt",
)
def process_context(context):
        formatted_context = ""
        for id, chunk, rating in zip(context['retrieved_context_ids'], context['retrieved_context'], context['retrieved_context_ratings']):
                formatted_context += f"- ID: {id}, rating: {rating}, description: {chunk}\n"
        return formatted_context


def get_formatted_context(query: str, top_k: int = 5) -> str:
        """
        Get the top k context, each representing an inventory item for a given query.

        Args:
                query: The query to get the top k context for
                top_k: The number of context chunks to retrieve, works best with 5 or more

        Returns:
                A string of the top k context chunks with IDs and average ratings prepending each chunk, each representing an inventory item for a given query.
        """

        context = retrieve_data(query, top_k)
        formatted_context = process_context(context)
        return formatted_context