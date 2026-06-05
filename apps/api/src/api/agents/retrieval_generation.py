from dotenv import load_dotenv
import os
import voyageai
from langsmith import traceable, get_current_run_tree

from qdrant_client import QdrantClient
import anthropic


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
def retrieve_data(voyageai_client, query, qdrant_client, k=5):
        query_embedding = get_embedding(voyageai_client, query)
        results = qdrant_client.query_points(
                collection_name="Amazon-items-collection-00",
                query=query_embedding,
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

@traceable(
        name="build_pompt",
        run_type="prompt",
)
def build_pompt(preprocessed_context, question):
        prompt = f"""
You are a shopping assistant that can answer questions about the products in stock.
You will be given a question and a list of context

Instrctions:
- You need to answer the question based on the provided context only
- Never use word context and refer to it as the available products

Context:
{preprocessed_context}

Question:
{question}
        """
        return prompt

@traceable(
        name="generate_answer",
        run_type="llm",
        metadata={"ls_provider": "anthropic", "ls_model_name": "claude-haiku-4-5"},
)
def generate_answer(anthropic_client, prompt):
        message = anthropic_client.messages.create(
                max_tokens=2000,
                messages=[
                        {
                        "role": "user",
                        "content": prompt,
                        }
                ],
                model="claude-haiku-4-5",
        )

        current_run = get_current_run_tree()
        if current_run:
                current_run.metadata["usage_metadata"] = {
                        "input_tokens": message.__dict__['usage'].__dict__['input_tokens'],
                        "output_tokens": message.__dict__['usage'].__dict__['output_tokens'],
                }
        return message.content[0].text

@traceable(
        name="rag_pipeline"
)
def rag_pipeline(question, qdrant_client, top_k=10):
        load_dotenv()
        VOYAGE_API_KEY = os.environ.get("VOYAGE_API_KEY")
        voyageai_client = voyageai.Client(api_key=VOYAGE_API_KEY)
        anthropic_client = anthropic.Anthropic()
        # qdrant_client = QdrantClient(url='http://qdrant:6333')
        retrieved_context = retrieve_data(voyageai_client, question, qdrant_client, top_k)
        preprocessed_context = process_context(retrieved_context)
        prompt = build_pompt(preprocessed_context, question)
        answer = generate_answer(anthropic_client, prompt)

        # for evaluation we should return the following
        final_result = {
                "answer": answer,
                "question": question,
                "retrieved_context_ids": retrieved_context["retrieved_context_ids"],
                "retrieved_context": retrieved_context["retrieved_context"],
                "similarity_scores": retrieved_context["similarity_scores"]
        }

        return final_result