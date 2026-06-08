from dotenv import load_dotenv
import os
import voyageai
from langsmith import traceable, get_current_run_tree

from qdrant_client import QdrantClient
from qdrant_client.models import Filter, FieldCondition, MatchValue
import anthropic
from pydantic import BaseModel, Field
import instructor
import numpy as np

class RAGUsedContext(BaseModel):
        id: str = Field(description="The ID of the item used to answer the question")
        description: str = Field(description="Short description of the item used to answe the question")

class RAGGenerationResponse(BaseModel):
        answer: str = Field(description="The answer to the question")
        references: list[RAGUsedContext] = Field(description="List of items used to answer the question")


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
        - As an output you need to provide:
                * The answer to the question based on the provided context.
                * The list of the IDs of the chunks that were used to answer the question. Only return the ones that are used in the answer
                * Short description (1-2 sentences) of the item based on the description provided in the context 
        - The answer description should have name of the item
        - The answer to the question should contain detailed information about the product and returned with the detailed specification in bullet points

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
        instructor_client = instructor.from_anthropic(anthropic.Anthropic())
        message, raw_response = instructor_client.messages.create_with_completion(
                max_tokens=2000,
                messages=[
                        {
                        "role": "user",
                        "content": prompt,
                        }
                ],
                model="claude-haiku-4-5",
                temperature=0,
                response_model=RAGGenerationResponse,
        )

        current_run = get_current_run_tree()
        if current_run:
                current_run.metadata["usage_metadata"] = {
                        "input_tokens": raw_response.__dict__['usage'].__dict__['input_tokens'],
                        "output_tokens": raw_response.__dict__['usage'].__dict__['output_tokens'],
                }
        return message

@traceable(
        name="rag_pipeline"
)
def rag_pipeline(question, qdrant_client, top_k=5):
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
                "answer": answer.answer,
                "references": answer.references,
                "question": question,
                "retrieved_context_ids": retrieved_context["retrieved_context_ids"],
                "retrieved_context": retrieved_context["retrieved_context"],
                "similarity_scores": retrieved_context["similarity_scores"]
        }

        return final_result


def rag_pipeline_wrapper(question, top_k=5):
        qdrant_client = QdrantClient(url='http://qdrant:6333')
        result = rag_pipeline(question, qdrant_client, top_k)
        
        used_context = []
        dummy_vector = np.zeros(1024).tolist()

        for item in result.get('references', []):
                # use the id to get the complete payload from qdrant 
                """ example payload
                {
                        "description":"USB C Hub, MCY USB C to HDMI Multiptort Adapter, 1…"
                        "image":"https://m.media-amazon.com/images/I/41KhTIzecrS._A…"
                        "rating_number":358
                        "price": NULL
                        "average_rating":4.6
                        "parent_asin":"B0BTYK7SB3"
                }
                """
                payload = qdrant_client.query_points(
                        collection_name="Amazon-items-collection-00",
                        query=dummy_vector,
                        limit=1,
                        with_payload=True,
                        query_filter=Filter(
                                must=[
                                        FieldCondition(
                                                key="parent_asin",
                                                match=MatchValue(value=item.id)
                                        )
                                ]
                        )
                ).points[0].payload
                image_url = payload.get("image")
                price = payload.get("price")
                if image_url:
                        used_context.append({
                                "image_url": image_url,
                                "price": price,
                                "description": item.description #from LLM
                        })
                
        return {
                "answer": result["answer"],
                "used_context": used_context,
        }
