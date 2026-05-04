from dotenv import load_dotenv
import os
import voyageai

from qdrant_client import QdrantClient
import anthropic



def get_embedding(voyageai_client, text, model = 'voyage-3'):
        result = voyageai_client.embed(
                [text],
                model=model,
                input_type="document"
        )
        return result.embeddings[0]


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


def process_context(context):
        formatted_context = ""
        for id, chunk, rating in zip(context['retrieved_context_ids'], context['retrieved_context'], context['retrieved_context_ratings']):
                formatted_context += f"- ID: {id}, rating: {rating}, description: {chunk}\n"
        return formatted_context


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


def generate_answer(anthropic_client, prompt):
        message = anthropic_client.messages.create(
                max_tokens=2000,
                messages=[
                        {
                        "role": "user",
                        "content": prompt,
                        }
                ],
                model="claude-opus-4-7",
        )
        return message.content[0].text


def rag_pipeline(question, top_k=10):
        load_dotenv()
        VOYAGE_API_KEY = os.environ.get("VOYAGE_API_KEY")
        voyageai_client = voyageai.Client(api_key=VOYAGE_API_KEY)
        anthropic_client = anthropic.Anthropic()
        qdrant_client = QdrantClient(url='http://qdrant:6333')
        retrieved_context = retrieve_data(voyageai_client, question, qdrant_client, top_k)
        preprocessed_context = process_context(retrieved_context)
        prompt = build_pompt(preprocessed_context, question)
        answer = generate_answer(anthropic_client, prompt)

        return answer