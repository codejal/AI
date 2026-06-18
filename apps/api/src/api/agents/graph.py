
import numpy as np

from qdrant_client import QdrantClient
from qdrant_client.models import Filter, FieldCondition, MatchValue

from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import ToolNode

from api.agents.utils.utils import get_tool_descriptions
from api.agents.tools import get_formatted_context
from api.agents.agents import agent_node, intent_router_node
from api.agents.models import State


def tool_router(state: State) -> str:
        """Decide wheater to continue or end"""
        if state.final_answer or state.iteration > 1:
                return "end"
        if len(state.tool_calls) > 0:
                return "tools"
        return "end"


def intent_router_conditional_edge(state: State):
        if state.question_relevant:
                return "agent"
        else:
                return "end"


workflow = StateGraph[State, None, State, State](State)

tools = [get_formatted_context]
tool_node = ToolNode(tools)
tool_descriptions = get_tool_descriptions(tools)

workflow.add_node("agent_node", agent_node)
workflow.add_node("tool_node", tool_node)
workflow.add_node("intent_router_node", intent_router_node)

workflow.add_edge(START, "intent_router_node")

workflow.add_conditional_edges(
        source = "intent_router_node",
        path = intent_router_conditional_edge,
        path_map = {
                "agent": "agent_node",
                "end": END,
        }
)
workflow.add_conditional_edges(
        source = "agent_node",
        path = tool_router,
        path_map = {
                "tools": "tool_node",
                "end": END,
        }
)

workflow.add_edge("tool_node", "agent_node")

graph = workflow.compile()


def run_agent(question: str) -> dict:
        initial_state = {
                "messages": [{"role": "user", "content": question}],
                "available_tools": tool_descriptions,
                "iteration": 0,

        }
        result = graph.invoke(initial_state)
        return result


def rag_agent_wrapper(question):
        qdrant_client = QdrantClient(url='http://qdrant:6333')
        result = run_agent(question)
        
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
                        collection_name="Amazon-items-collection-01-hybrid-search",
                        query=dummy_vector,
                        limit=1,
                        using='voyage-3',
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
                "answer": result.get("answer", ""),
                "used_context": used_context,
        }
