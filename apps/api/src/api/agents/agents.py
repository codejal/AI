import instructor
import anthropic

from langsmith import traceable

from api.agents.utils.utils import make_message_anthropic_compatible, make_message_langchain_compatible
from api.agents.models import AgentResponse, IntentRouterResponse
from api.agents.utils.prompt_management import prompt_template_config


@traceable(
name="agent_node",
run_type="llm",
metadata={"ls_provider": "anthropic", "ls_model_name": "claude-haiku-4-5-20251001"}
)
def agent_node(state) -> dict:
        template = prompt_template_config(yaml_file="api/agents/prompts/qa_agent.yaml", prompt_key="qa_agent")
        prompt = template.render(
                available_tools=state.available_tools
        )

        conversation = make_message_anthropic_compatible(state.messages)

        if not conversation:
                conversation = [{"role": "user", "content": "Please assist with the query."}]

        instructor_client = instructor.from_anthropic(anthropic.Anthropic())

        response, raw_response = instructor_client.messages.create_with_completion(
                model="claude-haiku-4-5-20251001",
                response_model=AgentResponse,
                max_tokens=8096,
                system=prompt,
                messages=conversation,
                temperature=0.5,
        )

        ai_message = make_message_langchain_compatible(response)

        return {
                "messages": [ai_message], # toolNode from langchain require this (of langchain compatible type) to actually execute tool calls
                "tool_calls": response.tool_calls,
                "iteration": state.iteration + 1,
                "answer": response.answer,
                "final_answer": response.final_answer,
                "references": response.references
        }





@traceable(
        name="intent_router_node",
        run_type="llm",
        metadata={"ls_provider": "anthropic", "ls_model_name": "claude-haiku-4-5-20251001"},
)
def intent_router_node(state):
        template = prompt_template_config(yaml_file="api/agents/prompts/intent_router_agent.yaml", prompt_key="intent_router_agent")
        prompt = template.render()

        conversation = conversation = make_message_anthropic_compatible(state.messages)
        if not conversation:
                conversation = [{"role": "user", "content": "Please assist with the query."}]

        instructor_client = instructor.from_anthropic(anthropic.Anthropic())

        response, raw_response = instructor_client.messages.create_with_completion(
                model="claude-haiku-4-5-20251001",
                response_model=IntentRouterResponse,
                max_tokens=8096,
                system=prompt,
                messages=conversation,
                temperature=0.5,
        )

        return {
                "question_relevant": response.question_relevant,
                "answer": response.answer
        }