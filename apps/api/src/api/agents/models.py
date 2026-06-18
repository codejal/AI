from pydantic import BaseModel, Field
from typing import Annotated, Any, Dict, List
from operator import add


class RAGUsedContext(BaseModel):
        id: str = Field(description="The ID of the item used to answer the question")
        description: str = Field(description="Short description of the item used to answe the question")

class ToolCall(BaseModel):
        name: str
        aruguments: dict

class AgentResponse(BaseModel):
        tool_calls: List[ToolCall] = []
        answer: str = Field(default="", description="Answer to the user's question based on current knowledge and tool results")
        final_answer: bool = Field(default=False, description="True if you have all information needed to provide a complete answer, False otherwise")
        references: List[RAGUsedContext] = Field(default=[], description="Items from retrieved context used to compile the answer")

class IntentRouterResponse(BaseModel):
        question_relevant: bool
        answer: str

class  State(BaseModel):
        messages: Annotated[List[Any], add] = []
        question_relevant: bool = False
        iteration: int = 0
        answer: str = ""
        available_tools: List[Dict[str, Any]] = []
        tool_calls: List[ToolCall] = []
        final_answer: bool = False
        references: Annotated[List[RAGUsedContext], add] = []