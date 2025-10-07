import inspect
from typing import Any, Awaitable, Callable, Optional, Sequence, Type, TypeVar, Union

from langchain_core.language_models import BaseChatModel
from langchain_core.tools import StructuredTool
from langchain_core.tools import tool as create_tool
from langgraph.graph import END, START, MessagesState, StateGraph
from langgraph.types import Command

from langgraph_codeact.utils import extract_and_combine_codeblocks

EvalFunction = Callable[[str, dict[str, Any]], tuple[str, dict[str, Any]]]
EvalCoroutine = Callable[[str, dict[str, Any]], Awaitable[tuple[str, dict[str, Any]]]]


class CodeActState(MessagesState):
    """State for CodeAct agent."""

    script: Optional[str]
    """The Python code script to be executed."""
    context: dict[str, Any]
    """Dictionary containing the execution context with available tools and variables."""


StateSchema = TypeVar("StateSchema", bound=CodeActState)
StateSchemaType = Type[StateSchema]


def create_default_prompt(base_prompt: Optional[str] = None):
    """Create default prompt for the CodeAct agent."""
    prompt = f"{base_prompt}\n\n" if base_prompt else ""
    prompt += """You will be given a task to perform. You should output either
- a Python code snippet that provides the solution to the task, or a step towards the solution. Any output you want to extract from the code should be printed to the console. Code should be output in a fenced code block.
- text to be shown directly to the user, if you want to ask for more information or provide the final answer.
- use the same language as the user's input.
"""

    return prompt


def create_codeact(
    model: BaseChatModel,
    eval_fn: Union[EvalFunction, EvalCoroutine],
    *,
    prompt: Optional[str] = None,
    state_schema: StateSchemaType = CodeActState,
) -> StateGraph:
    """Create a CodeAct agent.

    Args:
        model: The language model to use for generating code
        tools: List of tools available to the agent. Can be passed as python functions or StructuredTool instances.
        eval_fn: Function or coroutine that executes code in a sandbox. Takes code string and locals dict,
            returns a tuple of (stdout output, new variables dict)
        prompt: Optional custom system prompt. If None, uses default prompt.
            To customize default prompt you can use `create_default_prompt` helper:
            `create_default_prompt(tools, "You are a helpful assistant.")`
        state_schema: The state schema to use for the agent.

    Returns:
        A StateGraph implementing the CodeAct architecture
    """
    
    if prompt is None:
        prompt = create_default_prompt(prompt)

    def call_model(state: StateSchema) -> Command:
        messages = [{"role": "system", "content": prompt}] + state["messages"]
        response = model.invoke(messages)
        # Extract and combine all code blocks
        code = extract_and_combine_codeblocks(response.content)
        if code:
            return Command(goto="sandbox", update={"messages": [response], "script": code})
        else:
            # no code block, end the loop and respond to the user
            return Command(update={"messages": [response], "script": None})

    # If eval_fn is a async, we define async node function.
    if inspect.iscoroutinefunction(eval_fn):
        async def sandbox(state: StateSchema):
            # Execute the script in the sandbox
            output = await eval_fn(state["script"])
            return {
                "messages": [{"role": "user", "content": output}],
            }
    else:
        def sandbox(state: StateSchema):
            # Execute the script in the sandbox
            output = eval_fn(state["script"])
            return {
                "messages": [{"role": "user", "content": output}],
            }

    agent = StateGraph(state_schema)
    agent.add_node(call_model, destinations=(END, "sandbox"))
    agent.add_node(sandbox)
    agent.add_edge(START, "call_model")
    agent.add_edge("sandbox", "call_model")
    return agent
