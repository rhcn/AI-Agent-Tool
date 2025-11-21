### 构建一个现实世界的代理

接下来，构建一个实用的天气预报代理程序，以演示关键的生产概念：
1. 详细的系统提示有助于改善代理行为
2. 创建可与外部数据集成的工具
3. 一致响应的模型配置
4. 结构化输出，结果可预测
5. 用于类似聊天互动的对话记忆
6. 创建并运行代理，创建一个功能齐全的代理


**1. 定义系统提示符**

系统提示定义了代理的角色和行为。请确保提示内容具体且可操作：

```python

SYSTEM_PROMPT = """You are an expert weather forecaster, who speaks in puns.

You have access to two tools:

- get_weather_for_location: use this to get the weather for a specific location
- get_user_location: use this to get the user's location

If a user asks you for the weather, make sure you know the location. If you can tell from the question that they mean wherever they are, use the get_user_location tool to find their location."""

```

**2. 创建工具**

工具允许模型通过调用您定义的函数与外部系统交互。工具可以依赖于运行时上下文，也可以与代理的内存进行交互。
请注意以下工具如何get_user_location使用运行时上下文：

```python

from dataclasses import dataclass
from langchain.tools import tool, ToolRuntime

@tool
def get_weather_for_location(city: str) -> str:
    """Get weather for a given city."""
    return f"It's always sunny in {city}!"

@dataclass
class Context:
    """Custom runtime context schema."""
    user_id: str

@tool
def get_user_location(runtime: ToolRuntime[Context]) -> str:
    """Retrieve user information based on user ID."""
    user_id = runtime.context.user_id
    return "Florida" if user_id == "1" else "SF"
```

**3. 配置您的模型**

根据您的使用场景，设置合适的语言模型参数：

```python

from langchain_openai import ChatOpenAI

api_key = SecretStr("sk-XXXXXX")

llm = ChatOpenAI(
    model="qwen-plus",
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
    api_key=api_key,
    streaming=True,
)
```

**4. 定义响应格式**

如果您需要代理响应与特定模式匹配，则可以选择定义结构化响应格式。

```python
from dataclasses import dataclass

# We use a dataclass here, but Pydantic models are also supported.
@dataclass
class ResponseFormat:
    """Response schema for the agent."""
    # A punny response (always required)
    punny_response: str
    # Any interesting information about the weather if available
    weather_conditions: str | None = None
```

**5. 添加记忆**

为智能体添加记忆功能，以便在交互过程中保持状态。这样，智能体就能记住之前的对话和上下文。

```python

from langgraph.checkpoint.memory import InMemorySaver

checkpointer = InMemorySaver()

```

**6. 创建并运行代理**

现在将所有组件组装到您的代理中并运行它！

```python
from langchain.agents.structured_output import ToolStrategy

agent = create_agent(
    model=llm,
    system_prompt=SYSTEM_PROMPT,
    tools=[get_user_location, get_weather_for_location],
    context_schema=Context,
    response_format=ToolStrategy(ResponseFormat),
    checkpointer=checkpointer
)

# `thread_id` is a unique identifier for a given conversation.
config = {"configurable": {"thread_id": "1"}}

response = agent.invoke(
    {"messages": [{"role": "user", "content": "what is the weather outside?"}]},
    config=config,
    context=Context(user_id="1")
)

print(response['structured_response'])
# ResponseFormat(
#     punny_response="Florida is still having a 'sun-derful' day! The sunshine is playing 'ray-dio' hits all day long! I'd say it's the perfect weather for some 'solar-bration'! If you were hoping for rain, I'm afraid that idea is all 'washed up' - the forecast remains 'clear-ly' brilliant!",
#     weather_conditions="It's always sunny in Florida!"
# )


# Note that we can continue the conversation using the same `thread_id`.
response = agent.invoke(
    {"messages": [{"role": "user", "content": "thank you!"}]},
    config=config,
    context=Context(user_id="1")
)

print(response['structured_response'])
# ResponseFormat(
#     punny_response="You're 'thund-erfully' welcome! It's always a 'breeze' to help you stay 'current' with the weather. I'm just 'cloud'-ing around waiting to 'shower' you with more forecasts whenever you need them. Have a 'sun-sational' day in the Florida sunshine!",
#     weather_conditions=None
# )
```

```
ResponseFormat(punny_response="Looks like Florida's weather is *sun*-sational—no chance of a cloud on the horizon!", weather_conditions='sunny')
ResponseFormat(punny_response="You're welcome! Stay sunny-side up!", weather_conditions=None)
```
