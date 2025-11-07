
**自动执行的Agent**

在第二篇的基础上，创建智能体Agent，实现对工具的自动化调用执行。

```python

from langchain_openai import ChatOpenAI
from pydantic import SecretStr, BaseModel, Field
from langchain_core.prompts import ChatPromptTemplate
# 假设使用 OpenAI 类 LLM（根据实际 LLM 替换，如 Anthropic、本地化模型等）
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool, Tool
from langchain.agents import create_agent

api_key = SecretStr("sk-XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX")

# 1. 定义工具函数（二选一：要么用 @tool 装饰器，要么用 Tool.from_function）
# 方式一：使用 @tool 装饰器（推荐，更简洁）
class AddInputArgs(BaseModel):
    a: str = Field(description="first number")
    b: str = Field(description="second number")

@tool(
    description="add two numbers",
    args_schema=AddInputArgs,
    return_direct=True,
)

def add(a, b):
    # 注意：参数是 str 类型，需要先转换为数字再计算
    try:
        result = float(a) + float(b)
        # 若为整数，转换为 int 避免 .0 结尾
        return str(int(result) if result.is_integer() else result)
    except ValueError:
        return "错误：请输入有效的数字"

# 2. 初始化 LLM（根据实际情况替换，这里以 OpenAI 为例）
# 注意：需要设置环境变量 OPENAI_API_KEY，或直接传入 api_key 参数
llm = ChatOpenAI(
    model="qwen-plus",
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
    api_key=api_key,
    streaming=True,
)

# 3. 定义智能体提示词（明确智能体角色和工具使用规则）
system_prompt = """
你是一名智能计算助手小慕，专注于解决数学加法问题：
1. 当用户的请求是计算两个数字相加时，必须使用提供的 add 工具进行计算，不要手动心算
2. 若用户输入的不是数字（如文字、符号），直接返回工具的错误提示
3. 结果返回格式：简洁明了，直接给出计算过程和最终结果，无需额外冗余信息
"""

# prompt = ChatPromptTemplate.from_messages([
#     ("system", system_prompt),
#     ("human", "{user_input}"),
# ])

# 4. 创建Agent
# 工具列表：可后续扩展其他工具（如减法、乘法等）
tools = [add]
# 创建链条：prompt → LLM（绑定工具）→ 工具执行 → 结果输出
agent = create_agent(
    model=llm,
    tools=tools,
    system_prompt=system_prompt,
)

# 5. 封装智能体调用函数（方便重复使用）
def agent_calculate(user_input):
    inputs = {"messages": [{"role": "user", "content": user_input}]}
    """智能体入口函数：接收用户输入，自动调用工具并返回结果"""
    for chunk in agent.stream(inputs, stream_mode="updates"):
        for step, data in chunk.items():
            print(f"step: {step}")
            print(f"content: {data['messages'][-1].content_blocks}")
    # for token, metadata in agent.stream(inputs, stream_mode="messages"):
    #     print(f"node: {metadata['langgraph_node']}")
    #     print(f"content: {token.content_blocks}")
    #     print("\n")

    # print(agent.invoke(inputs))

# 6. 测试案例
if __name__ == "__main__":
    # 测试1：正常整数加法
    print("用户：计算100+100")
    agent_calculate('计算100+100')

    # # 测试2：小数加法
    # print("用户：3.14 + 2.86 等于多少？")
    # print(f"小慕：{agent_calculate('3.14 + 2.86 等于多少？')}\n")
    #
    # # 测试3：非数字输入
    # print("用户：计算a + 5")
    # print(f"小慕：{agent_calculate('计算a + 5')}\n")
    #
    # # 测试4：自然语言描述的加法
    # print("用户：我有2个苹果，再买8个，一共多少个？")
    # print(f"小慕：{agent_calculate('我有2个苹果，再买8个，一共多少个？')}\n")
```

运行结果

```
用户：计算100+100
step: model
content: [{'type': 'tool_call', 'name': 'add', 'args': {'a': '100', 'b': '100'}, 'id': 'call_5b65639996b7484e82d8f7'}]
step: tools
content: [{'type': 'text', 'text': '200'}]

```
