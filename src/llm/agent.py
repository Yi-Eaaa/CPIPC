from config.config import GLOABLE_CONFIG
from openai import OpenAI, AsyncOpenAI
from rag.utils import safe_unicode_decode

from llm.context_manager import ContextManager


class Agent:
    def __init__(self, api_key: str, base_url: str = "https://api.siliconflow.cn/v1"):
        self.api_key = api_key
        self.base_url = base_url
        self.client = OpenAI(base_url=self.base_url, api_key=self.api_key)
        self.context_manager = ContextManager()
        self.context_manager.create_session("default")

    def new_session(self, session_name: str, token_limit):
        """
        Create a new session with the specified name and token limit.
        """
        self.context_manager.create_session(session_name, token_limit)

    async def async_chat(self):
        pass

    def chat(
        self,
        prompt: str,
        model: str = GLOABLE_CONFIG["chat_model"],
        system_prompt: str = None,
        stream: bool = False,
        max_tokens: int = 8192,
        temperature=0.3,
        top_p=0.7,
        multi_turn: bool = False,
        session_name="default",
        **kwargs
    ):
        messages = []

        if multi_turn:
            history = self.context_manager.get_context(session_name)
            if len(history) != 0:
                messages.extend(history)

        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})

        user_context = {"role": "user", "content": prompt}
        messages.append(user_context)
        response = self.client.chat.completions.create(
            model=model,
            messages=messages,
            stream=stream,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            **kwargs
        )
        content = response.choices[0].message.content
        if r"\u" in content:
            content = safe_unicode_decode(content.encode("utf-8"))

        if multi_turn:
            added_context = []
            added_context.append(user_context)
            system_context = {"role": "system", "content": content}
            added_context.append(system_context)
            self.context_manager.add_context(session_name, added_context)
        return content


if __name__ == "__main__":
    agent = Agent(api_key=GLOABLE_CONFIG["chat_api_key"])
    agent.new_session("test", token_limit=200)
    history = [
        {
            "role": "user",
            "content": "我在使用DataParallel训练模型时，batch size应该如何设置？我有4张显卡，希望每张显卡上处理4个样本。",
        },
        {
            "role": "system",
            "content": "如果你使用 `torch.nn.DataParallel`，它会自动将你设置的 `batch_size` 平均分到每张显卡上。所以你只需要将 `DataLoader` 中的 `batch_size` 设置为 4 x 4 = 16 即可，每张显卡会获得4个样本。",
        },
        {
            "role": "user",
            "content": "有没有什么比较先进的从query中抽取关键实体的方法推荐？",
        },
        {
            "role": "system",
            "content": "你可以尝试基于指令微调（instruction tuning）的T5或GPT模型，输入query并要求其输出关键实体。此外，近年来的Prompt-based NER方法（如SpanMarker、UIE）在多任务场景中表现优秀，适合构建可泛化的RAG前端实体抽取模块。",
        },
        {
            "role": "user",
            "content": "谢谢！另外，我在RAG系统中使用Flan-T5来抽取RDF三元组，有什么场景下这个微调模型是有用的？",
        },
        {
            "role": "system",
            "content": "你的微调模型可以用于从用户查询中抽取结构化的知识片段，从而用于构建查询意图表示、增强召回、或构建知识图谱作为检索支撑，尤其在复杂问答、多轮指代消解中效果明显。",
        },
    ]

    agent.context_manager.add_context(session_name="test", added_context=history)

    query = "我还训练了一个NER模型，它在这场景下有什么用？"
    response = agent.chat(
        prompt=query,
        extra_body={"enable_thinking": False},
        multi_turn=True,
        session_name="test",
    )
    print(response)
    pass
