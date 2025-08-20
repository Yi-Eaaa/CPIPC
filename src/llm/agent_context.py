import re
import threading

from openai import OpenAI
from qwen_token_counter import TokenCounter

from config.config import GLOABLE_CONFIG
from llm.prompts import PROMPTS
from rag.utils import safe_unicode_decode


class AgentContext:
    token_counter = TokenCounter()
    compress_api = OpenAI(
        base_url=GLOABLE_CONFIG["chat_base_url"], api_key=GLOABLE_CONFIG["chat_api_key"]
    )
    section_titles = [
        "Primary Request and Intent",
        "Key Concepts and Domain Context",
        "Artifacts and Edits",
        "Challenges and Resolutions",
        "Problem Solving and Reasoning",
        "All User Messages",
        "Pending Tasks",
        "Current Focus",
        "Optional Next Step",
    ]

    def __init__(self, token_limit=200):
        self.contexts = []
        self.compressed_contexts = []
        self.token_limit = 0.92 * token_limit
        self.lock = threading.Lock()

    def add_context(self, added_context):
        if isinstance(added_context, list):
            self.contexts.extend(added_context)
            current_context_str = self.contexts_to_str()
            current_token_count = self.token_counter.count_tokens(current_context_str)
            # print(f"current_token_count:{current_token_count}")
            if current_token_count > self.token_limit:
                self.contexts = []
                threading.Thread(
                    target=self.compress_context,
                    args=(current_context_str,),
                    daemon=True,
                ).start()
        else:
            raise ValueError("Contexts to be added must be a list.")

    def get_context(self):
        history = []
        self.lock.acquire()
        try:
            if len(self.compressed_contexts) != 0:
                latest_compressed = self.compressed_contexts[-1]
                history.append(
                    {
                        "role": "system",
                        "content": PROMPTS["SUMMARY_IMPLICATION"].format(
                            previous_intent=latest_compressed[
                                "Primary Request and Intent"
                            ],
                            current_focus=latest_compressed["Current Focus"],
                        ),
                    }
                )
        finally:
            self.lock.release()

        if len(self.contexts) != 0:
            history.extend(self.contexts)

        return history

    def compress_context(self, context):
        self.lock.acquire()
        try:
            print("Compressing context...")
            if len(self.compressed_contexts) == 0:
                system_prompt = PROMPTS["CONTEXT_COMPRESS"].format(context=context)
            else:
                system_prompt = PROMPTS["CONTEXT_COMPRESS_WITH_HISTORY"].format(
                    previous_summary=self.compressed_contexts[-1]["Raw Content"],
                    current_context=context,
                )
            messages = []
            messages.append({"role": "system", "content": system_prompt})
            response = AgentContext.compress_api.chat.completions.create(
                model=GLOABLE_CONFIG["chat_model"],
                messages=messages,
                stream=False,
                temperature=0.3,
                top_p=0.7,
                # extra_body={"enable_thinking": False},
            )
            # print("Get response.")
            content = response.choices[0].message.content
            if r"\u" in content:
                content = safe_unicode_decode(content.encode("utf-8"))

            raw_content = content
            content = self.extract_sections(content)
            content["Raw Content"] = raw_content.strip()
            self.compressed_contexts.append(content)
        finally:
            # print(content)
            print("Context compressed successfully.")
            self.lock.release()

    def contexts_to_str(self):
        formatted_contexts = []
        for context in self.contexts:
            role = context.get("role", "").capitalize()
            content = context.get("content", "").strip()
            formatted_contexts.append(f"{role}: {content}\n")

        return "".join(formatted_contexts)

    def extract_sections(self, text):
        sections = {}
        for i, title in enumerate(AgentContext.section_titles):
            start_pattern = re.escape(title)
            if i + 1 < len(AgentContext.section_titles):
                end_pattern = re.escape(AgentContext.section_titles[i + 1])
                pattern = rf"{start_pattern}\s+(.*?)(?=\n{end_pattern})"
            else:
                pattern = rf"{start_pattern}\s+(.*)"
            match = re.search(pattern, text, re.DOTALL)
            if match:
                sections[title] = match.group(1).strip()
            else:
                sections[title] = None
        return sections


if __name__ == "__main__":
    agent_context = AgentContext()
    messages1 = [
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

    agent_context.add_context(messages1)

    messages2 = [
        {"role": "user", "content": "我还训练了一个NER模型，它在这场景下有什么用？"},
        {
            "role": "system",
            "content": "NER模型可以用于第一阶段的实体识别，从query中提取重要实体；RDF抽取模型则可进一步结构化这些实体之间的关系，为检索提供更高层次的语义表示。二者结合后能显著提升文档召回质量和生成精准度。",
        },
    ]
    agent_context.add_context(messages2)
    i = 0
    while i < 2:
        if len(agent_context.compressed_contexts) > i:
            i += 1
