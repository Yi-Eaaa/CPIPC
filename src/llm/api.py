from openai import OpenAI

from rag.utils import safe_unicode_decode
from config.config import GLOABLE_CONFIG

API_KEY = GLOABLE_CONFIG["chat_api_key"]
MODEL = GLOABLE_CONFIG["chat_model"]

class SiliconFlowAPI:
    def __init__(self, api_key: str, base_url: str = "https://api.siliconflow.cn/v1"):
        self.api_key = api_key
        self.base_url = base_url
        self.client = OpenAI(base_url=self.base_url, api_key=self.api_key)

    def chat(
        self,
        model: str,
        prompt: str,
        system_prompt: str = None,
        stream: bool = False,
        max_tokens: int = 8192,
        temperature=0.7,
        top_p=0.7,
        **kwargs
    ):
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})
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
        return content


if __name__ == "__main__":
    llm_client = SiliconFlowAPI(API_KEY)
    response = llm_client.chat(MODEL, "帮我鞭打洪逸")
    print(response)
    
