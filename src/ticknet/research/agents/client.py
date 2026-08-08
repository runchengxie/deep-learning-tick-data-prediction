"""LLM 客户端抽象：Brainstorm/Critic Agent 的推理后端。

第一版默认使用 ``TemplateClient``（确定性模板，不调用外部 API），保证闭环可
离线运行。后续可切换 ``OpenAIClient`` / ``DeepSeekClient``（同 OpenAI 兼容协议），
只需实现 ``generate``。
"""

from __future__ import annotations

import os
from abc import ABC, abstractmethod


class LLMClient(ABC):
    """Agent 推理核心的抽象接口。"""

    @abstractmethod
    def generate(
        self,
        system_prompt: str,
        user_prompt: str,
        *,
        temperature: float = 0.0,
    ) -> str:
        """给定系统提示和用户提示，返回模型输出。"""


class TemplateClient(LLMClient):
    """确定性模板客户端：不调用 LLM，第一版默认。"""

    def __init__(self, template: str = "") -> None:
        self.template = template

    def generate(
        self,
        system_prompt: str,
        user_prompt: str,
        *,
        temperature: float = 0.0,
    ) -> str:
        del system_prompt, temperature
        if self.template:
            return self.template
        return user_prompt


class OpenAIClient(LLMClient):
    """OpenAI 兼容客户端（DeepSeek 亦兼容，通过 base_url 切换）。"""

    def __init__(
        self,
        model: str,
        base_url: str,
        api_key: str | None = None,
    ) -> None:
        self.model = model
        self.base_url = base_url
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY", "")

    def generate(
        self,
        system_prompt: str,
        user_prompt: str,
        *,
        temperature: float = 0.0,
    ) -> str:
        from openai import OpenAI  # type: ignore[import-not-found]

        client = OpenAI(api_key=self.api_key, base_url=self.base_url)
        response = client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=temperature,
        )
        return response.choices[0].message.content or ""


def make_client(
    provider: str = "template",
    *,
    model: str | None = None,
    base_url: str | None = None,
    api_key: str | None = None,
) -> LLMClient:
    """按 provider 构造客户端。provider: template / openai / deepseek。"""
    if provider == "template":
        return TemplateClient()
    if provider == "deepseek":
        return OpenAIClient(
            model=model or "deepseek-chat",
            base_url=base_url or "https://api.deepseek.com",
            api_key=api_key,
        )
    if provider == "openai":
        return OpenAIClient(
            model=model or "gpt-4o-mini",
            base_url=base_url or "https://api.openai.com/v1",
            api_key=api_key,
        )
    raise ValueError(f"未知 provider: {provider}")
