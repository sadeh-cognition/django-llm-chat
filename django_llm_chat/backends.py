import json
import os
import requests
from typing import Iterable, Generator
from abc import ABC, abstractmethod

import litellm
from litellm import completion
from pydantic import BaseModel
from .models import Message


class LLMProvider(ABC):
    @abstractmethod
    def generate(
        self,
        model_name: str,
        messages: Iterable[Message],
        temperature: float | None = None,
        max_tokens: int | None = None,
        output_model: type[BaseModel] | None = None,
    ) -> tuple[str, dict]:
        pass

    @abstractmethod
    def stream(
        self,
        model_name: str,
        messages: Iterable[Message],
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> Generator[str | tuple[str, dict], None, None]:
        pass


class LiteLLMProvider(LLMProvider):
    def _prepare_messages(self, messages: Iterable[Message]) -> list[dict]:
        return [{"content": msg.text, "role": msg.type} for msg in messages]

    def generate(
        self,
        model_name: str,
        messages: Iterable[Message],
        temperature: float | None = None,
        max_tokens: int | None = None,
        output_model: type[BaseModel] | None = None,
    ) -> tuple[str, dict]:
        litellm_messages = self._prepare_messages(messages)
        completion_kwargs = {}
        if temperature is not None:
            completion_kwargs["temperature"] = temperature
        if max_tokens is not None:
            completion_kwargs["max_tokens"] = max_tokens
        if output_model is not None:
            completion_kwargs["response_format"] = output_model

        response = completion(
            model=model_name,
            messages=litellm_messages,
            **completion_kwargs,
        )

        message = response.choices[0].message.to_dict()
        response_text = response.choices[0].message.content
        message.pop("content", None)
        message.pop("role", None)

        response_data = {
            "message": message,
            "id": response.id,
            "model": response.model,
            "usage": response.usage.to_dict(),
        }
        return response_text, response_data

    def stream(
        self,
        model_name: str,
        messages: Iterable[Message],
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> Generator[str | tuple[str, dict], None, None]:
        litellm_messages = self._prepare_messages(messages)
        completion_kwargs = {}
        if temperature is not None:
            completion_kwargs["temperature"] = temperature
        if max_tokens is not None:
            completion_kwargs["max_tokens"] = max_tokens

        response = completion(
            model=model_name,
            messages=litellm_messages,
            stream=True,
            **completion_kwargs,
        )

        chunks = []
        for chunk in response:
            chunks.append(chunk)
            content = chunk.choices[0].delta.content or ""
            if content:
                yield content

        reconstructed_response = litellm.stream_chunk_builder(
            chunks, messages=litellm_messages
        )
        message_dict = reconstructed_response.choices[0].message.to_dict()
        response_text = reconstructed_response.choices[0].message.content
        message_dict.pop("content", None)
        message_dict.pop("role", None)

        response_data = {
            "message": message_dict,
            "id": reconstructed_response.id,
            "model": reconstructed_response.model,
            "usage": reconstructed_response.usage.to_dict(),
        }
        yield response_text, response_data


class LMStudioProvider(LLMProvider):
    def _prepare_messages(self, messages: Iterable[Message]) -> list[dict]:
        lms_messages = []
        for msg in messages:
            role = msg.type
            if role == Message.Type.SYSTEM:
                role = "system"
            elif role == Message.Type.USER:
                role = "user"
            elif role == Message.Type.ASSISTANT:
                role = "assistant"
            lms_messages.append({"role": role, "content": msg.text})
        return lms_messages

    def _get_api_url(self):
        base_url = os.environ.get("LM_STUDIO_API_BASE", "http://localhost:1234")
        return f"{base_url.rstrip('/')}/api/v1/chat"

    def generate(
        self,
        model_name: str,
        messages: Iterable[Message],
        temperature: float | None = None,
        max_tokens: int | None = None,
        output_model: type[BaseModel] | None = None,
    ) -> tuple[str, dict]:
        system_msg = next(
            (m.text for m in messages if m.type == Message.Type.SYSTEM), ""
        )
        user_msg = list(messages)[-1].text if messages else ""

        data = {
            "model": model_name,
            "system_prompt": system_msg,
            "input": user_msg,
            "stream": False,
        }
        if temperature is not None:
            data["temperature"] = temperature
        if max_tokens is not None:
            data["max_tokens"] = max_tokens

        response = requests.post(self._get_api_url(), json=data)
        response.raise_for_status()
        result = response.json()

        output_content = "".join(
            [
                o.get("content", "")
                for o in result.get("output", [])
                if o.get("type") == "message"
            ]
        )
        stats = result.get("stats", {})
        prompt_tokens = stats.get("input_tokens", 0)
        completion_tokens = stats.get("total_output_tokens", 0)

        response_data = {
            "message": {"content": output_content},
            "id": f"lms-{result.get('response_id', 'unknown')}",
            "model": model_name,
            "usage": {
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": prompt_tokens + completion_tokens,
            },
        }
        return output_content, response_data

    def stream(
        self,
        model_name: str,
        messages: Iterable[Message],
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> Generator[str | tuple[str, dict], None, None]:
        system_msg = next(
            (m.text for m in messages if m.type == Message.Type.SYSTEM), ""
        )
        user_msg_text = next(
            (m.text for m in reversed(list(messages)) if m.type == Message.Type.USER),
            "",
        )

        data = {
            "model": model_name,
            "system_prompt": system_msg,
            "input": user_msg_text,
            "stream": True,
        }
        if temperature is not None:
            data["temperature"] = temperature
        if max_tokens is not None:
            data["max_tokens"] = max_tokens

        response_text = ""
        prompt_tokens = 0
        completion_tokens = 0
        response_id = "unknown"

        with requests.Session().post(
            self._get_api_url(), json=data, stream=True
        ) as resp:
            resp.raise_for_status()
            for line in resp.iter_lines():
                if not line:
                    continue
                line_str = line.decode("utf-8")
                if line_str.startswith("data: "):
                    data_str = line_str[6:]
                    try:
                        event_data = json.loads(data_str)
                        if event_data.get("type") == "message.delta":
                            content = event_data.get("content", "")
                            response_text += content
                            yield content
                        elif event_data.get("type") == "chat.end":
                            result_data = event_data.get("result", {})
                            stats = result_data.get("stats", {})
                            prompt_tokens = stats.get("input_tokens", 0)
                            completion_tokens = stats.get("total_output_tokens", 0)
                            response_id = result_data.get("response_id", "unknown")
                    except json.JSONDecodeError:
                        continue

        response_data = {
            "message": {"content": response_text},
            "id": f"lms-{response_id}",
            "model": model_name,
            "usage": {
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": prompt_tokens + completion_tokens,
            },
        }
        yield response_text, response_data
