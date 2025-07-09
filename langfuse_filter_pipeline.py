"""
title: Langfuse Filter Pipeline
author: AI
date: 2025-07-09
version: 1.7
license: MIT
description: A filter pipeline that uses Langfuse.
type: filter
requirements: langfuse
"""
import uuid
from typing import Optional

from langfuse import Langfuse
from langfuse.api.resources.public import UnauthorizedError
from openwebui.pipelines import Pipeline as BasePipeline

class Pipeline(BasePipeline):
    def __init__(self, valves, model_names, log_func, generation_tasks):
        self.valves = valves
        self.model_names = model_names
        self.log = log_func
        self.GENERATION_TASKS = generation_tasks
        self.chat_traces = {}

        self.set_langfuse()

    def set_langfuse(self):
        try:
            self.langfuse = Langfuse(
                secret_key=self.valves.secret_key,
                public_key=self.valves.public_key,
                host=self.valves.host,
                debug=self.valves.debug,
            )
            self.langfuse.auth_check()
            self.log("Langfuse клиент успешно инициализирован.")
        except UnauthorizedError:
            print("Неверные учетные данные Langfuse. Проверьте настройки.")
        except Exception as e:
            print(f"Ошибка Langfuse: {e}")

    async def inlet(self, body: dict, user: Optional[dict] = None) -> dict:
        chat_id = body.get("chat_id")
        task_name = body.get("task_name")
        user_email = user.get("email") if user else "anonymous"
        metadata = {}
        tags_list = body.get("tags")

        if chat_id not in self.chat_traces:
            self.log(f"Создаем новый trace для chat_id: {chat_id}")

            trace = self.langfuse.create_trace(
                name=f"chat:{chat_id}",
                input=body,
                user_id=user_email,
                metadata=metadata,
                session_id=chat_id,
                tags=tags_list
            )
            self.chat_traces[chat_id] = trace
        else:
            trace = self.chat_traces[chat_id]
            self.log(f"Используем существующий trace для chat_id: {chat_id}")
            if tags_list:
                trace.update(tags=tags_list)

        if task_name in self.GENERATION_TASKS:
            model_id = self.model_names.get(chat_id, {}).get("id", body["model"])
            model_name = self.model_names.get(chat_id, {}).get("name", "unknown")
            model_value = model_name if self.valves.use_model_name_instead_of_id_for_generation else model_id

            metadata["model_id"] = model_id
            metadata["model_name"] = model_name

            self.langfuse.create_generation(
                trace_id=trace.id,
                name=f"{task_name}:{str(uuid.uuid4())}",
                model=model_value,
                input=body["messages"],
                metadata=metadata,
                tags=tags_list
            )
        else:
            self.langfuse.create_event(
                trace_id=trace.id,
                name=f"{task_name}:{str(uuid.uuid4())}",
                metadata=metadata,
                input=body["messages"],
                tags=tags_list
            )

        return body

    async def outlet(self, body: dict, user: Optional[dict] = None) -> dict:
        chat_id = body.get("chat_id")
        task_name = body.get("task_name")
        metadata = {}
        tags_list = body.get("tags")

        if chat_id not in self.chat_traces:
            self.log(f"[WARNING] Нет trace для chat_id: {chat_id}, повторно регистрируем.")
            return await self.inlet(body, user)

        trace = self.chat_traces[chat_id]

        assistant_message = self.get_last_assistant_message(body["messages"])
        assistant_message_obj = self.get_last_assistant_message_obj(body["messages"])

        usage = None
        if assistant_message_obj:
            info = assistant_message_obj.get("usage", {})
            if isinstance(info, dict):
                input_tokens = info.get("prompt_eval_count") or info.get("prompt_tokens")
                output_tokens = info.get("eval_count") or info.get("completion_tokens")
                if input_tokens is not None and output_tokens is not None:
                    usage = {
                        "input": input_tokens,
                        "output": output_tokens,
                        "unit": "TOKENS",
                    }
                    self.log(f"Извлечена статистика токенов: {usage}")

        trace.update(output=assistant_message)

        metadata["type"] = task_name
        metadata["interface"] = "open-webui"

        if task_name in self.GENERATION_TASKS:
            model_id = self.model_names.get(chat_id, {}).get("id", body.get("model"))
            model_name = self.model_names.get(chat_id, {}).get("name", "unknown")
            model_value = model_name if self.valves.use_model_name_instead_of_id_for_generation else model_id

            metadata["model_id"] = model_id
            metadata["model_name"] = model_name

            self.langfuse.end_generation(
                trace_id=trace.id,
                name=f"{task_name}:{str(uuid.uuid4())}",
                model=model_value,
                input=body["messages"],
                metadata=metadata,
                usage=usage,
                tags=tags_list
            )
            self.log(f"Завершен generation для chat_id: {chat_id}")
        else:
            if usage:
                metadata["usage"] = usage
            self.langfuse.end_event(
                trace_id=trace.id,
                name=f"{task_name}:{str(uuid.uuid4())}",
                metadata=metadata,
                input=body["messages"],
                tags=tags_list
            )
            self.log(f"Записан event для chat_id: {chat_id}")

        return body

    # Вспомогательные функции для получения последнего assistant-сообщения
    def get_last_assistant_message(self, messages):
        return next((m["content"] for m in reversed(messages) if m["role"] == "assistant"), None)

    def get_last_assistant_message_obj(self, messages):
        return next((m for m in reversed(messages) if m["role"] == "assistant"), None)
    