from typing import Iterable, Self
from django.db import models
from django.conf import settings
from django.contrib.auth import get_user_model



class Project(models.Model):
    name = models.CharField(max_length=255)
    description = models.TextField(null=True, blank=True)
    date_created = models.DateTimeField(auto_now_add=True)
    date_updated = models.DateTimeField(auto_now=True)

    def __str__(self):
        return self.name


class Chat(models.Model):
    project = models.ForeignKey(
        Project, on_delete=models.SET_NULL, null=True, blank=True, related_name="chats"
    )
    date_created = models.DateTimeField(auto_now_add=True)
    date_updated = models.DateTimeField(auto_now=True)
    input_tokens_count = models.IntegerField(default=0)
    output_tokens_count = models.IntegerField(default=0)

    @classmethod
    def get_llm_user(cls):
        return get_user_model().objects.get(username="llm")

    @classmethod
    def create_llm_user(cls):
        return get_user_model().objects.create_user(username="llm", password="llm")

    def add_token_counts(self, input_token_count: int, output_token_count: int):
        self.input_tokens_count += input_token_count
        self.output_tokens_count += output_token_count
        self.save()

    def get_messages(self) -> models.QuerySet:
        return self.messages.order_by("date_created")

class Message(models.Model):
    class Type(models.TextChoices):
        USER = "user"
        ASSISTANT = "assistant"
        SYSTEM = "system"

    type = models.CharField(max_length=10, choices=Type.choices)
    chat = models.ForeignKey(Chat, on_delete=models.CASCADE, related_name="messages")
    text = models.TextField()
    user = models.ForeignKey(settings.AUTH_USER_MODEL, on_delete=models.CASCADE)
    date_created = models.DateTimeField(auto_now_add=True)
    date_updated = models.DateTimeField(auto_now=True)

    @classmethod
    def create_user_message(cls, chat, text: str, user) -> Self:
        return Message.objects.create(
            chat=chat,
            text=text,
            type=Message.Type.USER,
            user=user,
        )

    @classmethod
    def create_llm_message(cls, chat, text: str, user) -> Self:
        return Message.objects.create(
            chat=chat,
            text=text,
            type=Message.Type.ASSISTANT,
            user=user,
        )

    @classmethod
    def create_system_message(cls, chat, text: str, user) -> Self:
        return Message.objects.create(
            chat=chat, text=text, type=Message.Type.SYSTEM, user=user
        )


class LLMCall(models.Model):
    class Status(models.TextChoices):
        NEW = "new"
        GENERATION_IN_PROGRESS = "generation_in_progress"
        GENERATION_COMPLETED = "generation_completed"

    messages = models.ManyToManyField(Message)
    input_tokens_count = models.IntegerField(default=0)
    output_tokens_count = models.IntegerField(default=0)
    status = models.CharField(max_length=30, choices=Status.choices)
    response_data = models.JSONField(null=True, blank=True)

    @classmethod
    def create(cls, *messages: Iterable[Message]) -> Self:
        db_model = LLMCall.objects.create(status=LLMCall.Status.NEW)

        for m in messages:  # TODO optimize this to save all messages in one query
            db_model.add_message(m)

        return db_model

    def add_response_data(
        self, response_data: dict, input_token_count: int, output_token_count: int
    ):
        self.input_tokens_count += input_token_count
        self.output_tokens_count += output_token_count
        self.response_data = response_data
        self.status = self.Status.GENERATION_COMPLETED
        self.save()

    def add_message(self, message: Message):
        self.messages.add(message)

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "status": self.status,
            "response_data": self.response_data,
        }


class LLMCache(models.Model):
    cache_key = models.CharField(max_length=64, unique=True, db_index=True)
    model_name = models.CharField(max_length=255)
    response_text = models.TextField()
    response_data = models.JSONField()
    hit_count = models.IntegerField(default=0)
    date_created = models.DateTimeField(auto_now_add=True)
    date_updated = models.DateTimeField(auto_now=True)

    def __str__(self):
        return f"{self.model_name} - {self.cache_key[:8]}"
