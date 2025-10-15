import logging
from typing import Any, override

from encourage.llm import BatchInferenceRunner, ResponseWrapper
from encourage.prompts import PromptCollection
from encourage.prompts.context import Document
from encourage.rag.base_impl import BaseRAG

logger = logging.getLogger(__name__)

class QueryRewritingRAG(BaseRAG):
    """Implementation of RAG for query rewriting."""

    def __init__(
        self,
        template_name: str,
        collection_name: str,
    ) -> None:
        """Initialize known context with context and metadata."""
        # Call parent's init with interface parameters
        self.template_name = template_name
        self.collection_name = collection_name


    def rewrite_query(self, conversation_queries: list[str]) -> ResponseWrapper:
        """Rewrites the last question of the conversation using the LLM."""
        system_prompt = "You are a helpful assistant that rewrites follow-up questions in conversations. Your task is to rewrite the **last user question** so that it stands on its own, without needing the previous conversation for context. Make sure the rewritten question is clear, specific, and suitable for retrieving relevant documents."
        
        # Create prompt collection using the for query rewriting
        prompt_collection = PromptCollection.create_prompts(
            sys_prompts=system_prompt,
            user_prompts=conversation_queries,
            template_name="./src/exp/prompts/templates/query_rewriting.j2",
        )

        if not self.runner:
            raise ValueError("No LLM runner provided for generating hypothetical answers.")

        # Get the response from the LLM using the main runner
        result = self.runner.run(prompt_collection)
        if not isinstance(result, ResponseWrapper):
            raise TypeError("Expected result to be a ResponseWrapper, got {}".format(type(result)))
        return result

    def run(
        self,
        runner: BatchInferenceRunner,
        sys_prompt: str,
        user_prompts: list[str] = [],
    ) -> ResponseWrapper:
        
        # Create prompt collection using the for query rewriting
        prompt_collection = PromptCollection.create_prompts(
            sys_prompts=sys_prompt,
            user_prompts=user_prompts,
            template_name=self.template_name,
        )

        return runner.run(prompt_collection)
