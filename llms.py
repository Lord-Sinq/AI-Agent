"""
LLM Manager Module

This module provides a unified interface for interacting with Large Language Models (LLMs),
with primary support for Azure OpenAI. It handles configuration, authentication, and API
calls to generate text completions using chat-based models.

Environment Variables Required for Azure OpenAI:
    AZURE_OPENAI_ENDPOINT: Your Azure OpenAI endpoint URL
    AZURE_OPENAI_API_KEY: Your Azure OpenAI API key
    AZURE_DEPLOYMENT_NAME: Name of your deployed model
    AZURE_API_VERSION: API version (defaults to "2024-02-15-preview")
    DEFAULT_MODEL: Default model to use (defaults to "gpt-4o-mini")

"""

import os
import json
import requests
from datetime import datetime
from pathlib import Path
from typing import Optional


class LLMManager:
    """
    Initialize the LLM Manager with Azure OpenAI configuration.

    Loads configuration from environment variables and prints status information.
    Currently uses Azure OpenAI as the primary provider (OpenAI API key parameter
    is reserved for future implementation).

    Args:
        openai_api_key (Optional[str]): Legacy parameter for OpenAI API key.
            Currently unused as the manager prioritizes Azure OpenAI.

    Environment Variables Used:
        AZURE_OPENAI_ENDPOINT: Required - Azure endpoint URL
        AZURE_OPENAI_API_KEY: Required - Azure API key
        AZURE_DEPLOYMENT_NAME: Required - Deployment name
        AZURE_API_VERSION: Optional - Defaults to "2024-02-15-preview"
        DEFAULT_MODEL: Optional - Defaults to "gpt-4o-mini"
        SAVE_RESPONSES: Optional - Set to "true" to save all responses (default: true)

    Prints:
        INFO messages showing the configuration status including endpoint,
        deployment name, API version, and whether API key is configured
    """

    def __init__(self, openai_api_key: Optional[str] = None):
        # Azure OpenAI configuration
        self.azure_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
        self.azure_api_key = os.getenv("AZURE_OPENAI_API_KEY")
        self.azure_deployment_name = os.getenv("AZURE_DEPLOYMENT_NAME")
        self.azure_api_version = os.getenv("AZURE_API_VERSION", "2024-02-15-preview")

        self.default_model = os.getenv("DEFAULT_MODEL", "gpt-4o-mini")
        self.save_responses = os.getenv("SAVE_RESPONSES", "true").lower() == "true"

        # Create responses directory if saving is enabled
        if self.save_responses:
            self.responses_dir = Path.cwd() / "responses"
            self.responses_dir.mkdir(exist_ok=True)
            print(f"[INFO] Responses will be saved to: {self.responses_dir}")

        print(f"[INFO] Endpoint: {self.azure_endpoint}")
        print(f"[INFO] Deployment: {self.azure_deployment_name}")
        print(f"[INFO] API Version: {self.azure_api_version}")
        print(f"[INFO] API Key configured: {bool(self.azure_api_key)}")
        print(f"[INFO] Save responses: {self.save_responses}")

    def _save_response(self, prompt: str, response: dict, model: str, max_tokens: int) -> None:
        """
        Save the LLM response to a file in the responses directory.

        Args:
            prompt (str): The input prompt sent to the LLM
            response (dict): The response from the LLM
            model (str): The model used for generation
            max_tokens (int): The max_tokens parameter used
        """
        if not self.save_responses:
            return

        # Create timestamp for filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]

        # Create a descriptive filename
        prompt_preview = prompt[:50].replace(" ", "_").replace("\n", "").replace("/", "_")
        filename = f"{timestamp}_{model}_{prompt_preview}.json"
        filepath = self.responses_dir / filename

        # Prepare data to save
        response_data = {
            "timestamp": datetime.now().isoformat(),
            "model": model,
            "max_tokens": max_tokens,
            "prompt": prompt,
            "response": response,
            "response_text": response.get("text", ""),
            "response_length": len(response.get("text", "")),
        }

        # Save to file
        try:
            with open(filepath, "w", encoding="utf-8") as f:
                json.dump(response_data, f, indent=2, ensure_ascii=False)
            print(f"[INFO] Response saved to: {filepath}")
        except Exception as e:
            print(f"[WARNING] Failed to save response: {e}")

    def generate(self, prompt: str, model: Optional[str] = None, provider: Optional[str] = None, max_tokens: int = 4000) -> dict:
        """
        Generate text completion from the configured LLM provider.

        Sends a prompt to the LLM and returns the generated response. Currently
        supports Azure OpenAI provider. Automatically selects the provider based
        on available configuration if not specified.

        Args:
            prompt (str): The input text prompt to send to the LLM.
            model (Optional[str]): Override the default model to use.
                If None, uses self.azure_deployment_name.
            provider (Optional[str]): Specify which provider to use.
                Currently only "azure" is supported. If None, automatically
                selects based on available configuration.
            max_tokens (int): Maximum number of tokens to generate. Default 4000.

        Returns:
            dict: A dictionary containing:
                - "text": The generated text response from the LLM
                - "raw": The complete JSON response from the API

        Raises:
            RuntimeError: If no provider is configured, if Azure OpenAI is not
                properly configured, if the deployment doesn't support chat
                completions, or if the API call fails.
        """
        model = model or self.azure_deployment_name

        if provider is None:
            if self.azure_endpoint and self.azure_api_key and self.azure_deployment_name:
                provider = "azure"
            else:
                raise RuntimeError(
                    "No LLM provider configured. Please set:\n" "  AZURE_OPENAI_ENDPOINT, AZURE_OPENAI_API_KEY, AZURE_DEPLOYMENT_NAME\n"
                )

        if provider == "azure":
            if not (self.azure_endpoint and self.azure_api_key and self.azure_deployment_name):
                raise RuntimeError("Azure OpenAI not fully configured")

            # Clean up endpoint
            base_endpoint = self.azure_endpoint.rstrip("/")

            # Construct URL for chat completions
            url = f"{base_endpoint}/openai/deployments/{self.azure_deployment_name}/chat/completions?api-version={self.azure_api_version}"

            print(f"[INFO] Calling Azure OpenAI with max_tokens={max_tokens}...")

            headers = {
                "api-key": self.azure_api_key,
                "Content-Type": "application/json",
            }
            payload = {
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": max_tokens,
                "temperature": 0.7,
            }

            try:
                r = requests.post(url, json=payload, headers=headers, timeout=60)
                r.raise_for_status()
                j = r.json()

                # Debug: print response structure
                print(f"[DEBUG] Response keys: {j.keys()}")

                # Handle different response formats
                text = None
                if "choices" in j and len(j["choices"]) > 0:
                    choice = j["choices"][0]
                    if "message" in choice and "content" in choice["message"]:
                        text = choice["message"]["content"]
                    elif "text" in choice:
                        text = choice["text"]
                    elif "content" in choice:
                        text = choice["content"]

                if not text:
                    print(f"[WARNING] Unexpected response format: {json.dumps(j, indent=2)[:500]}")
                    text = ""

                print(f"[INFO] Generated {len(text)} characters of text")

                # Create response dictionary
                response = {"text": text, "raw": j}

                # Save response if enabled
                self._save_response(prompt, response, model, max_tokens)

                return response

            except requests.exceptions.RequestException as e:
                if hasattr(r, "status_code") and r.status_code == 404:
                    # Provide helpful suggestions
                    working_models = ["gpt-4o-mini", "gpt-35-turbo", "gpt-4o", "gpt-4"]
                    raise RuntimeError(
                        f"\n❌ Deployment '{self.azure_deployment_name}' not found or doesn't support chat completions.\n\n"
                        f"Try one of these verified chat models in your .env file:\n"
                        f"  AZURE_DEPLOYMENT_NAME=gpt-4o-mini  (Recommended)\n"
                        f"  AZURE_DEPLOYMENT_NAME=gpt-35-turbo (Faster/Cheaper)\n"
                        f"  AZURE_DEPLOYMENT_NAME=gpt-4o       (Most Capable)\n\n"
                        f"Your current deployment: {self.azure_deployment_name}\n"
                        f"Run 'python main.py --list-deployments' to see all available models."
                    ) from e
                raise RuntimeError(f"Azure OpenAI API call failed: {str(e)}") from e

        raise RuntimeError(f"Unknown provider: {provider}")

    def list_azure_deployments(self) -> dict:
        """
        List available Azure OpenAI models that support chat completions.

        Queries the Azure OpenAI API to retrieve all available models and filters
        them to show only those capable of chat completions (GPT, chat, instruct,
        turbo, or deepseek variants).

        Returns:
            dict: A dictionary containing:
                - "route": The API endpoint used for listing models
                - "api_version": The API version used
                - "items": List of chat-capable model dictionaries with 'id' and
                  'chat_completion' fields
                - "raw": The complete JSON response from the API

        Raises:
            RuntimeError: If Azure endpoint or API key is not configured, or if
                the API request fails.

        Prints:
            A formatted list of available chat-capable models, showing up to 20
            models with proper formatting and counts.
        """
        if not (self.azure_endpoint and self.azure_api_key):
            raise RuntimeError("Azure OpenAI listing requires AZURE_OPENAI_ENDPOINT and AZURE_OPENAI_API_KEY")

        base_endpoint = self.azure_endpoint.rstrip("/")
        headers = {"api-key": self.azure_api_key, "Content-Type": "application/json"}
        list_api_version = "2024-02-15-preview"

        # List all models (this endpoint works for you)
        url = f"{base_endpoint}/openai/models?api-version={list_api_version}"

        try:
            r = requests.get(url, headers=headers, timeout=30)
            if r.status_code == 200:
                j = r.json()
                data = j.get("data", [])

                # Filter for chat-capable models
                chat_models = []
                chat_keywords = ["gpt", "chat", "instruct", "turbo", "deepseek"]

                for item in data:
                    if isinstance(item, dict):
                        model_id = item.get("id")
                        if model_id and any(keyword in model_id.lower() for keyword in chat_keywords):
                            chat_models.append({"id": model_id, "chat_completion": True})

                # Sort and display
                chat_models.sort(key=lambda x: x["id"])

                print(f"\n✅ Found {len(chat_models)} chat-capable models:")
                print("=" * 50)
                for model in chat_models[:20]:  # Show first 20
                    print(f"  • {model['id']}")
                if len(chat_models) > 20:
                    print(f"  ... and {len(chat_models) - 20} more")

                return {
                    "route": "/openai/models",
                    "api_version": list_api_version,
                    "items": chat_models,
                    "raw": j,
                }
            else:
                raise RuntimeError(f"Failed to list models: {r.status_code} - {r.text}")
        except Exception as e:
            raise RuntimeError(f"Could not list Azure OpenAI models: {e}") from e
