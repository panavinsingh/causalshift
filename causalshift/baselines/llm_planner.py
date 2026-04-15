"""LLM planner baselines via raw HTTP — no broken SDKs.

All providers use direct HTTP requests for reliability and speed.
  - AWS Bedrock: boto3 (works correctly)
  - Azure OpenAI: raw requests (SDK is broken on Azure endpoint routing)
  - GCP Vertex AI: raw requests with ADC token
"""

from __future__ import annotations

import json
import os
import re
import subprocess
from abc import ABC, abstractmethod

import numpy as np
import requests


SYSTEM_PROMPT_PRIVILEGED = (
    "You are an agent. State has S1 (cause) and S2 (effect). "
    "S1 is random. S2 = S1 XOR noise. Reward = 1 if action matches S1. "
    "Output ONLY: {\"action\": 0} or {\"action\": 1}"
)

SYSTEM_PROMPT_BLACKBOX = (
    "You observe 2 binary values. Pick action 0 or 1 to maximize reward. "
    "Output ONLY: {\"action\": 0} or {\"action\": 1}"
)

SYSTEM_PROMPT_COT = (
    "You are an agent. State has S1 (cause) and S2 (effect). "
    "S1 is random. S2 = S1 XOR noise. Reward = 1 if action matches S1. "
    "Think step by step, then output: {\"action\": 0} or {\"action\": 1}"
)


class LLMPlanner(ABC):
    def __init__(self, condition: str = "privileged"):
        assert condition in ("blackbox", "privileged", "cot")
        self.condition = condition
        self.system_prompt = {
            "blackbox": SYSTEM_PROMPT_BLACKBOX,
            "privileged": SYSTEM_PROMPT_PRIVILEGED,
            "cot": SYSTEM_PROMPT_COT,
        }[condition]

    @abstractmethod
    def get_action(self, state: np.ndarray) -> int:
        pass

    def _parse_action(self, text: str) -> int:
        matches = re.findall(r'"action"\s*:\s*(\d)', text)
        if matches:
            return int(matches[-1])
        return 0


class BedrockClaudePlanner(LLMPlanner):
    """Claude via AWS Bedrock — raw HTTP with SigV4 (boto3 hangs)."""

    def __init__(self, model_id: str, condition: str = "privileged"):
        super().__init__(condition)
        self.model_id = model_id
        self.region = os.environ.get("AWS_REGION", "us-east-1")

        from botocore.credentials import Credentials
        self._creds = Credentials(
            os.environ["AWS_ACCESS_KEY_ID"],
            os.environ["AWS_SECRET_ACCESS_KEY"],
        )
        self._url = (
            f"https://bedrock-runtime.{self.region}.amazonaws.com"
            f"/model/{requests.utils.quote(self.model_id, safe='')}/invoke"
        )
        # Persistent session for connection pooling — avoids SSL handshake per call
        self._session = requests.Session()

    def get_action(self, state: np.ndarray) -> int:
        import time as _time
        from botocore.auth import SigV4Auth
        from botocore.awsrequest import AWSRequest

        body = json.dumps({
            "anthropic_version": "bedrock-2023-05-31",
            "max_tokens": 100000,
            "temperature": 1,
            "thinking": {"type": "adaptive"},
            "output_config": {"effort": "high"},
            "system": self.system_prompt,
            "messages": [{"role": "user", "content": f"S1={int(state[0])}, S2={int(state[1])}"}],
        })
        for attempt in range(5):
            req = AWSRequest(method="POST", url=self._url, data=body, headers={"Content-Type": "application/json"})
            SigV4Auth(self._creds, "bedrock", self.region).add_auth(req)
            r = self._session.post(self._url, headers=dict(req.headers), data=body, timeout=60)
            if r.status_code == 200:
                break
            _time.sleep(2 ** attempt)
        r.raise_for_status()
        content = r.json()["content"]
        text_parts = [b["text"] for b in content if b.get("type") == "text"]
        return self._parse_action(" ".join(text_parts) if text_parts else str(content))


class AzureOpenAIPlanner(LLMPlanner):
    """GPT-5.4 via Azure OpenAI — raw HTTP (SDK routing is broken)."""

    def __init__(self, condition: str = "privileged"):
        super().__init__(condition)
        self.endpoint = os.environ["AZURE_OPENAI_ENDPOINT"].rstrip("/")
        self.api_key = os.environ["AZURE_OPENAI_API_KEY"]
        self.deployment = os.environ.get("AZURE_OPENAI_DEPLOYMENT", "gpt-5.4")
        self.api_version = os.environ.get("AZURE_OPENAI_API_VERSION", "2025-04-01-preview")
        self.url = f"{self.endpoint}/openai/deployments/{self.deployment}/chat/completions?api-version={self.api_version}"

    def get_action(self, state: np.ndarray) -> int:
        body = {
            "messages": [
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": f"S1={int(state[0])}, S2={int(state[1])}"},
            ],
            "max_completion_tokens": 100000,
            "reasoning_effort": "high",
        }
        r = requests.post(
            self.url,
            headers={"api-key": self.api_key, "Content-Type": "application/json"},
            json=body,
            timeout=300,
        )
        r.raise_for_status()
        return self._parse_action(r.json()["choices"][0]["message"]["content"])


class VertexGeminiPlanner(LLMPlanner):
    """Gemini via GCP Vertex AI — raw HTTP with ADC refresh token."""

    def __init__(self, model_name: str = "gemini-3.1-pro-preview", condition: str = "privileged"):
        super().__init__(condition)
        self.model_name = model_name
        self.project = os.environ.get("GOOGLE_CLOUD_PROJECT", "your-gcp-project")
        # Gemini 3.1 Pro requires global endpoint, not regional
        self.location = "global" if "3.1" in model_name else os.environ.get("GOOGLE_CLOUD_REGION", "us-central1")
        self._refresh_token()

    def _refresh_token(self):
        adc_path = os.path.join(os.environ.get("APPDATA", ""), "gcloud", "application_default_credentials.json")
        with open(adc_path) as f:
            adc = json.load(f)
        r = requests.post(
            "https://oauth2.googleapis.com/token",
            data={
                "client_id": adc["client_id"],
                "client_secret": adc["client_secret"],
                "refresh_token": adc["refresh_token"],
                "grant_type": "refresh_token",
            },
            timeout=10,
        )
        r.raise_for_status()
        self.token = r.json()["access_token"]

    def get_action(self, state: np.ndarray) -> int:
        if self.location == "global":
            host = "aiplatform.googleapis.com"
        else:
            host = f"{self.location}-aiplatform.googleapis.com"
        url = (
            f"https://{host}/v1/"
            f"projects/{self.project}/locations/{self.location}/"
            f"publishers/google/models/{self.model_name}:generateContent"
        )
        body = {
            "contents": [{"role": "user", "parts": [{"text": f"{self.system_prompt}\n\nS1={int(state[0])}, S2={int(state[1])}"}]}],
            "generationConfig": {
                "maxOutputTokens": 65536,
                "temperature": 1.0,
                "thinkingConfig": {"thinkingLevel": "HIGH"},
            },
        }
        r = requests.post(
            url,
            headers={"Authorization": f"Bearer {self.token}", "Content-Type": "application/json"},
            json=body,
            timeout=120,
        )
        if r.status_code == 401:
            self._refresh_token()
            r = requests.post(
                url,
                headers={"Authorization": f"Bearer {self.token}", "Content-Type": "application/json"},
                json=body,
                timeout=120,
            )
        r.raise_for_status()
        candidate = r.json()["candidates"][0]
        parts = candidate.get("content", {}).get("parts", [])
        # With thinkingConfig, response has thought parts + text parts.
        # Extract only non-thought text parts.
        text_parts = [p["text"] for p in parts if "thought" not in p and "text" in p]
        return self._parse_action(" ".join(text_parts) if text_parts else str(parts))


class BedrockDeepSeekPlanner(LLMPlanner):
    """DeepSeek V3.2 via AWS Bedrock — serverless, OpenAI-compatible format."""

    def __init__(self, condition: str = "privileged"):
        super().__init__(condition)
        self.region = os.environ.get("AWS_REGION", "us-east-1")

        from botocore.credentials import Credentials
        self._creds = Credentials(
            os.environ["AWS_ACCESS_KEY_ID"],
            os.environ["AWS_SECRET_ACCESS_KEY"],
        )
        self._model_id = "deepseek.v3.2"
        self._url = (
            f"https://bedrock-runtime.{self.region}.amazonaws.com"
            f"/model/{requests.utils.quote(self._model_id, safe='')}/invoke"
        )

    def get_action(self, state: np.ndarray) -> int:
        import time as _time
        # DeepSeek V3.2 thinking mode: no temperature/top_p when thinking enabled
        body = json.dumps({
            "messages": [
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": f"S1={int(state[0])}, S2={int(state[1])}"},
            ],
            "max_tokens": 100000,
        })
        from botocore.awsrequest import AWSRequest
        from botocore.auth import SigV4Auth
        session = requests.Session()
        for attempt in range(5):
            req = AWSRequest(method="POST", url=self._url, data=body, headers={"Content-Type": "application/json"})
            SigV4Auth(self._creds, "bedrock", self.region).add_auth(req)
            r = session.post(self._url, headers=dict(req.headers), data=body, timeout=120)
            if r.status_code == 200:
                break
            _time.sleep(2 ** attempt)
        r.raise_for_status()
        data = r.json()
        if "choices" in data:
            return self._parse_action(data["choices"][0]["message"]["content"])
        content = data.get("content", [])
        text = " ".join(b.get("text", "") for b in content if isinstance(b, dict))
        return self._parse_action(text)


class VertexQwenPlanner(LLMPlanner):
    """Qwen3-235B via GCP Vertex AI MaaS — serverless, global endpoint."""

    def __init__(self, condition: str = "privileged"):
        super().__init__(condition)
        self.project = os.environ.get("GOOGLE_CLOUD_PROJECT", "your-gcp-project")
        self.model_id = "qwen3-235b-a22b-instruct-2507-maas"
        self._refresh_token()

    def _refresh_token(self):
        adc_path = os.path.join(os.environ.get("APPDATA", ""), "gcloud", "application_default_credentials.json")
        with open(adc_path) as f:
            adc = json.load(f)
        r = requests.post(
            "https://oauth2.googleapis.com/token",
            data={
                "client_id": adc["client_id"],
                "client_secret": adc["client_secret"],
                "refresh_token": adc["refresh_token"],
                "grant_type": "refresh_token",
            },
            timeout=10,
        )
        r.raise_for_status()
        self.token = r.json()["access_token"]

    def get_action(self, state: np.ndarray) -> int:
        import time as _time
        url = (
            f"https://aiplatform.googleapis.com/v1/"
            f"projects/{self.project}/locations/global/"
            f"publishers/qwen/models/{self.model_id}:generateContent"
        )
        body = {
            "contents": [{"role": "user", "parts": [{"text": f"{self.system_prompt}\n\nS1={int(state[0])}, S2={int(state[1])}"}]}],
            "generationConfig": {"maxOutputTokens": 16384},
        }
        for attempt in range(5):
            r = requests.post(
                url,
                headers={"Authorization": f"Bearer {self.token}", "Content-Type": "application/json"},
                json=body,
                timeout=120,
            )
            if r.status_code == 401:
                self._refresh_token()
                continue
            if r.status_code == 200:
                break
            _time.sleep(2 ** attempt)
        r.raise_for_status()
        candidate = r.json()["candidates"][0]
        parts = candidate.get("content", {}).get("parts", [])
        text_parts = [p["text"] for p in parts if "thought" not in p and "text" in p]
        return self._parse_action(" ".join(text_parts) if text_parts else str(parts))


PLANNER_REGISTRY = {
    # Closed-source
    "claude-opus-4.6": lambda cond: BedrockClaudePlanner(os.environ.get("BEDROCK_CLAUDE_MODEL_ARN", "anthropic.claude-opus-4-6-v1"), cond),
    "gpt-5.4": lambda cond: AzureOpenAIPlanner(cond),
    "gemini-3.1-pro": lambda cond: VertexGeminiPlanner("gemini-3.1-pro-preview", cond),
    # Open-source (via Bedrock managed API)
    "deepseek-v3.2": lambda cond: BedrockDeepSeekPlanner(cond),
    "qwen3-235b": lambda cond: VertexQwenPlanner(cond),
}
