"""Unit tests for classifier LLM cost tracking, unified client, and LLMVisionDetector."""

import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch
import pytest
from PIL import Image

from classifier.llm.tracker import CostTracker, estimate_cost, price_for
from classifier.llm.client import (
    LLMClient,
    LLMResponse,
    infer_provider,
    resolve_model_for_provider,
    resolve_provider_chain,
    track_llm_call,
)
from classifier.image_model.llm_detector import LLMVisionDetector


# -----------------------------------------------------------------------------
# CostTracker Tests
# -----------------------------------------------------------------------------

def test_price_for_models():
    assert price_for("claude-sonnet-5-5") == (2.20, 11.00)
    assert price_for("claude-sonnet-4-6") == (3.00, 15.00)
    assert price_for("claude-haiku-4-5") == (1.00, 5.00)
    assert price_for("gpt-4o") == (2.50, 10.00)
    assert price_for("gpt-4o-mini") == (0.15, 0.60)
    assert price_for("o3-mini") == (1.10, 4.40)
    assert price_for("unsupported-xyz-model") is None


def test_estimate_cost():
    # 1M input ($3), 1M output ($15) on sonnet-4-6 = $18
    assert estimate_cost("claude-sonnet-4-6", 1_000_000, 1_000_000) == pytest.approx(18.0)
    # 1M input ($2.50), 1M output ($10.00) on gpt-4o = $12.50
    assert estimate_cost("gpt-4o", 1_000_000, 1_000_000) == pytest.approx(12.50)
    # Unknown model -> None
    assert estimate_cost("unknown-model", 100, 100) is None


def test_cost_tracker_file_logging(tmp_path: Path):
    log_file = tmp_path / "test_costs.txt"
    tracker = CostTracker(log_file=str(log_file))

    res1 = tracker.record_call(
        model_id="claude-sonnet-4-6",
        tokens_in=1000,
        tokens_out=500,
        tokens_thinking=100,
        latency_ms=350.2,
        tag="CLASSIFIER:anthropic",
        op="detect_fridge_items",
    )

    assert log_file.exists()
    content = log_file.read_text(encoding="utf-8")
    assert "[CLASSIFIER:anthropic]" in content
    assert "model=claude-sonnet-4-6" in content
    assert "tokens: in=1000, out=500, thinking=100, total=1500" in content
    assert "latency=350.2ms" in content

    # Second call accumulates
    res2 = tracker.record_call(
        model_id="gpt-4o",
        tokens_in=500,
        tokens_out=200,
        tokens_thinking=0,
        latency_ms=150.0,
        tag="CLASSIFIER:openai",
        op="detect_fridge_items",
    )

    totals = tracker.totals()
    assert totals["calls"] == 2
    assert totals["in"] == 1500
    assert totals["out"] == 700
    assert totals["thinking"] == 100
    assert totals["usd"] > 0

    lines = log_file.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 2
    assert "calls=2" in lines[1]


# -----------------------------------------------------------------------------
# LLMClient Tests
# -----------------------------------------------------------------------------

def test_infer_provider():
    assert infer_provider("claude-sonnet-5-5") == "anthropic"
    assert infer_provider("gpt-4o") == "openai"
    assert infer_provider("o1-mini") == "openai"
    assert infer_provider("my-model", explicit_provider="openai") == "openai"
    assert infer_provider("custom-unknown") == "anthropic"  # default fallback


def test_llm_client_complete_vision_anthropic(tmp_path: Path):
    log_file = tmp_path / "anthropic_costs.txt"
    tracker = CostTracker(log_file=str(log_file))
    client = LLMClient(tracker=tracker)

    # Mock Anthropic client
    mock_anthropic = MagicMock()
    mock_block = SimpleNamespace(type="text", text='{"detections": [{"label": "milk", "confidence": 0.9, "box_2d": [10, 10, 100, 100]}]}')
    mock_resp = SimpleNamespace(
        content=[mock_block],
        usage=SimpleNamespace(input_tokens=800, output_tokens=150, thinking_tokens=40),
    )
    mock_anthropic.messages.create.return_value = mock_resp
    client._clients["anthropic:mock_key"] = mock_anthropic

    response = client.complete_vision(
        prompt="Detect items",
        image_b64="fake_base64",
        system_prompt="System prompt",
        model_id="claude-sonnet-4-6",
        provider="anthropic",
        api_key="mock_key_12345678",
    )

    assert isinstance(response, LLMResponse)
    assert response.provider == "anthropic"
    assert response.input_tokens == 800
    assert response.output_tokens == 150
    assert response.thinking_tokens == 40
    assert response.total_tokens == 950
    assert response.cost_usd is not None
    assert "milk" in response.text

    # Verify file was written
    log_content = log_file.read_text(encoding="utf-8")
    assert "thinking=40" in log_content
    assert "model=claude-sonnet-4-6" in log_content


def test_llm_client_complete_vision_openai(tmp_path: Path):
    log_file = tmp_path / "openai_costs.txt"
    tracker = CostTracker(log_file=str(log_file))
    client = LLMClient(tracker=tracker)

    # Mock OpenAI client with reasoning tokens
    mock_openai = MagicMock()
    mock_choice = SimpleNamespace(
        message=SimpleNamespace(content='{"detections": [{"label": "apple", "confidence": 0.95, "box_2d": [20, 20, 80, 80]}]}')
    )
    mock_resp = SimpleNamespace(
        choices=[mock_choice],
        usage=SimpleNamespace(
            prompt_tokens=600,
            completion_tokens=250,
            total_tokens=850,
            completion_tokens_details=SimpleNamespace(reasoning_tokens=80),
        ),
    )
    mock_openai.chat.completions.create.return_value = mock_resp
    client._clients["openai:mock_key:"] = mock_openai

    response = client.complete_vision(
        prompt="Detect items",
        image_b64="fake_base64",
        model_id="gpt-4o",
        provider="openai",
        api_key="mock_key_12345678",
    )

    assert isinstance(response, LLMResponse)
    assert response.provider == "openai"
    assert response.input_tokens == 600
    assert response.output_tokens == 250
    assert response.thinking_tokens == 80
    assert response.total_tokens == 850
    assert response.cost_usd is not None
    assert "apple" in response.text

    # Verify file was written
    log_content = log_file.read_text(encoding="utf-8")
    assert "thinking=80" in log_content
    assert "model=gpt-4o" in log_content


def test_track_llm_call_decorator(tmp_path: Path):
    log_file = tmp_path / "decorator_costs.txt"
    tracker = CostTracker(log_file=str(log_file))

    with patch("classifier.llm.client.get_cost_tracker", return_value=tracker):
        @track_llm_call(op="custom_test_op", tag="CLASSIFIER:test")
        def dummy_llm_call(model_id: str):
            return SimpleNamespace(
                usage=SimpleNamespace(prompt_tokens=400, completion_tokens=100)
            )

        dummy_llm_call(model_id="gpt-4o-mini")

        assert log_file.exists()
        log_content = log_file.read_text(encoding="utf-8")
        assert "op=custom_test_op" in log_content
        assert "model=gpt-4o-mini" in log_content
        assert "in=400, out=100" in log_content


# -----------------------------------------------------------------------------
# LLMVisionDetector & ClaudeVisionDetector Tests
# -----------------------------------------------------------------------------

def test_llm_vision_detector_with_mock_client():
    mock_client = MagicMock(spec=LLMClient)
    mock_client.complete_vision.return_value = LLMResponse(
        text=json.dumps({
            "detections": [
                {"label": "cheese", "confidence": 0.92, "box_2d": [50, 50, 200, 200]}
            ]
        }),
        model_id="claude-sonnet-5-5",
        provider="anthropic",
        input_tokens=500,
        output_tokens=100,
        thinking_tokens=0,
        total_tokens=600,
        cost_usd=0.0022,
        latency_ms=250.0,
    )

    detector = LLMVisionDetector(
        model_id="claude-sonnet-5-5",
        confidence_threshold=0.2,
        coord_format="pixel",
        llm_client=mock_client,
    )

    img = Image.new("RGB", (640, 480), color="white")
    results = detector.detect(img)

    assert len(results) == 1
    assert results[0].label == "cheese"
    assert results[0].confidence == 0.92
    assert results[0].bbox == [50.0, 50.0, 200.0, 200.0]
    assert mock_client.complete_vision.called


# -----------------------------------------------------------------------------
# Amazon Bedrock & Fallback Tests
# -----------------------------------------------------------------------------

def test_infer_provider_bedrock(monkeypatch):
    assert infer_provider("global.anthropic.claude-sonnet-4-6", explicit_provider="bedrock") == "bedrock"
    assert infer_provider("bedrock-claude") == "bedrock"

    monkeypatch.setenv("CLASSIFIER_LLM_PROVIDER", "bedrock")
    assert infer_provider("claude-sonnet-5-5") == "bedrock"


def test_resolve_model_for_provider():
    assert "sonnet" in resolve_model_for_provider("claude-sonnet-5-5", "bedrock")
    assert resolve_model_for_provider("global.anthropic.claude-sonnet-4-6", "bedrock") == "global.anthropic.claude-sonnet-4-6"
    assert resolve_model_for_provider("global.anthropic.claude-sonnet-4-6", "anthropic") == "claude-sonnet-4-6"
    assert resolve_model_for_provider("claude-sonnet-5-5", "openai") == "gpt-4o-mini"


def test_resolve_provider_chain(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    monkeypatch.setenv("OPENAI_API_KEY", "test-openai-key")
    chain = resolve_provider_chain("bedrock")
    assert chain == ["bedrock", "anthropic", "openai"]

    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("AGENT_OPENAI_API_KEY", raising=False)
    chain = resolve_provider_chain("bedrock")
    assert chain == ["bedrock", "anthropic"]

    chain = resolve_provider_chain("anthropic")
    assert chain == ["anthropic"]


def test_llm_client_complete_vision_bedrock_success(tmp_path: Path):
    log_file = tmp_path / "bedrock_costs.txt"
    tracker = CostTracker(log_file=str(log_file))
    client = LLMClient(tracker=tracker)

    mock_bedrock = MagicMock()
    mock_block = SimpleNamespace(type="text", text='{"detections": [{"label": "yogurt", "confidence": 0.93, "box_2d": [15, 15, 80, 80]}]}')
    mock_resp = SimpleNamespace(
        content=[mock_block],
        usage=SimpleNamespace(input_tokens=700, output_tokens=120, thinking_tokens=0),
    )
    mock_bedrock.messages.create.return_value = mock_resp

    with patch.object(client, "_get_bedrock_client", return_value=mock_bedrock):
        response = client.complete_vision(
            prompt="Detect items",
            image_b64="fake_base64",
            model_id="global.anthropic.claude-sonnet-4-6",
            provider="bedrock",
        )

    assert response.provider == "bedrock"
    assert response.input_tokens == 700
    assert response.output_tokens == 120
    assert "yogurt" in response.text
    assert mock_bedrock.messages.create.called

    log_content = log_file.read_text(encoding="utf-8")
    assert "[CLASSIFIER:bedrock]" in log_content


def test_llm_client_bedrock_fallback_to_anthropic(monkeypatch, tmp_path: Path):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-anthropic-key")
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("AGENT_OPENAI_API_KEY", raising=False)

    log_file = tmp_path / "fallback_costs.txt"
    tracker = CostTracker(log_file=str(log_file))
    client = LLMClient(tracker=tracker)

    # Bedrock client raises an error (e.g. 404 / NotFoundError / auth failure)
    mock_bedrock = MagicMock()
    mock_bedrock.messages.create.side_effect = RuntimeError("Bedrock model not available or expired")

    # Native Anthropic client succeeds
    mock_anthropic = MagicMock()
    mock_block = SimpleNamespace(type="text", text='{"detections": [{"label": "butter", "confidence": 0.88, "box_2d": [10, 10, 50, 50]}]}')
    mock_resp = SimpleNamespace(
        content=[mock_block],
        usage=SimpleNamespace(input_tokens=650, output_tokens=110, thinking_tokens=0),
    )
    mock_anthropic.messages.create.return_value = mock_resp

    with patch.object(client, "_get_bedrock_client", return_value=mock_bedrock), \
         patch.object(client, "_get_anthropic_client", return_value=mock_anthropic):

        response = client.complete_vision(
            prompt="Detect items",
            image_b64="fake_base64",
            model_id="claude-sonnet-5-5",
            provider="bedrock",
        )

    # Should have attempted bedrock, failed, and seamlessly succeeded on anthropic
    assert mock_bedrock.messages.create.called
    assert mock_anthropic.messages.create.called
    assert response.provider == "anthropic"
    assert "butter" in response.text
    assert response.input_tokens == 650

    log_content = log_file.read_text(encoding="utf-8")
    assert "[CLASSIFIER:anthropic]" in log_content


def test_llm_client_text_bedrock_fallback_to_anthropic(monkeypatch, tmp_path: Path):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-anthropic-key")

    log_file = tmp_path / "fallback_text_costs.txt"
    tracker = CostTracker(log_file=str(log_file))
    client = LLMClient(tracker=tracker)

    mock_bedrock = MagicMock()
    mock_bedrock.messages.create.side_effect = Exception("Bedrock timeout")

    mock_anthropic = MagicMock()
    mock_block = SimpleNamespace(type="text", text="Hello from fallback Anthropic")
    mock_resp = SimpleNamespace(
        content=[mock_block],
        usage=SimpleNamespace(input_tokens=100, output_tokens=20, thinking_tokens=0),
    )
    mock_anthropic.messages.create.return_value = mock_resp

    with patch.object(client, "_get_bedrock_client", return_value=mock_bedrock), \
         patch.object(client, "_get_anthropic_client", return_value=mock_anthropic):

        response = client.complete_text(
            prompt="Hello",
            model_id="claude-sonnet-5-5",
            provider="bedrock",
        )

    assert mock_bedrock.messages.create.called
    assert mock_anthropic.messages.create.called
    assert response.provider == "anthropic"
    assert response.text == "Hello from fallback Anthropic"

