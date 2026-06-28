"""Tests for the delivery routing module."""

import pytest

from gateway.config import GatewayConfig, Platform
from gateway.delivery import DeliveryRouter, DeliveryTarget
from gateway.platforms.base import SendResult
from gateway.session import SessionSource


class TestParseTargetPlatformChat:
    def test_explicit_telegram_chat(self):
        target = DeliveryTarget.parse("telegram:12345")
        assert target.platform == Platform.TELEGRAM
        assert target.chat_id == "12345"
        assert target.is_explicit is True

    def test_platform_only_no_chat_id(self):
        target = DeliveryTarget.parse("discord")
        assert target.platform == Platform.DISCORD
        assert target.chat_id is None
        assert target.is_explicit is False

    def test_local_target(self):
        target = DeliveryTarget.parse("local")
        assert target.platform == Platform.LOCAL
        assert target.chat_id is None

    def test_origin_with_source(self):
        origin = SessionSource(platform=Platform.TELEGRAM, chat_id="789", thread_id="42")
        target = DeliveryTarget.parse("origin", origin=origin)
        assert target.platform == Platform.TELEGRAM
        assert target.chat_id == "789"
        assert target.thread_id == "42"
        assert target.is_origin is True

    def test_origin_without_source(self):
        target = DeliveryTarget.parse("origin")
        assert target.platform == Platform.LOCAL
        assert target.is_origin is True

    def test_unknown_platform(self):
        target = DeliveryTarget.parse("unknown_platform")
        assert target.platform == Platform.LOCAL


class TestTargetToStringRoundtrip:
    def test_origin_roundtrip(self):
        origin = SessionSource(platform=Platform.TELEGRAM, chat_id="111", thread_id="42")
        target = DeliveryTarget.parse("origin", origin=origin)
        assert target.to_string() == "origin"

    def test_local_roundtrip(self):
        target = DeliveryTarget.parse("local")
        assert target.to_string() == "local"

    def test_platform_only_roundtrip(self):
        target = DeliveryTarget.parse("discord")
        assert target.to_string() == "discord"

    def test_explicit_chat_roundtrip(self):
        target = DeliveryTarget.parse("telegram:999")
        s = target.to_string()
        assert s == "telegram:999"

        reparsed = DeliveryTarget.parse(s)
        assert reparsed.platform == Platform.TELEGRAM
        assert reparsed.chat_id == "999"


class TestCaseSensitiveChatIdParsing:
    """Test that chat IDs preserve their original case (issue #11768)."""
    
    def test_slack_uppercase_chat_id_preserved(self):
        """Slack channel IDs like C123ABC should preserve case."""
        target = DeliveryTarget.parse("slack:C123ABC")
        assert target.platform == Platform.SLACK
        assert target.chat_id == "C123ABC"  # Should NOT be lowercased to c123abc
        assert target.is_explicit is True
    
    def test_slack_chat_id_with_thread_preserved(self):
        """Slack channel:thread IDs should preserve case."""
        target = DeliveryTarget.parse("slack:C123ABC:thread123")
        assert target.platform == Platform.SLACK
        assert target.chat_id == "C123ABC"
        assert target.thread_id == "thread123"
    
    def test_matrix_room_id_preserved(self):
        """Matrix room IDs like !RoomABC:example.org should preserve case.
        
        Note: Matrix room IDs contain colons (e.g., !RoomABC:example.org).
        Due to the platform:chat_id:thread_id format, these are parsed as
        chat_id=!RoomABC and thread_id=example.org. This is a known limitation
        of the current format. The fix preserves case but doesn't change the
        parsing structure.
        """
        target = DeliveryTarget.parse("matrix:!RoomABC:example.org")
        assert target.platform == Platform.MATRIX
        # The room ID is split at the first colon after the platform prefix
        # This is a format limitation - the case is preserved but the structure is split
        assert target.chat_id == "!RoomABC"
        assert target.thread_id == "example.org"
    
    def test_mixed_case_chat_id_roundtrip(self):
        """Mixed-case chat IDs should survive parse-to_string roundtrip."""
        original = "telegram:ChatId123ABC"
        target = DeliveryTarget.parse(original)
        s = target.to_string()
        reparsed = DeliveryTarget.parse(s)
        assert reparsed.chat_id == "ChatId123ABC"


class TestPlatformNameCaseInsensitivity:
    """Test that platform names are case-insensitive."""
    
    def test_uppercase_platform_name(self):
        """Platform names should be case-insensitive."""
        target = DeliveryTarget.parse("TELEGRAM:12345")
        assert target.platform == Platform.TELEGRAM
        assert target.chat_id == "12345"
    
    def test_mixed_case_platform_name(self):
        """Mixed-case platform names should work."""
        target = DeliveryTarget.parse("TeleGram:12345")
        assert target.platform == Platform.TELEGRAM
        assert target.chat_id == "12345"

class RecordingAdapter:
    def __init__(self):
        self.calls = []
        self.ensure_dm_topic_calls = []

    async def send(self, chat_id, content, metadata=None):
        self.calls.append({"chat_id": chat_id, "content": content, "metadata": metadata})
        return {"success": True}

    async def ensure_dm_topic(self, chat_id, topic_name, force_create=False):
        self.ensure_dm_topic_calls.append(
            {"chat_id": chat_id, "topic_name": topic_name, "force_create": force_create}
        )
        return "38049"


class StaleTopicAdapter:
    def __init__(self):
        self.calls = []
        self.ensure_dm_topic_calls = []

    async def send(self, chat_id, content, metadata=None):
        self.calls.append({"chat_id": chat_id, "content": content, "metadata": dict(metadata or {})})
        if len(self.calls) == 1:
            return SendResult(success=False, error="Bad Request: message thread not found")
        return SendResult(success=True, message_id="fresh-message")

    async def ensure_dm_topic(self, chat_id, topic_name, force_create=False):
        self.ensure_dm_topic_calls.append(
            {"chat_id": chat_id, "topic_name": topic_name, "force_create": force_create}
        )
        return "38064" if force_create else "32343"


@pytest.mark.asyncio
async def test_explicit_telegram_private_thread_requires_reply_anchor(tmp_path, monkeypatch):
    monkeypatch.setattr("gateway.delivery.get_hermes_home", lambda: tmp_path)
    adapter = RecordingAdapter()
    router = DeliveryRouter(GatewayConfig(), adapters={Platform.TELEGRAM: adapter})
    target = DeliveryTarget.parse("telegram:722341991:32344")

    with pytest.raises(RuntimeError, match="requires telegram_reply_to_message_id"):
        await router._deliver_to_platform(target, "hello", metadata=None)

    assert adapter.calls == []


@pytest.mark.asyncio
async def test_named_telegram_private_topic_is_created_before_delivery(tmp_path, monkeypatch):
    monkeypatch.setattr("gateway.delivery.get_hermes_home", lambda: tmp_path)
    adapter = RecordingAdapter()
    router = DeliveryRouter(GatewayConfig(), adapters={Platform.TELEGRAM: adapter})
    target = DeliveryTarget.parse("telegram:722341991:Hermes API Test")

    await router._deliver_to_platform(target, "hello", metadata=None)

    assert adapter.ensure_dm_topic_calls == [
        {"chat_id": "722341991", "topic_name": "Hermes API Test", "force_create": False}
    ]
    assert adapter.calls == [
        {
            "chat_id": "722341991",
            "content": "hello",
            "metadata": {
                "thread_id": "38049",
                "telegram_dm_topic_created_for_send": True,
            },
        }
    ]


@pytest.mark.asyncio
async def test_named_telegram_private_topic_refreshes_stale_thread_id(tmp_path, monkeypatch):
    monkeypatch.setattr("gateway.delivery.get_hermes_home", lambda: tmp_path)
    adapter = StaleTopicAdapter()
    router = DeliveryRouter(GatewayConfig(), adapters={Platform.TELEGRAM: adapter})
    target = DeliveryTarget.parse("telegram:722341991:Personal")

    result = await router._deliver_to_platform(target, "hello", metadata=None)

    assert getattr(result, "message_id", None) == "fresh-message"
    assert adapter.ensure_dm_topic_calls == [
        {"chat_id": "722341991", "topic_name": "Personal", "force_create": False},
        {"chat_id": "722341991", "topic_name": "Personal", "force_create": True},
    ]
    assert [call["metadata"]["thread_id"] for call in adapter.calls] == ["32343", "38064"]
    assert all(call["metadata"]["telegram_dm_topic_created_for_send"] is True for call in adapter.calls)


@pytest.mark.asyncio
async def test_explicit_telegram_private_thread_uses_reply_fallback_with_anchor(tmp_path, monkeypatch):
    monkeypatch.setattr("gateway.delivery.get_hermes_home", lambda: tmp_path)
    adapter = RecordingAdapter()
    router = DeliveryRouter(GatewayConfig(), adapters={Platform.TELEGRAM: adapter})
    target = DeliveryTarget.parse("telegram:722341991:32344")

    await router._deliver_to_platform(
        target,
        "hello",
        metadata={"telegram_reply_to_message_id": "9001"},
    )

    assert adapter.calls == [
        {
            "chat_id": "722341991",
            "content": "hello",
            "metadata": {
                "telegram_reply_to_message_id": "9001",
                "thread_id": "32344",
                "telegram_dm_topic_reply_fallback": True,
            },
        }
    ]


@pytest.mark.asyncio
async def test_explicit_telegram_direct_messages_topic_metadata_is_respected(tmp_path, monkeypatch):
    monkeypatch.setattr("gateway.delivery.get_hermes_home", lambda: tmp_path)
    adapter = RecordingAdapter()
    router = DeliveryRouter(GatewayConfig(), adapters={Platform.TELEGRAM: adapter})
    target = DeliveryTarget.parse("telegram:722341991:32344")

    await router._deliver_to_platform(
        target,
        "hello",
        metadata={"telegram_direct_messages_topic_id": "32344"},
    )

    assert adapter.calls[0]["metadata"] == {"telegram_direct_messages_topic_id": "32344"}


@pytest.mark.asyncio
async def test_explicit_telegram_group_thread_does_not_mark_dm_fallback(tmp_path, monkeypatch):
    monkeypatch.setattr("gateway.delivery.get_hermes_home", lambda: tmp_path)
    adapter = RecordingAdapter()
    router = DeliveryRouter(GatewayConfig(), adapters={Platform.TELEGRAM: adapter})
    target = DeliveryTarget.parse("telegram:-100123:42")

    await router._deliver_to_platform(target, "hello", metadata=None)

    assert adapter.calls[0]["metadata"] == {"thread_id": "42"}


class FailingAdapter:
    async def send(self, chat_id, content, metadata=None):
        return SendResult(success=False, error="route failed", retryable=False)


@pytest.mark.asyncio
async def test_platform_send_failure_raises_for_delivery_result(tmp_path, monkeypatch):
    monkeypatch.setattr("gateway.delivery.get_hermes_home", lambda: tmp_path)
    router = DeliveryRouter(GatewayConfig(), adapters={Platform.TELEGRAM: FailingAdapter()})
    target = DeliveryTarget.parse("telegram:722341991:32344")

    with pytest.raises(RuntimeError, match="route failed"):
        await router._deliver_to_platform(target, "hello", metadata={"telegram_reply_to_message_id": "9001"})


# ---------------------------------------------------------------------------
# Cron output truncation / adapter-aware chunking (issue #50126)
# ---------------------------------------------------------------------------

class ChunkingAdapter:
    """Adapter that declares splits_long_messages=True (like Discord/Telegram)."""
    splits_long_messages = True

    def __init__(self):
        self.calls = []

    async def send(self, chat_id, content, metadata=None):
        self.calls.append({"chat_id": chat_id, "content": content, "metadata": metadata})
        return {"success": True}


class NonChunkingAdapter:
    """Adapter without splits_long_messages (default False — legacy behavior)."""

    def __init__(self):
        self.calls = []

    async def send(self, chat_id, content, metadata=None):
        self.calls.append({"chat_id": chat_id, "content": content, "metadata": metadata})
        return {"success": True}


@pytest.mark.asyncio
async def test_long_output_truncated_for_non_chunking_adapter(tmp_path, monkeypatch):
    """Non-chunking adapters receive truncated content with a footer + file save."""
    monkeypatch.setattr("gateway.delivery.get_hermes_home", lambda: tmp_path)
    adapter = NonChunkingAdapter()
    router = DeliveryRouter(GatewayConfig(), adapters={Platform.DISCORD: adapter})
    target = DeliveryTarget.parse("discord:123")

    long_content = "x" * 5000
    await router._deliver_to_platform(target, long_content, metadata={"job_id": "job1"})

    delivered = adapter.calls[0]["content"]
    assert len(delivered) < 5000  # was truncated
    assert "truncated" in delivered.lower()
    assert "full output saved to" in delivered
    # Full output was saved to disk
    saved_files = list(tmp_path.glob("cron/output/job1_*.txt"))
    assert len(saved_files) == 1
    assert saved_files[0].read_text() == long_content


@pytest.mark.asyncio
async def test_long_output_preserved_for_chunking_adapter(tmp_path, monkeypatch):
    """Chunking adapters (splits_long_messages=True) receive the FULL content."""
    monkeypatch.setattr("gateway.delivery.get_hermes_home", lambda: tmp_path)
    adapter = ChunkingAdapter()
    router = DeliveryRouter(GatewayConfig(), adapters={Platform.DISCORD: adapter})
    target = DeliveryTarget.parse("discord:123")

    long_content = "x" * 5000
    await router._deliver_to_platform(target, long_content, metadata={"job_id": "job2"})

    delivered = adapter.calls[0]["content"]
    assert delivered == long_content  # NOT truncated — adapter handles chunking
    assert "truncated" not in delivered.lower()
    # Full output still saved to disk as audit trail
    saved_files = list(tmp_path.glob("cron/output/job2_*.txt"))
    assert len(saved_files) == 1
    assert saved_files[0].read_text() == long_content


@pytest.mark.asyncio
async def test_short_output_never_truncated(tmp_path, monkeypatch):
    """Output under the limit passes through untouched for any adapter."""
    monkeypatch.setattr("gateway.delivery.get_hermes_home", lambda: tmp_path)
    adapter = NonChunkingAdapter()
    router = DeliveryRouter(GatewayConfig(), adapters={Platform.DISCORD: adapter})
    target = DeliveryTarget.parse("discord:123")

    short_content = "x" * 100
    await router._deliver_to_platform(target, short_content, metadata={"job_id": "job3"})

    assert adapter.calls[0]["content"] == short_content
    # Nothing saved to disk
    assert not list(tmp_path.glob("cron/output/*.txt"))


@pytest.mark.asyncio
async def test_audit_save_failure_does_not_break_chunking_delivery(tmp_path, monkeypatch):
    """If the audit save fails (disk full, permissions), chunking adapters
    still receive the full content — the save is best-effort."""
    monkeypatch.setattr("gateway.delivery.get_hermes_home", lambda: tmp_path)

    adapter = ChunkingAdapter()
    router = DeliveryRouter(GatewayConfig(), adapters={Platform.DISCORD: adapter})
    target = DeliveryTarget.parse("discord:123")

    long_content = "x" * 5000

    call_count = {"n": 0}

    def failing_save(content, job_id):
        call_count["n"] += 1
        raise OSError("No space left on device")

    monkeypatch.setattr(router, "_save_full_output", failing_save)

    # Should NOT raise — audit failure is caught for chunking adapters
    await router._deliver_to_platform(target, long_content, metadata={"job_id": "job6"})

    # Adapter still got the full content
    assert adapter.calls[0]["content"] == long_content
    # Save was attempted (best-effort, swallowed)
    assert call_count["n"] == 1


@pytest.mark.asyncio
async def test_save_failure_during_truncation_raises_for_non_chunking_adapter(tmp_path, monkeypatch):
    """For a non-chunking adapter, the truncation footer needs a valid saved
    path. If the save fails there, that is a real delivery problem and the
    error propagates (not swallowed like the chunking best-effort save)."""
    monkeypatch.setattr("gateway.delivery.get_hermes_home", lambda: tmp_path)

    adapter = NonChunkingAdapter()
    router = DeliveryRouter(GatewayConfig(), adapters={Platform.DISCORD: adapter})
    target = DeliveryTarget.parse("discord:123")

    long_content = "x" * 5000

    def failing_save(content, job_id):
        raise OSError("No space left on device")

    monkeypatch.setattr(router, "_save_full_output", failing_save)

    # Non-chunking adapter must truncate → needs a valid saved path → the
    # Step 1 best-effort catch swallows the first attempt, but the Step 2
    # retry (footer needs the path) re-raises.
    with pytest.raises(OSError, match="No space left on device"):
        await router._deliver_to_platform(target, long_content, metadata={"job_id": "job7"})

class TestSlackThreePartTargetEndToEnd:
    """Regression for AO #684: 3-part ``slack:CHAN:thread_ts`` must reach
    chat.postMessage as ``thread_ts``.

    Background: the LLM-callable ``send_message`` tool used to drop the
    ``:thread_ts`` segment of a 3-part Slack target, landing threaded
    replies at the channel root as orphan posts. 10+ misroutes confirmed
    in the field since 2026-06-09:

    - 2026-06-13 20:57 UTC, ts 1781384270.728329 — Ag-f- architecture
      thread reply leaked to home channel ``C0AJQ5M0A0Y``.
    - 2026-06-13 18:08 UTC, ts 1781338930.938689 — PR #7480 dispatch ack
      ``target=slack:C0B9W8D609M:1781338930.938689`` fell back to
      ``C0AJQ5M0A0Y`` as a top-level orphan.
    - 2026-06-12 — Dropped-thread followup on ``C09GRLXF9GR``,
      thread root ``1781118730.141049`` — 10 narration posts leaked
      before recovery curl landed.
    - 2026-06-11, channel ``C0AH3RY3DK6`` — run interrupted mid-stream,
      queued followup reply sent at channel root (PR #27 / fix
      ``fix/slack-thread-ts-injected-reply-leak``).

    The gateway's own :class:`DeliveryTarget.parse` (gateway/delivery.py)
    already splits 3-part targets correctly with ``split(":", 2)`` and
    stores ``thread_id``. The defect was downstream: ``tools/send_message_tool._parse_target_ref``
    and ``_send_slack`` discarded the captured ``thread_id`` before
    posting to ``chat.postMessage``. This test exercises the
    DeliveryTarget → send_message_tool → chat.postMessage pipeline to
    catch the bug at the integration boundary where it actually broke
    in production.

    Acceptance: 3-part Slack target must produce a chat.postMessage
    payload carrying ``thread_ts`` matching the requested segment. The
    2-part form (no thread_ts) must remain deterministic and never
    silently grow a ``thread_ts`` key.
    """

    def _build_mock_session(self, ok_payload):
        """Return a mock aiohttp session whose ``post()`` yields ``ok_payload``."""
        from unittest.mock import AsyncMock, MagicMock

        mock_resp = MagicMock()
        mock_resp.status = 200
        mock_resp.json = AsyncMock(return_value=ok_payload)
        mock_resp.text = AsyncMock(return_value="")
        mock_resp.__aenter__ = AsyncMock(return_value=mock_resp)
        mock_resp.__aexit__ = AsyncMock(return_value=None)

        mock_session = MagicMock()
        mock_session.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session.__aexit__ = AsyncMock(return_value=None)
        mock_session.post = MagicMock(return_value=mock_resp)
        return mock_session, mock_resp

    def test_3part_slack_target_forwards_thread_ts_to_chat_postmessage(self):
        """``slack:C0AH3RY3DK6:1781465902.728229`` → ``thread_ts=1781465902.728229``.

        Reference: AO #684 misroute instance 1 (2026-06-13 20:57 UTC,
        target ``slack:C0BA4MCBPFB:<ts>`` should have threaded under the
        user's Ag-f- architecture thread, not landed at the channel root).
        """
        from unittest.mock import patch
        from tools.send_message_tool import send_message_tool

        # Slack response shape on success: ok=True, ts=posted_message_ts,
        # message={thread_ts: parent_ts}. We include the echoed thread_ts
        # so the fail-loud check in _send_slack passes.
        ok_payload = {
            "ok": True,
            "ts": "1781465902.728230",
            "message": {
                "ts": "1781465902.728230",
                "thread_ts": "1781465902.728229",
                "text": "echo of reply",
            },
        }
        mock_session, _ = self._build_mock_session(ok_payload)

        # Bypass gateway config lookup by providing a minimal GatewayConfig
        # with an enabled slack platform carrying a token.
        from gateway.config import GatewayConfig, PlatformConfig
        gw_config = GatewayConfig(
            platforms={
                Platform.SLACK: PlatformConfig(
                    enabled=True,
                    token="xoxb-test-token",
                ),
            },
        )

        # The same target the AO #684 cluster of misroutes used.
        target = "slack:C0AH3RY3DK6:1781465902.728229"
        # Pre-conditions: DeliveryTarget must capture the thread_ts.
        from gateway.delivery import DeliveryTarget
        dt = DeliveryTarget.parse(target)
        assert dt.platform == Platform.SLACK
        assert dt.chat_id == "C0AH3RY3DK6"
        assert dt.thread_id == "1781465902.728229"
        assert dt.is_explicit is True

        with patch("gateway.config.load_gateway_config", return_value=gw_config), \
             patch("aiohttp.ClientSession", return_value=mock_session):
            # Pass the 3-part target through the LLM-callable entry point.
            result = send_message_tool(
                {
                    "action": "send",
                    "platform": "slack",
                    "target": target,
                    "message": "reply in thread",
                }
            )

        # The capture must show that the chat.postMessage payload carried
        # the thread_ts key. This is the property the AO #684 bug lacked.
        kwargs = mock_session.post.call_args.kwargs
        payload = kwargs["json"]
        assert payload["channel"] == "C0AH3RY3DK6"
        assert payload["thread_ts"] == "1781465902.728229"
        # Endpoint is the canonical chat.postMessage URL.
        assert mock_session.post.call_args.args[0] == "https://slack.com/api/chat.postMessage"
        # send_message_tool returns a JSON string on success.
        import json
        result_obj = json.loads(result) if isinstance(result, str) else result
        assert result_obj.get("success") is True, (
            f"Expected success=True, got {result_obj!r}. The AO #684 "
            "fail-loud check should NOT have fired when the Slack API "
            "echoed back the thread_ts."
        )

    def test_3part_slack_target_fail_loud_when_slack_strips_thread_ts(self):
        """If Slack returns ``ok: True`` without echoing ``thread_ts``,
        the tool must return an error — not silently succeed.

        Reference: AO #684 misroute instance 2 (2026-06-13 18:08 UTC,
        ts 1781338930.938689 — PR #7480 dispatch ack 3-part form
        ``slack:C0B9W8D609M:1781338930.938689`` fell back to home channel
        as a top-level orphan. The previous silent-success shape returned
        ``success=True`` for an out-of-thread post — that is the exact
        behavior the gateway fix must prevent.)
        """
        from unittest.mock import patch
        from tools.send_message_tool import send_message_tool
        import json

        # Simulate the API's silent strip: ok=True but no message.thread_ts.
        # This is the misroute shape that produced the AO #684 cluster.
        ok_payload = {
            "ok": True,
            "ts": "1781338930.938700",
            "message": {
                "ts": "1781338930.938700",
                # NO thread_ts — Slack honored the request as channel-root.
                "text": "echo of reply",
            },
        }
        mock_session, _ = self._build_mock_session(ok_payload)

        from gateway.config import GatewayConfig, PlatformConfig
        gw_config = GatewayConfig(
            platforms={
                Platform.SLACK: PlatformConfig(
                    enabled=True,
                    token="***",
                ),
            },
        )

        target = "slack:C0B9W8D609M:1781338930.938689"

        from gateway.config import GatewayConfig, PlatformConfig
        gw_config = GatewayConfig(
            platforms={
                Platform.SLACK: PlatformConfig(
                    enabled=True,
                    token="xoxb-test-token",
                ),
            },
        )

        with patch("gateway.config.load_gateway_config", return_value=gw_config), \
             patch("aiohttp.ClientSession", return_value=mock_session):
            result = send_message_tool(
                {
                    "action": "send",
                    "platform": "slack",
                    "target": target,
                    "message": "dispatch ack",
                }
            )

        result_obj = json.loads(result) if isinstance(result, str) else result
        # Must NOT be silent success — this is the AO #684 fail-loud invariant.
        assert "error" in result_obj, (
            f"Expected fail-loud error when Slack stripped thread_ts, "
            f"got {result_obj!r}. The previous silent-success behavior is "
            f"exactly the bug that produced 10+ AO #684 misroutes."
        )
        # Error must name the target attempted AND the channel landed in
        # (per the user's spec: "naming the target attempted vs the
        # channel landed in").
        error_text = json.dumps(result_obj)
        assert "slack:C0B9W8D609M:1781338930.938689" in error_text, (
            f"Error must name the 3-part target attempted: {error_text!r}"
        )
        assert "C0B9W8D609M" in error_text, (
            f"Error must name the channel landed in: {error_text!r}"
        )

    def test_2part_slack_target_behaves_deterministically(self):
        """2-part ``slack:CHAN`` (no thread_ts) is deterministic:
        no ``thread_ts`` key in payload, no fail-loud.

        Per the user's spec: "2-part slack:CHAN behaves deterministically".
        """
        from unittest.mock import patch
        from tools.send_message_tool import send_message_tool
        import json

        # Channel-root post: ok=True with no thread_ts in response.
        ok_payload = {
            "ok": True,
            "ts": "1781338930.999999",
            "message": {
                "ts": "1781338930.999999",
                "text": "channel-root",
                # No thread_ts — this is a deliberate channel-root post.
            },
        }
        mock_session, _ = self._build_mock_session(ok_payload)

        from gateway.config import GatewayConfig, PlatformConfig
        gw_config = GatewayConfig(
            platforms={
                Platform.SLACK: PlatformConfig(
                    enabled=True,
                    token="***",
                ),
            },
        )

        target = "slack:C0B0QV5434G"

        from gateway.config import GatewayConfig, PlatformConfig
        gw_config = GatewayConfig(
            platforms={
                Platform.SLACK: PlatformConfig(
                    enabled=True,
                    token="xoxb-test-token",
                ),
            },
        )

        with patch("gateway.config.load_gateway_config", return_value=gw_config), \
             patch("aiohttp.ClientSession", return_value=mock_session):
            result = send_message_tool(
                {
                    "action": "send",
                    "platform": "slack",
                    "target": target,
                    "message": "top-level channel post",
                }
            )

        kwargs = mock_session.post.call_args.kwargs
        payload = kwargs["json"]
        assert payload["channel"] == "C0B0QV5434G"
        # 2-part form: NO thread_ts key — that is the deterministic shape.
        assert "thread_ts" not in payload, (
            f"2-part Slack target must NOT inject a thread_ts; got payload={payload!r}"
        )
        result_obj = json.loads(result) if isinstance(result, str) else result
        # 2-part form is a deliberate channel-root post — success is correct.
        assert result_obj.get("success") is True
        assert "error" not in result_obj

    def test_2part_slack_does_not_parse_third_segment(self):
        """``slack:CHAN:NOT_A_TS`` falls through to channel-name resolution,
        not silently treated as a 3-part target with a bad thread_ts.

        Defensive: if the regex ever loosens too far, this test catches
        over-broad matching.
        """
        from gateway.delivery import DeliveryTarget

        target = "slack:C0AH3RY3DK6:not-a-thread-ts"
        dt = DeliveryTarget.parse(target)
        # DeliveryTarget splits on ':' regardless of segment shape — the
        # gating against malformed thread_ts is enforced by
        # tools/send_message_tool._SLACK_TARGET_RE, not by DeliveryTarget.
        assert dt.platform == Platform.SLACK
        assert dt.chat_id == "C0AH3RY3DK6"
        assert dt.thread_id == "not-a-thread-ts"

        # The send_message_tool layer must reject "not-a-thread-ts" as a
        # non-thread target (it doesn't match the channel ID pattern and
        # falls through to channel-name resolution).
        from tools.send_message_tool import _parse_target_ref
        chat_id, thread_id, is_explicit = _parse_target_ref("slack", "C0AH3RY3DK6:not-a-thread-ts")
        # The widened regex requires thread_ts to match \d+\.\d+ — "not-a-thread-ts"
        # does not, so the whole target is rejected (returns None, None, False).
        assert (chat_id, thread_id, is_explicit) == (None, None, False), (
            f"Malformed thread_ts must reject the whole target, got "
            f"({chat_id!r}, {thread_id!r}, {is_explicit!r})"
        )



