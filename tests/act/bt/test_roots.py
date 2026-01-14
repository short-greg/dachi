from __future__ import annotations

from dachi.act.bt import BT


class TestBTBaseCtx:
    """Test set_base_ctx and get_base_ctx on BT"""

    def test_set_base_ctx_stores_value_retrievable_by_get_base_ctx(self):
        """Test set_base_ctx stores value retrievable by get_base_ctx"""
        bt = BT(root=None)

        bt.set_base_ctx("key", "value")

        assert bt.get_base_ctx("key") == "value"

    def test_get_base_ctx_returns_default_when_key_not_found(self):
        """Test get_base_ctx returns default when key not found"""
        bt = BT(root=None)

        assert bt.get_base_ctx("missing", default="default") == "default"
