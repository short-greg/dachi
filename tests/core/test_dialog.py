import pytest
from pydantic import ValidationError
from dachi.core import (
    Msg, Resp, 
    ListDialog, 
    TreeDialog
)
import typing as t

class Msg(t.TypedDict):
    text: str
    role: str
    metadata: t.Optional[dict] = None


# Helpers
def _sample_msg(**overrides):
    """Return a minimal valid `Msg`, applying any keyword overrides."""
    base = {"role": "user", "text": "hello"}
    base.update(overrides)
    return Msg(**base)


# Tests for Msg

def _msg(role: str = "user", text: str = "hi", **kw) -> Msg:
    """Convenience factory for a valid Msg object."""
    return Msg(role=role, text=text, **kw)


class TestTreeDialog:
    # ------------------------------------------------------------------
    # Construction & base properties
    # ------------------------------------------------------------------

    def test_empty_dialog_basics(self):
        td = TreeDialog()
        assert len(td) == 0
        assert td.indices == []
        assert td.counts == []
        assert td.root is None

    def test_single_append_sets_root_leaf(self):
        td = TreeDialog()
        m0 = _msg("system", "root")
        td.append(m0)

        assert len(td) == 1
        assert td.indices == [0]
        assert td.counts == [1]
        assert td.root is m0
        assert td.leaf is m0

    # # ------------------------------------------------------------------
    # # Append chain
    # # ------------------------------------------------------------------

    def test_multiple_appends_creates_linear_path(self):
        td = TreeDialog()
        ms = [_msg(text=f"m{i}") for i in range(3)]
        for m in ms:
            td.append(m)

        assert len(td) == 3
        assert td.indices == [0, 0, 0]
        assert td.counts == [1, 1, 1]
        # Ensure leaf contains last appended msg
        assert td.leaf is ms[-1]

    # # ------------------------------------------------------------------
    # # Insert branching positive & edge
    # # ------------------------------------------------------------------

    def test_insert_replaces_leaf_with_new_msg(self):
        td = TreeDialog()
        td.append(_msg("sys", "root"))
        # td.append(_msg("assistant", "child"))  # path depth = 2

        new_msg = _msg("user", "inserted")
        td.insert(0, new_msg)  # Insert at depth 1

        assert td.root is new_msg

    def test_insert_replaces_leaf_with_new_msg_with_append(self):
        td = TreeDialog()
        td.append(_msg("sys", "root"))
        td.append(_msg("assistant", "child"))  # path depth = 2
        
        new_msg = _msg("user", "inserted")
        td.insert(1, new_msg)  # Insert at depth 1

        # After inserting at position 1, the inserted message should be at position 1 in the path
        assert td[1] is new_msg



    def test_insert_depth_equals_len_raises(self):
        td = TreeDialog()
        td.append(_msg())
        with pytest.raises(IndexError):
            td.insert(2, _msg())  # Should raise when trying to insert beyond length

    def test_insert_with_non_msg_raises(self):
        td = TreeDialog()
        td.append(_msg())
        with pytest.raises(ValueError):
            td.insert(0, "not a message")

    # # ------------------------------------------------------------------
    # # Replace
    # # ------------------------------------------------------------------

    def test_replace_swaps_message_keeps_path(self):
        td = TreeDialog()
        td.append(_msg("sys", "root"))
        m_old = _msg("assistant", "old")
        td.append(m_old)

        m_new = _msg("assistant", "new")
        td.replace(1, m_new)

        assert td.leaf is m_new
        assert td.indices == [0, 0]
        assert td.counts == [1, 1]

    def test_replace_with_invalid_depth(self):
        td = TreeDialog()
        with pytest.raises(ValueError):
            td.replace(0, _msg())

    # # ------------------------------------------------------------------
    # # Remove
    # # ------------------------------------------------------------------

    def test_remove_leaf_reduces_depth(self):
        td = TreeDialog()
        msg1 = _msg("sys")
        msg2 = _msg("assistant")
        td.append(msg1)
        td.append(msg2)
        td.remove(msg2)

        assert len(td) == 1
        # assert td.depth() == 1
        assert td.indices == [0]
        assert td.counts == [1]

    def test_remove_root_empties_dialog(self):
        td = TreeDialog()
        msg = _msg()
        td.append(msg)
        with pytest.raises(ValueError):
            td.remove(msg)

    # # ------------------------------------------------------------------
    # # Navigation helpers
    # # ------------------------------------------------------------------

    def _build_three_level_tree(self):
        """Utility: returns TreeDialog with root->child->grandchild."""
        td = TreeDialog()
        td.append(_msg("sys", "root"))
        td.append(_msg("assistant", "child"))
        td.append(_msg("user", "grandchild"))
        return td

    def test_child_navigation(self):
        td = self._build_three_level_tree()
        leaf_msg = td.leaf
        # go up then back down
        td.rise(1)
        td.leaf_child(0)
        assert td.leaf is leaf_msg  # should now match leaf message

    def test_sibling_navigation(self):
        td = TreeDialog()
        td.append(_msg("sys", "root"))
        td.append(_msg("assistant", "c0"))
        td.append(_msg("assistant", "c1"))  # create sibling via insert
        sibling_msg = td.leaf
        td.rise(1)
        td.append(_msg("assistant", "c2"))  # create sibling via insert

        # Now leaf is sibling1; move back to original using sibling(-1)
        td.leaf_sibling(-1)
        assert td.leaf is sibling_msg

    # # ------------------------------------------------------------------
    # # Clone behaviour
    # # ------------------------------------------------------------------

    def test_clone_deep_structure_shallow_msgs(self):
        td = self._build_three_level_tree()
        clone = td.clone()

        assert clone is not td
        assert clone.indices == td.indices
        assert clone.counts == td.counts
        # Shallow: same Msg object identities
        for m_original, m_clone in zip(td, clone):
            assert m_original is m_clone

        # Mutating original structure doesn’t touch clone
        td.append(_msg("assistant", "extra"))
        assert len(td) == len(clone) + 1

    # # ------------------------------------------------------------------
    # # Iteration & render basics
    # # ------------------------------------------------------------------

    def test_iteration_order_root_to_leaf(self):
        td = self._build_three_level_tree()
        contents = [m.text for m in td]
        assert contents == ["root", "child", "grandchild"]
