import pytest
from pydantic import ValidationError
from dachi.core import (
    Msg, Resp, 
    ListDialog, 
    TreeDialog
)

# Helpers
def _sample_msg(**overrides):
    """Return a minimal valid `Msg`, applying any keyword overrides."""
    base = {"role": "user", "text": "hello"}
    base.update(overrides)
    return Msg(**base)


# Tests for Msg

class TestMsg:
    # constructor 
    # @pytest.mark.parametrize(
    #     "field,value",
    #     [
    #         ("role", None),  # non‑str role
    #         ("role", 123),
    #         ("alias", 123),  # alias must be str | None
    #         ("meta", "not‑a‑dict"),  # meta must be mapping
    #     ],
    # )
    # def test_ctor_type_enforcement(self, field, value):
    #     kwargs = {"role": "user", "content": "hi", field: value}
    #     with pytest.raises(ValidationError):
    #         Msg(**kwargs)

    def test_ctor_minimal(self):
        msg = _sample_msg()
        as_dict = msg.model_dump()
        assert as_dict["role"] == "user"
        assert as_dict["text"] == "hello"
        # optional fields defaulted
        assert as_dict.get("alias") is None

    def test_ctor_allows_extra_fields(self):
        msg = _sample_msg(foo=1)
        assert msg.foo == 1  # type: ignore[attr-defined]
        # round‑trip through dump & reload keeps the extra field
        reloaded = Msg(**msg.model_dump())
        assert reloaded.foo == 1  # type: ignore[attr-defined]

    def test_large_content_and_dict_content(self):
        big = "x" * 1_048_576  # 1 MiB
        Msg(role="system", text=big)  # should not raise
        nested = {"a": 1, "b": {"c": 2}}
        Msg(role="assistant", text=nested)  # should not raise

#     # --------------------------- behaviour ---------------------------

    # TODO: Apply was removed so remove
    # def test_apply_raises_when_function_raises(self):
    #     msg = _sample_msg()
    #     with pytest.raises(RuntimeError):
    #         msg.apply(lambda m: (_ for _ in ()).throw(RuntimeError("boom")))

#     # --------------------------- output ---------------------------

    # TODO: Remove
    # def test_output_present_key(self):
    #     msg = _sample_msg(meta={"tool_out": 9})
    #     assert msg.output() == 9

    # def test_output_missing_key_returns_default(self):
    #     msg = _sample_msg()
    #     sentinel = object()
    #     assert msg.output(default=sentinel) is sentinel
    #     assert "tool_out" not in msg.meta  # default *not* inserted

    def test_render_with_alias(self):
        msg = _sample_msg(alias="USR")
        assert msg.render() == "USR: hello"

    def test_render_without_alias(self):
        msg = _sample_msg()
        assert msg.render() == "user: hello"

    def test_render_with_unicode_alias(self):
        msg = _sample_msg(alias="🦊")
        assert msg.render() == "🦊: hello"

    def test_to_input_normal(self):
        # Note: to_input() method removed - conversion now handled by AIAdapt
        msg = _sample_msg()
        # Test basic message structure instead
        assert msg.role == "user"
        assert msg.text == "hello"

    # def test_to_input_filtered_returns_empty(self):
    #     # Note: to_input() method removed - conversion now handled by AIAdapt
    #     msg = _sample_msg(filtered=True)
    #     # Test filtered flag is set correctly
    #     assert msg.filtered == True

    def test_equality_semantics(self):
        # Use fixed timestamp to ensure equality
        from datetime import datetime, timezone
        fixed_time = datetime(2023, 1, 1, 12, 0, 0, tzinfo=timezone.utc)
        a = _sample_msg(timestamp=fixed_time)
        b = _sample_msg(timestamp=fixed_time)
        assert a == b and a is not b


class TestResp:

    def _new_resp(self):
        return Resp(msg=_sample_msg())

    def test_ctor_initialises_empty_data_dict(self):
        resp = self._new_resp()
        # Note: implementation bug uses list, but we assert interface behaves like dict
        assert isinstance(resp._raw, dict)  # type: ignore[attr-defined]
        assert resp._raw == {}

    def test_set_and_get_data(self):
        resp = self._new_resp()
        out = resp.raw["k"] = 1
        # set_data is chainable
        assert out is resp.raw['k']
        resp.raw["k"] = 2
        assert resp.raw["k"] == 2

    def test_data_unknown_key_raises(self):
        resp = self._new_resp()
        with pytest.raises(KeyError):
            resp.raw["missing"]

    def test_set_data_allows_non_hashable_values(self):
        lst = [1, 2]
        resp = self._new_resp()
        resp.raw["l"] = lst
        assert resp.raw["l"] is lst

#     # --------------------------- get_tmp ---------------------------

    def test_get_tmp_without_tmp_attr_raises(self):
        resp = self._new_resp()
        with pytest.raises(KeyError):
            resp.raw['a']

#     # --------------------------- isolation ---------------------------

    def test_multiple_resp_instances_isolated_data(self):
        msg = _sample_msg()
        r1 = Resp(msg=msg)
        r2 = Resp(msg=msg)
        r1.raw["x"] = 1
        assert "x" not in r2._raw

#     # --------------------------- repr ---------------------------

    def test_repr_contains_role_for_debug(self):
        resp = self._new_resp()
        representation = repr(resp)
        # crude sanity check – ensure role string appears
        assert resp.msg.role in representation


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _msg(role="user", text="hi", **kw):
    """Convenience for quickly making valid Msg objects."""
    return Msg(role=role, text=text, **kw)


# ---------------------------------------------------------------------------
# Test class for ListDialog --------------------------------------------------
# ---------------------------------------------------------------------------

class TestListDialog:
    """Black‑box invariants for the concrete list‑backed dialog container."""

    # ----- Construction -----------------------------------------------------

    def test_create_empty(self):
        dlg = ListDialog()
        assert len(dlg) == 0
        assert list(dlg) == []

    def test_create_with_messages(self):
        m1, m2 = _msg("system", "a"), _msg("user", "b")
        dlg = ListDialog(messages=[m1, m2])
        assert len(dlg) == 2
        assert list(dlg) == [m1, m2]
        assert dlg[0] is m1 and dlg[1] is m2

    def test_validation_on_bad_messages(self):
        with pytest.raises(ValidationError):
            ListDialog(messages=["not a msg"])  # type: ignore[arg-type]

    # ----- __getitem__ / __setitem__ ---------------------------------------

    @pytest.mark.parametrize("idx", [0, -1])
    def test_getitem_valid_indices(self, idx):
        m = _msg()
        dlg = ListDialog(messages=[m])
        assert dlg[idx] is m

    @pytest.mark.parametrize("idx", [1, -2])
    def test_getitem_out_of_range(self, idx):
        dlg = ListDialog(messages=[_msg()])
        with pytest.raises(IndexError):
            _ = dlg[idx]

    def test_getitem_slice_unsupported(self):
        dlg = ListDialog(messages=[_msg(), _msg()])
        msg1, = dlg[0:1]
        assert msg1 is dlg[0]

    def test_setitem_replacement(self):
        m1, m2 = _msg("user", "old"), _msg("user", "new")
        dlg = ListDialog(messages=[m1])
        dlg[0] = m2
        assert dlg[0] is m2 and len(dlg) == 1

    def test_setitem_append_when_index_equals_len(self):
        dlg = ListDialog(messages=[_msg("a"), _msg("b")])
        m3 = _msg("assistant", "c")
        dlg[len(dlg)] = m3  # acts like append
        assert dlg[-1] is m3 and len(dlg) == 3

    # # ----- append / insert --------------------------------------------------

    def test_append_positive(self):
        dlg = ListDialog()
        m = _msg()
        returned = dlg.append(m)
        assert returned is dlg and dlg[-1] is m and len(dlg) == 1

    def test_append_invalid(self):
        dlg = ListDialog()
        with pytest.raises(ValueError):
            dlg.append(None)  # type: ignore[arg-type]

    @pytest.mark.parametrize(
        "index", [0, 1, -1, -2]
    )
    def test_insert_various_positions(self, index):
        # base list has two msgs
        m0, m1, m_new = _msg("a"), _msg("b"), _msg("new")
        dlg = ListDialog(messages=[m0, m1])
        dlg.insert(index, m_new)
        assert len(dlg) == 3
        assert m_new in dlg

    # def test_insert_out_of_bounds(self):
    #     dlg = ListDialog(messages=[_msg()])
    #     with pytest.raises(IndexError):
    #         dlg.insert(999, _msg())

    # # ----- pop / remove -----------------------------------------------------

    def test_pop_returns_msg_and_shrinks(self):
        m0, m1 = _msg("x"), _msg("y")
        dlg = ListDialog(messages=[m0, m1])
        popped = dlg.pop()  # default = last
        assert popped is m1 and len(dlg) == 1 and dlg[0] is m0

    def test_pop_empty_raises(self):
        dlg = ListDialog()
        with pytest.raises(IndexError):
            dlg.pop()

    def test_remove_first_occurrence_only(self):
        msg = _msg("dup")
        dlg = ListDialog(messages=[msg, msg, _msg("other")])
        dlg.remove(msg)
        assert len(dlg) == 2 and dlg[0] is not dlg[1]

    # # ----- extend -----------------------------------------------------------

    def test_extend_with_other_dialog(self):
        base = ListDialog(messages=[_msg("a")])
        tail = ListDialog(messages=[_msg("b"), _msg("c")])
        base.extend(tail)
        assert len(base) == 3 and list(base)[1:] == list(tail)

    def test_extend_with_non_msg_iterable_raises(self):
        dlg = ListDialog()
        with pytest.raises(ValueError):
            dlg.extend([_msg(), "bad"])  # type: ignore[list-item]

    # # ----- clone ------------------------------------------------------------

    def test_clone_is_shallow(self):
        m1, m2 = _msg("x"), _msg("y")
        dlg = ListDialog(messages=[m1, m2])
        clone = dlg.clone()
        assert clone is not dlg and list(clone) == list(dlg)
        # shallow: mutate message object shared
        m1.text = "changed"
        assert clone[0].text == "changed"
        # but structural independence
        dlg.append(_msg("z"))
        assert len(clone) == 2 < len(dlg)

    # # ----- to_input / render helpers ----------------------------------------

    # def test_to_input_filters(self):
    #     # Note: to_input() method removed - conversion now handled by AIAdapt
    #     visible = _msg("user", "show")
    #     hidden = _msg("assistant", "hide", filtered=True)
    #     dlg = ListDialog(messages=[visible, hidden])
    #     # Test filtering behavior in dialog
    #     assert len(dlg) == 2
    #     assert not visible.filtered
    #     assert hidden.filtered

    def test_render_default(self):
        m1, m2 = _msg("system", "hello"), _msg("user", "world")
        dlg = ListDialog(messages=[m1, m2])
        rendered = dlg.render()
        assert rendered.split("\n") == [m1.render(), m2.render()]

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


    # @pytest.mark.parametrize("depth", [0, 1])
    # def test_insert_creates_branch_and_updates_indices(self, depth):
    #     """Insert at various depths should create sibling and move leaf to new node."""
    #     td = TreeDialog()
    #     td.append(_msg("sys", "root"))
    #     td.append(_msg("assistant", "child"))  # depth 1 path now

    #     m_new = _msg("user", "inserted")
    #     td.insert(depth, m_new)

    #     assert td.leaf.message is m_new
    #     assert 0 <= td.indices[depth] < td.counts[depth]
    #     # Path coherence invariant
    #     for i, idx in enumerate(td.indices):
    #         assert 0 <= idx < td.counts[i]

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

    # # ------------------------------------------------------------------
    # # Path coherence invariant after mix‑ops
    # # ------------------------------------------------------------------

    # def test_path_coherence_after_complex_ops(self):
    #     td = TreeDialog()
    #     td.append(_msg("sys", "r"))
    #     td.append(_msg("assistant", "a"))  # depth 1
    #     td.insert(1, _msg("user", "branch"))  # sibling at depth 1
    #     td.append(_msg("assistant", "new leaf"))
    #     td.replace(2, _msg("assistant", "replaced"))
    #     td.remove(1)  # remove sibling 1 at depth 1 -> shrink counts

    #     # invariant
    #     assert len(td) == len(td.indices) == len(td.counts)
    #     for i, idx in enumerate(td.indices):
    #         assert 0 <= idx < td.counts[i]



# class TestDialogTurn:

#     def test_empty_node_basics(self):
#         m = _msg()
#         node = DialogTurn(message=m)

#         assert node.message is m
#         assert node.children == []
#         assert node.parent is None  # private attr but exposed via property
#         assert node.depth() == 1
#         assert node.root() is node
#         assert node.leaf() is node
#         assert node.n_children() == 0
#         assert node.n_siblings() == 1  # root counts itself only

#     def test_constructor_sets_parent_on_children(self):
#         child1 = DialogTurn(message=_msg("assistant", "a1"))
#         child2 = DialogTurn(message=_msg("assistant", "a2"))
#         root = DialogTurn(message=_msg("system", "root"), children=[child1, child2])

#         assert child1.parent is root
#         assert child2.parent is root
#         # depth chain correctness
#         assert child1.depth() == 2
#         assert child1.root() is root
#         assert root.n_children() == 2

#     def test_append_child(self):
#         root = DialogTurn(message=_msg())
#         child_msg = _msg("assistant", "child")
#         child_turn = root.append(child_msg)

#         assert child_turn in root.children
#         assert child_turn.parent is root
#         assert root.n_children() == 1
#         assert child_turn.depth() == 2
#         # append returns the new DialogTurn
#         assert isinstance(child_turn, DialogTurn)

#     def test_prepend_child(self):
#         root = DialogTurn(message=_msg())
#         first = root.append(_msg("assistant", "1"))
#         prepended = root.prepend(_msg("assistant", "0"))

#         # new child should be at index 0
#         assert root.parent is prepended
#         assert prepended.children[0] is root
#         assert root.children[0] is first

#     @pytest.mark.parametrize("method", ["append", "prepend"])
#     def test_append_prepend_invalid_type(self, method):
#         root = DialogTurn(message=_msg())
#         with pytest.raises(ValidationError):
#             getattr(root, method)(42)  # type: ignore[arg-type]

#     def test_prune_detaches_child_and_updates_counts(self):
#         root = DialogTurn(message=_msg())
#         c1 = root.append(_msg("assistant", "1"))
#         c2 = root.append(_msg("assistant", "2"))

#         detached = root.prune(0)
#         assert detached is c1
#         assert detached.parent is None
#         assert root.n_children() == 1
#         assert root.children[0] is c2

#     @pytest.mark.parametrize("bad_idx", [-1, 1])
#     def test_prune_invalid_index(self, bad_idx):
#         root = DialogTurn(message=_msg())
#         root.append(_msg())
#         with pytest.raises(IndexError):
#             root.prune(bad_idx)

#     def _build_chain(self, depth: int = 3):  # helper to build left chain of given depth
#         root = DialogTurn(message=_msg("system", "root"))
#         cur = root
#         for i in range(depth):
#             cur = cur.append(_msg("assistant", f"lvl{i}"))
#         return root, cur

#     def test_child_navigation(self):
#         root = DialogTurn(message=_msg())
#         c1 = root.append(_msg("assistant", "c1"))
#         c2 = root.append(_msg("assistant", "c2"))

#         assert root.child(0) is c1
#         assert root.child(1) is c2
#         with pytest.raises(IndexError):
#             root.child(2)
#         with pytest.raises(TypeError):
#             root.child(1.1)  # type: ignore[arg-type]

#     def test_sibling_navigation(self):
#         root = DialogTurn(message=_msg())
#         a = root.append(_msg("assistant", "a"))
#         b = root.append(_msg("assistant", "b"))
#         c = root.append(_msg("assistant", "c"))

#         assert b.sibling(-1) is a
#         assert b.sibling(1) is c
#         assert b.sibling(0) is b
#         with pytest.raises(IndexError):
#             a.sibling(-1)
#         with pytest.raises(IndexError):
#             c.sibling(1)
#         with pytest.raises(TypeError):
#             b.sibling("x")  # type: ignore[arg-type]

#     def test_ancestor_navigation(self):
#         root, leaf = self._build_chain(4)
#         assert leaf.ancestor(4) is root
#         assert leaf.ancestor(0) is leaf
#         with pytest.raises(IndexError):
#             leaf.ancestor(5)
#         with pytest.raises(IndexError):
#             leaf.ancestor(-1)  # type: ignore[arg-type]
#         with pytest.raises(TypeError):
#             leaf.ancestor(1.2)  # type: ignore[arg-type]

#     def test_depth_root_leaf_properties(self):
#         root, leaf = self._build_chain(3)
#         assert root.depth() == 1
#         assert leaf.depth() == 4
#         assert root.root() is root
#         assert leaf.root() is root
#         assert root.leaf() is leaf
#         # Add another branch to root and ensure leaf still leftmost
#         other = root.append(_msg("assistant", "other"))
#         assert root.leaf() is leaf
#         # leaf of 'other' subtree should be itself (no children)
#         assert other.leaf() is other

#     def test_reparent_pruned_subtree(self):
#         root = DialogTurn(message=_msg())
#         child = root.append(_msg("assistant", "c"))
#         subchild = child.append(_msg("assistant", "sc"))

#         # prune child from root
#         detached = root.prune(0)
#         assert detached is child
#         assert detached.parent is None
#         # attach to subchild (creates deeper nesting)
#         new_parent = subchild.append(_msg("assistant", "x"))
#         assert new_parent.parent is subchild

#     def test_duplicate_child_append(self):
#         root = DialogTurn(message=_msg())
#         child = root.append(_msg("assistant", "dup"))
#         # trying to append same DialogTurn instance again should raise or reparent safely
#         with pytest.raises((ValueError, ValidationError, RuntimeError)):
#             root.append(child)  # type: ignore[arg-type]

#     def test_counts_update_after_mutations(self):
#         root = DialogTurn(message=_msg())
#         assert root.n_children() == 0
#         c1 = root.append(_msg())
#         assert root.n_children() == 1
#         c2 = root.append(_msg())
#         assert root.n_children() == 2
#         root.prune(1)
#         assert root.n_children() == 1
#         root.prune(0)
#         assert root.n_children() == 0

#     # # ------------------------------------------------------------------
#     # # Cycle protection (smoke)
#     # # ------------------------------------------------------------------
#     # def test_cycle_detection_on_root(self):
#     #     node = DialogTurn(message=_msg())
#     #     # forge an explicit cycle – not a valid use but we want robustness
#     #     node._parent = node  # type: ignore[attr-defined]
#     #     with pytest.raises((RecursionError, RuntimeError)):
#     #         _ = node.root()
