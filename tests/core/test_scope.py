from typing import Tuple
import pytest

from dachi.core._scope import Scope, Ctx


class TestScope:
    
    def test_ctx_creates_child_at_index_when_called(self):
        scope = Scope()
        child_ctx = scope.ctx(0)
        assert isinstance(child_ctx, Ctx)
        assert child_ctx.index_path == (0,)
        assert child_ctx.scope is scope
    
    def test_ctx_registers_alias_when_tag_provided(self):
        scope = Scope()
        scope.ctx(0, tag="plan")
        assert scope.aliases["plan"] == (0,)
    
    def test_ctx_retags_existing_child_when_called_again(self):
        scope = Scope()
        scope.ctx(0, tag="old_tag")
        scope.ctx(0, tag="new_tag")
        assert scope.aliases["new_tag"] == (0,)
        assert "old_tag" not in scope.aliases
    
    def test_getitem_accesses_direct_tuple_key(self):
        scope = Scope()
        scope[(0, "goal")] = "test_value"
        assert scope[(0, "goal")] == "test_value"
    
    def test_getitem_resolves_string_path_to_tuple_key(self):
        scope = Scope()
        scope[(0, "goal")] = "test_value"
        assert scope["0.goal"] == "test_value"
    
    def test_getitem_resolves_tag_path_via_aliases(self):
        scope = Scope()
        scope.aliases["plan"] = (0,)
        scope[(0, "goal")] = "test_value" 
        assert scope["plan.goal"] == "test_value"
    
    def test_getitem_handles_nested_index_paths(self):
        scope = Scope()
        scope[(0, 1, "goal")] = "nested_value"
        assert scope["0.1.goal"] == "nested_value"


class TestScopeStringPaths:
    
    def test_resolve_path_parses_index_path_correctly(self):
        scope = Scope()
        scope[(0, "goal")] = "value"
        assert scope._resolve_path("0.goal") == (0, "goal")
    
    def test_resolve_path_handles_nested_subfields(self):
        scope = Scope()
        data = {"pose": {"x": 1, "y": 2}}
        scope[(0, "sensor")] = data
        result = scope["0.sensor"]
        assert result["pose"]["x"] == 1
    
    def test_resolve_path_converts_numeric_strings_to_int(self):
        scope = Scope()
        scope[(0, 1, "goal")] = "value"
        path = scope._resolve_path("0.1.goal")
        assert path == (0, 1, "goal")
    
    def test_resolve_path_resolves_tag_to_index_path(self):
        scope = Scope()
        scope.aliases["plan"] = (0, 1)
        path = scope._resolve_path("plan.goal")
        assert path == (0, 1, "goal")
    
    def test_resolve_path_raises_keyerror_when_tag_missing(self):
        scope = Scope()
        with pytest.raises(KeyError, match="Unknown tag: missing_tag"):
            scope._resolve_path("missing_tag.goal")


class TestCtx:
    
    def test_setitem_stores_data_in_scope_at_index_path(self):
        scope = Scope()
        ctx = scope.ctx(0)
        ctx["goal"] = "test_value"
        assert scope[(0, "goal")] == "test_value"
    
    def test_getitem_retrieves_data_from_scope_at_index_path(self):
        scope = Scope()
        scope[(0, "goal")] = "test_value"
        ctx = scope.ctx(0)
        assert ctx["goal"] == "test_value"
    
    def test_child_creates_nested_index_path(self):
        scope = Scope()
        parent_ctx = scope.ctx(0)
        child_ctx = parent_ctx.child(1)
        assert child_ctx.index_path == (0, 1)
        assert child_ctx.scope is scope
    
    def test_child_registers_tag_alias_when_provided(self):
        scope = Scope()
        parent_ctx = scope.ctx(0)
        child_ctx = parent_ctx.child(1, tag="move")
        assert scope.aliases["move"] == (0, 1)
        assert child_ctx.index_path == (0, 1)
    
    def test_child_creates_multi_level_nesting(self):
        scope = Scope()
        ctx_0 = scope.ctx(0)
        ctx_0_1 = ctx_0.child(1)
        ctx_0_1_2 = ctx_0_1.child(2, tag="deep")
        assert ctx_0_1_2.index_path == (0, 1, 2)
        assert scope.aliases["deep"] == (0, 1, 2)


class TestCtxDataAccess:
    
    def test_multiple_access_patterns_return_same_data(self):
        scope = Scope()
        ctx = scope.ctx(0, tag="plan")
        ctx["goal"] = "target_value"
        
        # All these should return the same data
        assert scope[(0, "goal")] == "target_value"
        assert scope["0.goal"] == "target_value"
        assert scope["plan.goal"] == "target_value"
    
    def test_tag_and_index_access_equivalent(self):
        scope = Scope()
        plan_ctx = scope.ctx(0, tag="plan")
        move_ctx = plan_ctx.child(1, tag="move")
        
        move_ctx["status"] = "running"
        
        # Index-based access
        assert scope["0.1.status"] == "running"
        # Tag-based access  
        assert scope["move.status"] == "running"
        # Mixed access
        assert scope["0.move.status"] == "running"
    
    def test_nested_data_storage_and_retrieval(self):
        scope = Scope()
        ctx = scope.ctx(0, tag="sensor")
        
        # Store nested data
        pose_data = {"x": 10, "y": 20, "theta": 1.5}
        ctx["pose"] = pose_data
        
        # Retrieve via different paths
        assert scope["0.pose"] == pose_data
        assert scope["sensor.pose"] == pose_data
        assert scope["0.pose"]["x"] == 10
    
    def test_complex_behavior_tree_scenario(self):
        scope = Scope()
        
        # Create a behavior tree structure: Sequence -> [Plan, Move]
        seq_ctx = scope.ctx(0, tag="sequence")
        plan_ctx = seq_ctx.child(0, tag="plan")
        move_ctx = seq_ctx.child(1, tag="move")
        
        # Plan outputs a goal
        plan_ctx["goal"] = (10, 20, 0)
        
        # Move should be able to access plan's goal
        assert scope["plan.goal"] == (10, 20, 0)
        assert scope["0.0.goal"] == (10, 20, 0)
        
        # Invalid pattern: alias.index.variable should raise error
        with pytest.raises(KeyError):
            scope["sequence.0.goal"]
        
        # Move outputs arrival status
        move_ctx["arrived"] = True
        
        # Multiple access patterns for move output
        assert scope["move.arrived"] == True
        assert scope["0.1.arrived"] == True


class TestScopeEdgeCases:
    
    def test_empty_scope_returns_keyerror_for_missing_keys(self):
        scope = Scope()
        with pytest.raises(KeyError):
            scope["nonexistent.key"]
    
    def test_accessing_nonexistent_tag_raises_keyerror(self):
        scope = Scope()
        with pytest.raises(KeyError, match="Unknown tag"):
            scope["unknown_tag.field"]
    
    def test_numeric_only_path_raises_error_when_no_variable(self):
        scope = Scope()
        scope[(0, 1)] = "numeric_path_value"
        # Invalid pattern: indices without variable name should raise error
        with pytest.raises(KeyError):
            scope["0.1"]
    
    def test_invalid_alias_index_variable_pattern_raises_error(self):
        scope = Scope()
        scope.aliases["plan"] = (0,)
        scope[(0, 1, "status")] = "mixed_value"
        # Invalid pattern: alias.index.variable should raise error
        with pytest.raises(KeyError):
            scope["plan.1.status"]
    
    def test_ctx_setitem_with_nested_key_creates_proper_path(self):
        scope = Scope()
        ctx = scope.ctx(0, 1)  # Multi-part index path
        ctx["result"] = "nested_result"
        assert scope[(0, 1, "result")] == "nested_result"
        assert scope["0.1.result"] == "nested_result"