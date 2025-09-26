from typing import Tuple
import pytest

from dachi.core._scope import Scope, Ctx, BoundCtx, BoundScope


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
        assert scope["/0.goal"] == "test_value"
    
    def test_getitem_resolves_tag_path_via_aliases(self):
        scope = Scope()
        scope.aliases["plan"] = (0,)
        scope[(0, "goal")] = "test_value" 
        assert scope["plan.goal"] == "test_value"
    
    def test_getitem_handles_nested_index_paths(self):
        scope = Scope()
        scope[(0, 1, "goal")] = "nested_value"
        assert scope["/0.1.goal"] == "nested_value"


class TestScopeStringPaths:
    
    def test_resolve_path_parses_index_path_correctly(self):
        scope = Scope()
        scope[(0, "goal")] = "value"
        assert scope._resolve_path("/0.goal") == (0, "goal")
    
    def test_resolve_path_handles_nested_subfields(self):
        scope = Scope()
        data = {"pose": {"x": 1, "y": 2}}
        scope[(0, "sensor")] = data
        result = scope["/0.sensor"]
        assert result["pose"]["x"] == 1
    
    def test_resolve_path_converts_numeric_strings_to_int(self):
        scope = Scope()
        scope[(0, 1, "goal")] = "value"
        path = scope._resolve_path("/0.1.goal")
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
        assert ctx["/0.goal"] == "test_value"
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
        assert scope["/0.goal"] == "target_value"
        assert scope["plan.goal"] == "target_value"
    
    def test_tag_and_index_access_equivalent(self):
        scope = Scope()
        plan_ctx = scope.ctx(0, tag="plan")
        move_ctx = plan_ctx.child(1, tag="move")
        
        move_ctx["status"] = "running"
        
        # Index-based access
        assert scope["/0.1.status"] == "running"
        # Tag-based access  
        assert scope["move.status"] == "running"
    
    def test_nested_data_storage_and_retrieval(self):
        scope = Scope()
        ctx = scope.ctx(0, tag="sensor")
        
        # Store nested data
        pose_data = {"x": 10, "y": 20, "theta": 1.5}
        ctx["pose"] = pose_data
        
        # Retrieve via different paths
        assert scope["/0.pose"] == pose_data
        assert scope["sensor.pose"] == pose_data
        assert scope["/0.pose"]["x"] == 10
    
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
        assert scope["/0.0.goal"] == (10, 20, 0)
        
        # Invalid pattern: alias.index.variable should raise error
        with pytest.raises(KeyError):
            scope["sequence.0.goal"]
        
        # Move outputs arrival status
        move_ctx["arrived"] = True
        
        # Multiple access patterns for move output
        assert scope["move.arrived"] == True
        assert scope["/0.1.arrived"] == True

    def test_ctx_scope_hierarchy_resolution(self):
        """Test that Scope correctly resolves hierarchy through child Ctx"""
        scope = Scope()
        ctx = scope.ctx()
        
        child_ctx = ctx.child(0)
        child_ctx['target'] = (1, 2, 3)  # Scope stores at (0, 'target')
        
        # Explicit path access
        assert child_ctx['/0.target'] == (1, 2, 3)
        # Alias access should resolve to child's value  
        assert child_ctx['target'] == (1, 2, 3)
        
        # Parent updates root level
        ctx['target'] = (1, 2, 4)  # Scope stores at ((), 'target')
        assert ctx['/target'] == (1, 2, 4)
        assert ctx['target'] == (1, 2, 4)
        
        # Child should return its own value, not parent's newer value
        assert child_ctx['target'] == (1, 2, 4)
        # But explicit child path unchanged
        assert child_ctx['/0.target'] == (1, 2, 3)

    def test_ctx_hierarchy_fallback(self):
        """Test that child contexts fall back to parent data when key not found locally"""
        scope = Scope()
        ctx = scope.ctx()
        
        # Parent stores data
        ctx['target'] = (1, 2, 3)  # Scope stores at ((), 'target')
        
        # Child should be able to access parent's data (fallback)
        child_ctx = ctx.child(0)
        assert child_ctx['target'] == (1, 2, 3)  # Should fall back to parent


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
            scope["/0.1"]
    
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
        assert scope["/0.1.result"] == "nested_result"


class TestBoundCtx:
    """Test BoundCtx - used for modular Leaf nodes to bind input variables"""
    
    def test_bound_ctx_resolves_simple_binding(self):
        """Leaf node binds input 'x' to local context data 'position'"""
        scope = Scope()
        ctx = scope.ctx(0) 
        ctx["position"] = (10, 20, 0)
        
        # Leaf binds its input 'x' to the context's 'position' field
        bound_ctx = ctx.bind({"x": "position"})
        assert bound_ctx["x"] == (10, 20, 0)
    
    def test_bound_ctx_resolves_alias_binding(self):
        """Leaf node binds input to data from another tagged context"""
        scope = Scope()
        plan_ctx = scope.ctx(0, tag="plan")
        plan_ctx["goal"] = (5, 10, 1.5)
        
        # Move leaf binds its 'target' input to plan's goal
        move_ctx = scope.ctx(1, tag="move")
        bound_ctx = move_ctx.bind({"target": "plan.goal"})
        assert bound_ctx["target"] == (5, 10, 1.5)
    
    def test_bound_ctx_resolves_index_path_binding(self):
        """Leaf node binds input to data via index path"""
        scope = Scope()
        ctx_0 = scope.ctx(0)
        ctx_0["sensor_data"] = {"x": 1, "y": 2}
        
        # Processing leaf binds 'input' to sensor data via index path
        ctx_1 = scope.ctx(1)
        bound_ctx = ctx_1.bind({"input": "/0.sensor_data"})
        assert bound_ctx["input"]["x"] == 1
    
    def test_bound_ctx_falls_back_to_unbound_access(self):
        """Accessing unbound keys falls back to base context"""
        scope = Scope() 
        ctx = scope.ctx(0)
        ctx["local_data"] = "local_value"
        
        bound_ctx = ctx.bind({"x": "some_binding"})
        # Accessing non-bound key should fall back to base context
        assert bound_ctx["local_data"] == "local_value"
    
    def test_bound_ctx_setitem_with_binding(self):
        """Setting bound variable stores at bound location"""
        scope = Scope()
        plan_ctx = scope.ctx(0, tag="plan")
        move_ctx = scope.ctx(1, tag="move")
        
        # Move leaf binds 'target' to plan's goal
        bound_ctx = move_ctx.bind({"target": "plan"})
        bound_ctx["target"] = (15, 25, 0.5)
        
        # Should set data at the bound location (plan context)
        assert plan_ctx["plan"] == (15, 25, 0.5)
        assert scope["plan"] == (15, 25, 0.5)
    
    def test_bound_ctx_modular_leaf_scenario(self):
        """Realistic scenario: Move leaf with x,y inputs bound to tree data"""
        scope = Scope()
        
        # Tree has sensor and planner data
        sensor_ctx = scope.ctx(0, tag="sensor")
        sensor_ctx["current_pos"] = (0, 0, 0)
        
        planner_ctx = scope.ctx(1, tag="planner") 
        planner_ctx["target_pos"] = (10, 15, 1.57)
        
        # Move leaf defines inputs as 'current' and 'target'
        move_ctx = scope.ctx(2, tag="move")
        bound_move = move_ctx.bind({
            "current": "sensor.current_pos",
            "target": "planner.target_pos"  
        })
        
        # Leaf can access its inputs without knowing tree structure
        assert bound_move["current"] == (0, 0, 0)
        assert bound_move["target"] == (10, 15, 1.57)
        
        # Leaf outputs results locally
        bound_move["arrived"] = False
        bound_move["distance"] = 18.0
        
        # Outputs stored in leaf's own context
        assert move_ctx["arrived"] == False
        assert move_ctx["distance"] == 18.0
    
    def test_bound_ctx_child_preserves_bindings(self):
        """Child contexts preserve parent bindings"""
        scope = Scope()
        plan_ctx = scope.ctx(0, tag="plan") 
        plan_ctx["goal"] = (1, 1, 1)
        
        ctx = scope.ctx(1)
        bound_ctx = ctx.bind({"target": "plan.goal"})
        
        child_bound_ctx = bound_ctx.child(0, tag="child")
        assert child_bound_ctx["target"] == (1, 1, 1)
        assert child_bound_ctx.index_path == (1, 0)


class TestBoundScope:
    """Test BoundScope - used by BT roots to bind incoming context into tree scope"""
    
    def test_bound_scope_creation(self):
        """BT creates BoundScope to bind incoming context"""
        scope = Scope()
        incoming_ctx = scope.ctx(0)  # Incoming context from parent BT
        bindings = {"mission_goal": "external.target"}
        
        bound_scope = scope.bind(incoming_ctx, bindings)
        assert isinstance(bound_scope, BoundScope)
        assert bound_scope.base_ctx is incoming_ctx
        assert bound_scope.base_scope is scope
        assert bound_scope.bindings == bindings
    
    def test_bound_scope_getitem_with_at_prefix(self):
        """BT accesses bound variables from incoming context using @ prefix"""
        external_scope = Scope()
        external_ctx = external_scope.ctx(0, tag="external")
        external_ctx["target"] = (100, 200, 0)
        
        # BT creates its own scope and binds to external context
        bt_scope = Scope()
        bound_scope = bt_scope.bind(external_ctx, {"mission_goal": "external.target"})
        
        assert bound_scope["@mission_goal"] == (100, 200, 0)
    
    def test_bound_scope_setitem_with_at_prefix(self):
        """BT can write to bound variables in incoming context"""
        external_scope = Scope()
        external_ctx = external_scope.ctx(0, tag="external")
        
        bt_scope = Scope()
        bound_scope = bt_scope.bind(
            external_ctx, {"status": "external"}
        )
        bound_scope["@status"] = "completed"
        
        assert external_ctx["external"] == "completed"
    
    def test_bound_scope_contains_with_at_prefix(self):
        """BT can check if bound variables exist"""
        external_scope = Scope()
        external_ctx = external_scope.ctx(0, tag="external")
        external_ctx["data"] = "value"
        
        bt_scope = Scope()
        bound_scope = bt_scope.bind(external_ctx, {"input": "data"})
        
        assert "@input" in bound_scope
    
    def test_bound_scope_raises_keyerror_for_missing_binding(self):
        """BT gets error when accessing undefined binding"""
        external_scope = Scope()
        external_ctx = external_scope.ctx(0)
        
        bt_scope = Scope()
        bound_scope = bt_scope.bind(external_ctx, {"input": "source.data"})
        
        with pytest.raises(KeyError, match="Binding for missing not found"):
            bound_scope["@missing"]
    
    def test_bound_scope_fallback_to_base_scope(self):
        """BT can access its own scope data without @ prefix"""
        external_scope = Scope()
        external_ctx = external_scope.ctx(0)
        
        bt_scope = Scope()
        bt_scope[(0, "internal_data")] = "bt_value"
        
        bound_scope = bt_scope.bind(external_ctx, {"external_input": "some.binding"})
        
        # Access to BT's own scope (no @ prefix)
        assert bound_scope["/0.internal_data"] == "bt_value"
    
# #     def test_bound_scope_bt_root_scenario(self):
# #         """Realistic BT root scenario with incoming and internal context"""
# #         # External system context
# #         external_scope = Scope()
# #         mission_ctx = external_scope.ctx(0, tag="mission")
# #         mission_ctx["target"] = (50, 75, 1.0)
# #         mission_ctx["priority"] = "high"
        
# #         # BT creates its own scope and binds to external context
# #         bt_scope = Scope()
# #         bound_scope = bt_scope.bind(mission_ctx, {
# #             "goal": "mission.target",
# #             "urgency": "mission.priority"
# #         })
        
# #         # BT can access external data via bindings
# #         assert bound_scope["@goal"] == (50, 75, 1.0)
# #         assert bound_scope["@urgency"] == "high"
        
# #         # BT maintains its own internal state
# #         bt_scope[(0, "plan_status")] = "planning"
# #         assert bound_scope["/0.plan_status"] == "planning"
        
# #         # BT can report results back to external system
# #         bound_scope["@status"] = "in_progress"  # This would need a binding for "status"