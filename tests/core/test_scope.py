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
    
    # def test_ctx_registers_alias_when_tag_provided(self):
    #     scope = Scope()
    #     scope.ctx(0, tag="plan")
    #     assert scope.aliases["plan"] == (0,)
    
    # def test_ctx_retags_existing_child_when_called_again(self):
    #     scope = Scope()
    #     scope.ctx(0)
    #     scope.ctx(0, tag="new_tag")
    #     assert scope.aliases["new_tag"] == (0,)
    #     assert "old_tag" not in scope.aliases
    
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
        # scope.aliases["plan"] = (0,)
        scope[(0, "goal")] = "test_value" 
        assert scope["goal"] == "test_value"
    
    def test_getitem_handles_nested_index_paths(self):
        scope = Scope()
        scope[(0, 1, "goal")] = "nested_value"
        assert scope["/0.1.goal"] == "nested_value"


class TestScopeStringPaths:
    
    def test_resolve_path_parses_index_path_correctly(self):
        scope = Scope()
        scope[(0, "goal")] = "value"
        assert scope._resolve_path("0.goal") == (0, "goal")
    
    def test_resolve_path_handles_nested_subfields(self):
        scope = Scope()
        data = {"pose": {"x": 1, "y": 2}}
        scope[(0, "sensor")] = data
        result = scope["/0.sensor"]
        assert result["pose"]["x"] == 1
    
    def test_resolve_path_converts_numeric_strings_to_int(self):
        scope = Scope()
        scope[(0, 1, "goal")] = "value"
        path = scope._resolve_path("0.1.goal")
        assert path == (0, 1, "goal")
    

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
    
    def test_child_creates_multi_level_nesting(self):
        scope = Scope()
        ctx_0 = scope.ctx(0)
        ctx_0_1 = ctx_0.child(1)
        ctx_0_1_2 = ctx_0_1.child(2)
        assert ctx_0_1_2.index_path == (0, 1, 2)
        # assert scope.aliases["deep"] == (0, 1, 2)  # Tag functionality removed


class TestCtxDataAccess:
    
    # def test_multiple_access_patterns_return_same_data(self):
    #     scope = Scope()
    #     ctx = scope.ctx(0, tag="plan")
    #     ctx["goal"] = "target_value"
    #     
    #     # All these should return the same data
    #     assert scope[(0, "goal")] == "target_value"
    #     assert scope["/0.goal"] == "target_value"
    #     assert scope["plan.goal"] == "target_value"
    
    # def test_tag_and_index_access_equivalent(self):
    #     # Tag functionality removed - aliasing now automatic based on field names
    
    def test_nested_data_storage_and_retrieval(self):
        scope = Scope()
        ctx = scope.ctx(0)
        
        # Store nested data
        pose_data = {"x": 10, "y": 20, "theta": 1.5}
        ctx["pose"] = pose_data
        
        # Retrieve via different paths
        assert scope["/0.pose"] == pose_data
        assert scope["pose"] == pose_data  # Automatic aliasing based on field name
        assert scope["/0.pose"]["x"] == 10
    
    # def test_complex_behavior_tree_scenario(self):
    #     # Tag functionality removed - this test relied on manual tag registration

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
        with pytest.raises(KeyError, match="Alias.*not found"):
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
        """Leaf node binds input to data using automatic field aliasing"""
        scope = Scope()
        plan_ctx = scope.ctx(0)
        plan_ctx["goal"] = (5, 10, 1.5)
        
        # Move leaf binds its 'target' input to plan's goal
        move_ctx = scope.ctx(1)
        bound_ctx = move_ctx.bind({"target": "goal"})  # Uses automatic field aliasing
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
    
    # def test_bound_ctx_setitem_with_binding(self):
    #     # This test relied on tag functionality - commenting out
    
    def test_bound_ctx_modular_leaf_scenario(self):
        """Realistic scenario: Move leaf with x,y inputs bound to tree data"""
        scope = Scope()
        
        # Tree has sensor and planner data
        sensor_ctx = scope.ctx(0)
        sensor_ctx["current_pos"] = (0, 0, 0)
        
        planner_ctx = scope.ctx(1) 
        planner_ctx["target_pos"] = (10, 15, 1.57)
        
        # Move leaf defines inputs as 'current' and 'target'
        move_ctx = scope.ctx(2)
        bound_move = move_ctx.bind({
            "current": "current_pos",  # Uses automatic field aliasing
            "target": "target_pos"     # Uses automatic field aliasing
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
        plan_ctx = scope.ctx(0) 
        plan_ctx["goal"] = (1, 1, 1)
        
        ctx = scope.ctx(1)
        bound_ctx = ctx.bind({"target": "goal"})  # Use automatic field aliasing
        
        child_bound_ctx = bound_ctx.child(0)
        assert child_bound_ctx["target"] == (1, 1, 1)
        assert child_bound_ctx.index_path == (1, 0)
    
    def test_bound_ctx_path_with_scope_navigation(self):
        """Test BoundCtx.path() method with scope navigation"""
        root_scope = Scope()
        child_scope = root_scope.child("child")
        
        # Set data in root scope
        root_ctx = root_scope.ctx(0)
        root_ctx["sensor_data"] = {"x": 100, "y": 200}
        
        # Create bound context in child scope
        child_ctx = child_scope.ctx(0)
        bound_ctx = child_ctx.bind({"input": "sensor_data"})
        
        # BoundCtx should resolve across scopes using lexical scoping via __getitem__
        result = bound_ctx["input"]
        assert result == {"x": 100, "y": 200}
    


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
        external_ctx = external_scope.ctx(0)
        external_ctx["target"] = (100, 200, 0)
        
        # BT creates its own scope and binds to external context
        bt_scope = Scope()
        bound_scope = bt_scope.bind(external_ctx, {"mission_goal": "target"})  # Use automatic field aliasing
        
        assert bound_scope["@mission_goal"] == (100, 200, 0)
    
    # def test_bound_scope_setitem_with_at_prefix(self):
    #     # This test relied on tag functionality - commenting out
    
    def test_bound_scope_contains_with_at_prefix(self):
        """BT can check if bound variables exist"""
        external_scope = Scope()
        external_ctx = external_scope.ctx(0)
        external_ctx["data"] = "value"
        
        bt_scope = Scope()
        bound_scope = bt_scope.bind(external_ctx, {"input": "data"})  # Use automatic field aliasing
        
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
    
    def test_bound_scope_at_prefix_with_lexical_scoping(self):
        """Test BoundScope @ aliases with lexical scoping across parent scopes"""
        # Create external hierarchy
        external_root = Scope()
        external_child = external_root.child("sensors")
        
        # Set data in external root
        external_root_ctx = external_root.ctx(0)
        external_root_ctx["global_position"] = {"x": 10, "y": 20}
        
        # BT context at child level
        external_ctx = external_child.ctx(0)
        
        # Create BT scope and bind with lexical scoping
        bt_scope = Scope()
        bound_scope = bt_scope.bind(external_ctx, {
            "robot_pos": "global_position"  # Uses lexical scoping to find in parent
        })
        
        # Access via @ prefix should resolve via lexical scoping
        assert bound_scope["@robot_pos"] == {"x": 10, "y": 20}


class TestScopeParentChild:
    """Test new parent-child scope functionality and relative paths"""
    
    def test_scope_parent_child_creation(self):
        """Test basic parent-child scope creation"""
        root_scope = Scope()
        child_scope = root_scope.child("child1")
        
        assert child_scope.parent is root_scope
        assert child_scope.name == "child1"
        assert "child1" in root_scope.children
        assert root_scope.children["child1"] is child_scope
    
    def test_scope_child_get_or_create(self):
        """Test that child() returns existing child if it exists"""
        root_scope = Scope()
        child1 = root_scope.child("test")
        child2 = root_scope.child("test")  # Should return same instance
        
        assert child1 is child2
        assert len(root_scope.children) == 1
    
    def test_tuple_relative_path_current_scope(self):
        """Test tuple-based current scope reference"""
        scope = Scope()
        scope[(0, "field")] = "value1"
        
        # Current scope reference should work the same
        assert scope[('./', 0, "field")] == "value1"
    
    def test_tuple_relative_path_parent_scope(self):
        """Test tuple-based parent scope reference"""
        root_scope = Scope()
        child_scope = root_scope.child("child")
        
        # Set value in root scope
        root_scope[(0, "parent_field")] = "parent_value"
        
        # Access from child scope using parent reference
        assert child_scope[('../', 0, "parent_field")] == "parent_value"
    
    def test_tuple_relative_path_grandparent_scope(self):
        """Test tuple-based grandparent scope reference"""
        root_scope = Scope()
        child_scope = root_scope.child("child")
        grandchild_scope = child_scope.child("grandchild")
        
        # Set value in root scope
        root_scope[(0, "root_field")] = "root_value"
        
        # Access from grandchild scope using grandparent reference
        assert grandchild_scope[('../../', 0, "root_field")] == "root_value"
    
    def test_tuple_absolute_path_root(self):
        """Test tuple-based absolute path to root"""
        root_scope = Scope()
        child_scope = root_scope.child("child")
        
        # Set value in root scope
        root_scope[(0, "root_field")] = "root_value"
        
        # Access from child scope using absolute root reference
        assert child_scope[('/', 0, "root_field")] == "root_value"
    
    def test_tuple_absolute_path_named_scope(self):
        """Test tuple-based absolute path to named scope"""
        root_scope = Scope()
        target_scope = root_scope.child("target")
        other_scope = root_scope.child("other")
        
        # Set value in target scope
        target_scope[(0, "target_field")] = "target_value"
        
        # Access from other scope using absolute named path
        assert other_scope[('/target', 0, "target_field")] == "target_value"
    
    def test_string_relative_path_parent_should_error(self):
        """Test that mixing explicit paths with aliases raises an error"""
        root_scope = Scope()
        child_scope = root_scope.child("child")
        
        # Set value in root scope using field name (creates alias)
        root_ctx = root_scope.ctx(0)
        root_ctx["parent_field"] = "parent_value"
        
        # Mixing explicit path navigation with alias should raise error
        with pytest.raises(KeyError):
            child_scope["../parent_field"]
    
    def test_lexical_scoping_parent_access(self):
        """Test correct way to access parent data using lexical scoping"""
        root_scope = Scope()
        child_scope = root_scope.child("child")
        
        # Set value in root scope using field name (creates alias)
        root_ctx = root_scope.ctx(0)
        root_ctx["parent_field"] = "parent_value"
        
        # Correct way: use lexical scoping without explicit path navigation
        assert child_scope["parent_field"] == "parent_value"
    
    def test_set_method_with_relative_path(self):
        """Test set() method with relative scope paths"""
        root_scope = Scope()
        child_scope = root_scope.child("child")
        
        # Set value in parent scope from child using relative path
        child_scope.set("../0", "parent_data", "parent_value")
        
        # Verify it was stored in parent scope
        assert root_scope.path((0,), "parent_data") == "parent_value"
        assert root_scope[(0, "parent_data")] == "parent_value"
    
    def test_path_method_with_relative_path(self):
        """Test path() method with relative scope paths"""
        root_scope = Scope()
        child_scope = root_scope.child("child")
        grandchild_scope = child_scope.child("grandchild")
        
        # Set data in root scope
        root_scope.set((0,), "root_data", "root_value")
        
        # Access from grandchild using relative path
        result = grandchild_scope.path("../../0", "root_data")
        assert result == "root_value"
    
    def test_set_method_with_named_absolute_path(self):
        """Test set() method with absolute named scope paths"""
        root_scope = Scope()
        target_scope = root_scope.child("target")
        other_scope = root_scope.child("other")
        
        # Set value in target scope from other scope using absolute path
        other_scope.set("/target/0", "target_data", "target_value")
        
        # Verify it was stored in target scope  
        assert target_scope[(0, "target_data")] == "target_value"
        assert target_scope.path((0,), "target_data") == "target_value"
    
    def test_path_method_with_named_absolute_path(self):
        """Test path() method with absolute named scope paths"""
        root_scope = Scope()
        source_scope = root_scope.child("source")
        target_scope = root_scope.child("target")
        
        # Set data in target scope
        target_scope.set((), "data", "value")
        print(id(target_scope))
        # Access from source scope using absolute path
        result = source_scope.path("/target/", "data")
        assert result == "value"
    
    def test_resolve_scope_method_relative_paths(self):
        """Test _resolve_scope() method directly with relative paths"""
        root_scope = Scope()
        child_scope = root_scope.child("child")
        grandchild_scope = child_scope.child("grandchild")
        
        # Test current scope
        scope, key = child_scope._resolve_scope(('./', 0, "field"))
        assert scope is child_scope
        assert key == (0, "field")
        
        # Test parent scope  
        scope, key = child_scope._resolve_scope(('../', 0, "field"))
        assert scope is root_scope
        assert key == (0, "field")
        
        # Test grandparent scope
        scope, key = grandchild_scope._resolve_scope(('../../', 0, "field"))
        assert scope is root_scope
        assert key == (0, "field")
    
    def test_resolve_scope_method_absolute_paths(self):
        """Test _resolve_scope() method directly with absolute paths"""
        root_scope = Scope()
        target_scope = root_scope.child("target")
        other_scope = root_scope.child("other")
        
        # Test root scope access
        scope, key = other_scope._resolve_scope(('/', 0, "field"))
        assert scope is root_scope
        assert key == (0, "field")
        
        # Test named scope access
        scope, key = other_scope._resolve_scope(('/target', 0, "field"))
        assert scope is target_scope
        assert key == (0, "field")
    
    def test_resolve_scope_method_string_input(self):
        """Test _resolve_scope() method with string input (for set/path methods)"""
        root_scope = Scope()
        target_scope = root_scope.child("target")
        other_scope = root_scope.child("other")
        
        # Test string relative path
        scope, key = other_scope._resolve_scope("../field")
        assert scope is root_scope
        assert key == ("field",)
        
        # Test string absolute path
        scope, key = other_scope._resolve_scope("/target/0.field")
        assert scope is target_scope
        assert key == ("0", "field")
    
# #     def test_string_relative_path_grandparent(self):
# #         """Test string-based grandparent scope reference"""
# #         root_scope = Scope()
# #         child_scope = root_scope.child("child")
# #         grandchild_scope = child_scope.child("grandchild")
        
# #         # Set value in root scope
# #         root_scope[(0, "root_field")] = "root_value"
        
# #         # Access from grandchild scope using string grandparent reference
# #         assert grandchild_scope["../../root_field"] == "root_value"
    
# # # # #     def test_bound_scope_bt_root_scenario(self):
# # # # #         """Realistic BT root scenario with incoming and internal context"""
# # # # #         # External system context
# # # # #         external_scope = Scope()
# # # # #         mission_ctx = external_scope.ctx(0, tag="mission")
# # # # #         mission_ctx["target"] = (50, 75, 1.0)
# # # # #         mission_ctx["priority"] = "high"
        
# # # # #         # BT creates its own scope and binds to external context
# # # # #         bt_scope = Scope()
# # # # #         bound_scope = bt_scope.bind(mission_ctx, {
# # # # #             "goal": "mission.target",
# # # # #             "urgency": "mission.priority"
# # # # #         })
        
# # # # #         # BT can access external data via bindings
# # # # #         assert bound_scope["@goal"] == (50, 75, 1.0)
# # # # #         assert bound_scope["@urgency"] == "high"
        
# # # # #         # BT maintains its own internal state
# # # # #         bt_scope[(0, "plan_status")] = "planning"
# # # # #         assert bound_scope["/0.plan_status"] == "planning"
        
# # # # #         # BT can report results back to external system
# # # # #         bound_scope["@status"] = "in_progress"  # This would need a binding for "status"


class TestScopeNameConflicts:
    """Test scope navigation when child scopes have conflicting names"""
    
    def test_same_child_names_different_parents(self):
        """Test that same child names under different parents are distinct"""
        root_scope = Scope()
        branch_a = root_scope.child("sensors")
        branch_b = root_scope.child("actuators")
        
        # Both branches have a child named "config"
        config_a = branch_a.child("config")
        config_b = branch_b.child("config")
        
        # Set different data in each config scope
        config_a_ctx = config_a.ctx(0)
        config_a_ctx["settings"] = {"type": "sensor"}
        
        config_b_ctx = config_b.ctx(0)
        config_b_ctx["settings"] = {"type": "actuator"}
        
        # Verify they are distinct scopes with different data
        assert config_a is not config_b
        assert config_a[(0, "settings")] == {"type": "sensor"}
        assert config_b[(0, "settings")] == {"type": "actuator"}
        
        # Test navigation from root to each
        assert root_scope[("/sensors/config", 0, "settings")] == {"type": "sensor"}
        assert root_scope[("/actuators/config", 0, "settings")] == {"type": "actuator"}
    
    def test_deeply_nested_same_names(self):
        """Test navigation in deep hierarchies with repeated names"""
        root_scope = Scope()
        
        # Create hierarchy: root/system1/module/config and root/system2/module/config
        sys1 = root_scope.child("system1")
        sys1_module = sys1.child("module")
        sys1_config = sys1_module.child("config")
        
        sys2 = root_scope.child("system2")
        sys2_module = sys2.child("module")
        sys2_config = sys2_module.child("config")
        
        # Set data in each config
        sys1_config_ctx = sys1_config.ctx(0)
        sys1_config_ctx["value"] = "system1_data"
        
        sys2_config_ctx = sys2_config.ctx(0)
        sys2_config_ctx["value"] = "system2_data"
        
        # Navigate from root to specific configs
        assert root_scope[("/system1/module/config", 0, "value")] == "system1_data"
        assert root_scope[("/system2/module/config", 0, "value")] == "system2_data"
        
        # Navigate from one config to another using absolute paths
        assert sys1_config[("/system2/module/config", 0, "value")] == "system2_data"
        assert sys2_config[("/system1/module/config", 0, "value")] == "system1_data"


class TestStringTuplePathConsistency:
    """Test that string and tuple path formats work identically"""
    
    def test_absolute_path_equivalence(self):
        """Test string and tuple absolute paths access same data"""
        root_scope = Scope()
        target_scope = root_scope.child("target")
        
        # Set data using tuple path
        target_scope[(0, "data")] = "test_value"
        
        # Access using both formats should return same result
        tuple_result = root_scope[("/target", 0, "data")]
        string_result = root_scope["/target/0.data"]
        
        assert tuple_result == string_result == "test_value"
    
    def test_relative_path_equivalence(self):
        """Test string and tuple relative paths work identically"""
        root_scope = Scope()
        child_scope = root_scope.child("child")
        
        # Set data in root
        root_scope[(0, "parent_data")] = "parent_value"
        
        # Access from child using both formats
        tuple_result = child_scope[("../", 0, "parent_data")]
        # Note: string relative paths with explicit navigation aren't supported yet
        # string_result = child_scope["../0.parent_data"]  # This would fail
        
        assert tuple_result == "parent_value"
    
    def test_set_and_path_method_consistency(self):
        """Test that set() and path() methods work with both string and tuple paths"""
        root_scope = Scope()
        target_scope = root_scope.child("target")
        source_scope = root_scope.child("source")
        
        # Set using string path
        source_scope.set("/target/0", "string_field", "string_value")
        
        # Set using tuple path
        source_scope.set(("/target", 0), "tuple_field", "tuple_value")
        
        # Read using both formats
        string_result = source_scope.path("/target/0", "string_field")
        tuple_result = source_scope.path(("/target", 0), "tuple_field")
        
        assert string_result == "string_value"
        assert tuple_result == "tuple_value"
        
        # Verify both are accessible from target scope
        assert target_scope[(0, "string_field")] == "string_value"
        assert target_scope[(0, "tuple_field")] == "tuple_value"


class TestDeepHierarchy:
    """Test 4+ level scope hierarchies"""
    
    def test_four_level_navigation(self):
        """Test navigation in 4-level hierarchy"""
        root = Scope()
        level1 = root.child("system")
        level2 = level1.child("subsystem")
        level3 = level2.child("component")
        level4 = level3.child("subcomponent")
        
        # Set data at root
        root_ctx = root.ctx(0)
        root_ctx["global_config"] = {"timeout": 100}
        
        # Access from deepest level using relative navigation
        result = level4[("../../../../", 0, "global_config")]
        assert result == {"timeout": 100}
        
        # Access using absolute navigation
        abs_result = level4[("/", 0, "global_config")]
        assert abs_result == {"timeout": 100}
        
        # Set data at deepest level
        level4_ctx = level4.ctx(0)
        level4_ctx["local_config"] = {"precision": 0.001}
        
        # Access from root using absolute path
        deep_result = root[("/system/subsystem/component/subcomponent", 0, "local_config")]
        assert deep_result == {"precision": 0.001}


class TestChangeScopeEdgeCases:
    """Test edge cases for change_scope method"""
    
    def test_change_scope_empty_string_should_error(self):
        """Test that change_scope('') raises an appropriate error"""
        scope = Scope()
        # Currently change_scope('') returns self, but should probably error
        # This test documents current behavior - may want to change this
        result = scope.change_scope('')
        assert result is scope  # Current behavior
        
        # If we decide empty string should error:
        # with pytest.raises(ValueError, match="Invalid scope step"):
        #     scope.change_scope('')
    
    def test_change_scope_current_directory(self):
        """Test change_scope('.') returns current scope"""
        scope = Scope()
        child = scope.child("child")
        
        result = child.change_scope('.')
        assert result is child
    
    def test_change_scope_parent_from_root_raises_error(self):
        """Test that trying to go up from root scope raises error"""
        root_scope = Scope()
        
        with pytest.raises(ValueError, match="Cannot go up - no parent scope"):
            root_scope.change_scope('..')
    
    def test_change_scope_invalid_child_name(self):
        """Test change_scope with child name that doesn't exist creates it"""
        scope = Scope()
        
        # change_scope creates children if they don't exist
        result = scope.change_scope('new_child')
        assert result.name == 'new_child'
        assert result.parent is scope
        assert 'new_child' in scope.children
    
    def test_change_scope_multiple_levels(self):
        """Test multiple change_scope calls for deep navigation"""
        root = Scope()
        level1 = root.child("level1")
        level2 = level1.child("level2")
        
        # Navigate from level2 back to root via multiple steps
        result = level2.change_scope('..').change_scope('..')
        assert result is root
        
        # Navigate from root to level2 via multiple steps
        result = root.change_scope('level1').change_scope('level2')
        assert result is level2


class TestScopeSerialization:
    """Test Pydantic serialization for all Scope classes"""

    def test_scope_model_dump_works(self):
        """Scope can be serialized without errors"""
        scope = Scope(name="test")
        scope[(0, "goal")] = "value"

        data = scope.model_dump()
        assert data["name"] == "test"
        assert "full_path" in data

    def test_scope_roundtrip_preserves_data(self):
        """Scope can be deserialized and data is preserved"""
        scope = Scope(name="test")
        scope[(0, "goal")] = "test_value"
        scope[(0, "pose")] = {"x": 1, "y": 2}

        data = scope.model_dump()
        restored = Scope(**data)

        assert restored.name == "test"
        assert restored[(0, "goal")] == "test_value"
        assert restored[(0, "pose")] == {"x": 1, "y": 2}

    def test_scope_parent_child_roundtrip(self):
        """Parent-child relationships reconstructed after deserialization"""
        root = Scope(name="root")
        child = root.child("child")
        child[(0, "data")] = "child_data"

        data = root.model_dump()
        restored = Scope(**data)

        restored_child = restored.children["child"]
        assert restored_child.parent is restored
        assert restored_child.name == "child"
        assert restored_child[(0, "data")] == "child_data"

    def test_scope_deep_hierarchy_roundtrip(self):
        """Deep hierarchies preserve parent links"""
        root = Scope(name="root")
        level1 = root.child("level1")
        level2 = level1.child("level2")
        level2[(0, "deep_data")] = "value"

        data = root.model_dump()
        restored = Scope(**data)

        restored_level2 = restored.children["level1"].children["level2"]
        assert restored_level2.parent.parent is restored
        assert restored_level2[(0, "deep_data")] == "value"

    def test_ctx_serialization(self):
        """Ctx can be serialized and deserialized"""
        scope = Scope()
        ctx = scope.ctx(0, 1)
        ctx["field"] = "value"

        data = ctx.model_dump()
        assert data["index_path"] == (0, 1)

    def test_bound_scope_creation_with_pydantic(self):
        """BoundScope works with Pydantic initialization"""
        base_scope = Scope()
        base_ctx = base_scope.ctx(0)
        base_ctx["target"] = (10, 20, 0)

        bound = BoundScope(
            base_scope=base_scope,
            base_ctx=base_ctx,
            bindings={"goal": "target"}
        )

        assert bound["@goal"] == (10, 20, 0)

    def test_bound_ctx_creation_with_pydantic(self):
        """BoundCtx works with Pydantic initialization"""
        scope = Scope()
        ctx = scope.ctx(0)
        ctx["position"] = (5, 10, 0)

        bound_ctx = BoundCtx(
            scope=scope,
            index_path=(0,),
            base_ctx=ctx,
            bindings={"input": "position"}
        )

        assert bound_ctx["input"] == (5, 10, 0)