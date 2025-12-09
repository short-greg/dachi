import pytest

from dachi.core._base import to_kind
from dachi.core._registry import Registry, RegistryEntry
from dachi.core._module import Module


class TestRegistry:

    def test_register_when_decorator_used_adds_class_to_registry(self):
        registry = Registry[Module]()

        @registry.register()
        class TestModule(Module):
            pass

        # Registry uses __qualname__ which includes the full path
        assert to_kind(TestModule) in registry.list_entries()

    def test_register_when_custom_name_provided_uses_custom_name(self):
        registry = Registry[Module]()
        
        @registry.register(name="CustomName")
        class TestModule(Module):
            pass
        
        assert "CustomName" in registry.list_entries()

    def test_register_when_tags_provided_stores_tags(self):
        registry = Registry[Module]()

        @registry.register(tags={"category": "test", "version": 1})
        class TestModule(Module):
            pass

        entry = registry[to_kind(TestModule)]
        assert entry.tags["category"] == "test"
        assert entry.tags["version"] == 1

    def test_register_when_description_provided_stores_description(self):
        registry = Registry[Module]()

        @registry.register(description="Test module description")
        class TestModule(Module):
            pass

        entry = registry[to_kind(TestModule)]
        assert entry.description == "Test module description"

    def test_register_when_duplicate_name_prints_warning(self, capsys):
        registry = Registry[Module]()
        
        @registry.register(name="Duplicate")
        class TestModule1(Module):
            pass
        
        @registry.register(name="Duplicate")
        class TestModule2(Module):
            pass
        
        captured = capsys.readouterr()
        assert "Warning: Overwriting existing entry 'Duplicate'" in captured.out

    def test_getitem_when_single_key_returns_entry(self):
        registry = Registry[Module]()

        @registry.register()
        class TestModule(Module):
            pass

        entry = registry[to_kind(TestModule)]
        assert entry.obj is TestModule

    def test_getitem_when_list_of_keys_returns_dict(self):
        registry = Registry[Module]()

        @registry.register()
        class TestModule1(Module):
            pass

        @registry.register()
        class TestModule2(Module):
            pass

        entries = registry[[to_kind(TestModule1), to_kind(TestModule2)]]
        assert isinstance(entries, dict)
        assert to_kind(TestModule1) in entries
        assert to_kind(TestModule2) in entries

    def test_getitem_when_key_not_found_raises_keyerror(self):
        registry = Registry[Module]()
        
        with pytest.raises(KeyError, match="Registry entry 'NonExistent' not found"):
            _ = registry["NonExistent"]

    def test_filter_when_no_criteria_returns_all(self):
        registry = Registry[Module]()
        
        @registry.register()
        class TestModule1(Module):
            pass
        
        @registry.register()
        class TestModule2(Module):
            pass
        
        results = registry.filter()
        assert len(results) == 2

    def test_filter_when_tags_criteria_returns_matching_only(self):
        registry = Registry[Module]()

        @registry.register(tags={"type": "encoder"})
        class EncoderModule(Module):
            pass

        @registry.register(tags={"type": "decoder"})
        class DecoderModule(Module):
            pass

        results = registry.filter(tags={"type": "encoder"})
        assert len(results) == 1
        assert to_kind(EncoderModule) in results

    def test_filter_when_obj_type_criteria_returns_matching_only(self):
        registry = Registry()

        @registry.register()
        class TestClass:
            pass

        @registry.register()
        def test_function():
            pass

        results = registry.filter(obj_type="class")
        assert len(results) == 1
        assert TestClass.__qualname__ in results

    def test_deregister_when_key_exists_removes_entry(self):
        registry = Registry[Module]()

        @registry.register()
        class TestModule(Module):
            pass

        registry.deregister(TestModule.__qualname__)
        assert TestModule.__qualname__ not in registry.list_entries()

    def test_list_entries_returns_all_registered_names(self):
        registry = Registry[Module]()

        @registry.register()
        class TestModule1(Module):
            pass

        @registry.register()
        class TestModule2(Module):
            pass

        entries = registry.list_entries()
        assert to_kind(TestModule1) in entries
        assert to_kind(TestModule2) in entries

    def test_list_types_returns_unique_object_types(self):
        registry = Registry()
        
        @registry.register()
        class TestClass:
            pass
        
        @registry.register()
        def test_function():
            pass
        
        types = registry.list_types()
        assert "class" in types
        assert "function" in types

    def test_list_tags_returns_all_unique_tag_keys(self):
        registry = Registry[Module]()
        
        @registry.register(tags={"category": "test", "version": 1})
        class TestModule1(Module):
            pass
        
        @registry.register(tags={"category": "prod", "priority": "high"})
        class TestModule2(Module):
            pass
        
        tags = registry.list_tags()
        assert "category" in tags
        assert "version" in tags
        assert "priority" in tags

