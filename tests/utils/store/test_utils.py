from dachi.utils import store as utils


class TestGetMember(object):

    def test_get_member_gets_immediate_child(self):

        class X:
            y = 2

        x = X()

        assert utils.get_member(
            x, 'y'
        ) == 2

    def test_get_member_gets_sub_child(self):

        class X:
            y = 2

            def __getattr__(self, key):

                o = X()
                object.__setattr__(self, key, o)
                return o

        x = X()

        assert utils.get_member(
            x, 'z.y'
        ) == 2


class TestGetOrSpawn(object):

    def test_get_or_spawn_doesnt_spawn_new_state(self):

        state = {'child': {}}
        target = state['child']
        child = utils.get_or_spawn(state, 'child')
        assert child is target

    def test_get_or_spawn_spawns_new_state(self):

        state = {'child': {}}
        target = state['child']
        child = utils.get_or_spawn(state, 'other')
        assert not child is target


class TestGetOrSet(object):

    def test_get_or_set_doesnt_set_new_value(self):

        state = {'val': 2}
        target = state['val']
        child = utils.get_or_set(state, 'val', 3)
        assert child is target

    def test_get_or_spawn_sets_a_new_value(self):

        state = {}
        child = utils.get_or_set(state, 'val', 3)
        assert child == 3
