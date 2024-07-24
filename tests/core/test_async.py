from typing import Any
from dachi._core import _process as core
from dachi._core import _async


class Append(core.Module):

    def __init__(self, append: str):
        super().__init__()
        self._append = append

    def forward(self, name: str='') -> Any:
        return name + self._append


class WaitAppend(core.Module):

    def __init__(self, append: str):
        super().__init__()
        self._append = append

    def forward(self, name: str='') -> Any:
        return name + self._append


class TestAsyncModule:

    def test_two_appends_output_correct_result(self):

        module = _async.AsyncModule([
            Append('_t'),
            Append('_x')
        ])

        t = module.link([core.Args('xyz'), core.Args('xyz')])
        assert t.val[0] == 'xyz_t'
        assert t.val[1] == 'xyz_x'

    def test_two_appends_output_undefined_if_an_input_is_undefined(self):

        module = _async.AsyncModule([
            Append('_t'),
            Append('_x')
        ])

        t1 = core.T(core.UNDEFINED)
        t2 = core.T('xyz')

        t = module.link([core.Args(t1), core.Args(t2)])
        assert t[0].val is core.UNDEFINED
        assert t[1].val is core.UNDEFINED

    def test_probe_returns_tuple_of_values(self):
        """_summary_
        """
        module = _async.AsyncModule([
            Append('_t'),
            Append('_x')
        ])

        t1 = core.T(core.UNDEFINED)
        t2 = core.T('xyz')

        t = module.link([core.Args(t1), core.Args(t2)])
        
        res = t.probe(
            {t1: 'abc'}
        )
        assert res[0] == 'abc_t'
        assert res[1] == 'xyz_x'

    def test_probe_idx_returns_value(self):

        module = _async.AsyncModule([
            Append('_t'),
            Append('_x')
        ])

        t1 = core.T(core.UNDEFINED)
        t2 = core.T('xyz')

        t = module.link([core.Args(t1), core.Args(t2)])
        t = t[0]
        res = t.probe(
            {t1: 'abc'}
        )
        assert res == 'abc_t'

    def test_probe_idx_returns_value(self):

        module = _async.AsyncModule([
            Append('_t'),
            Append('_x')
        ])

        res = module([core.Args('xyz'), core.Args('abc')])
        assert res[1] == 'abc_x'
