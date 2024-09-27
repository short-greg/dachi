import pytest
from .test_core import SimpleStruct
from dachi._core import _utils as utils


class TestStrFormatter(object):

    def test_formatter_formats_positional_variables(self):

        assert utils.str_formatter(
            '{} {}', 1, 2
        ) == '1 2'

    def test_formatter_formats_positional_variables(self):

        assert utils.str_formatter(
            '{0} {1}', 1, 2
        ) == '1 2'

    def test_formatter_formats_named_variables(self):

        assert utils.str_formatter(
            '{x} {y}', x=1, y=2
        ) == '1 2'

    def test_formatter_raises_error_if_positional_and_named_variables(self):

        with pytest.raises(ValueError):
            utils.str_formatter(
                '{0} {y}', 1, y=2
            )

    def test_get_variables_gets_all_pos_variables(self):

        assert utils.get_str_variables(
            '{0} {1}'
        ) == [0, 1]

    def test_get_variables_gets_all_named_variables(self):

        assert utils.get_str_variables(
            '{x} {y}'
        ) == ['x', 'y']
