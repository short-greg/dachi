
from dachi.msg._instruct import style_formatter


class TestStyleFormat:

    def test_basic_string_formatting(self):
        """Tests standard Python formatting without special styling."""
        result = style_formatter("Hello, {name}!", name="Alice")
        assert result == "Hello, Alice!"

    def test_bullet_list_formatting(self):
        """Tests whether lists are correctly formatted as bullet points."""
        result = style_formatter("Shopping List:\n\n{x:bullet}", x=["Milk", "Eggs", "Bread"])
        expected = "Shopping List:\n\n- Milk\n- Eggs\n- Bread"
        print(result, expected)
        assert result == expected

    def test_bold_formatting(self):
        """Tests bold formatting."""
        result = style_formatter("Total: {total:bold}", total=100)
        expected = "Total: **100**"
        assert result == expected

    def test_italic_formatting(self):
        """Tests italic formatting."""
        result = style_formatter("Style: {text_:italic}", text_="Fancy")
        expected = "Style: *Fancy*"
        assert result == expected

    def test_mixed_formatting(self):
        """Tests multiple styles in one string."""
        result = style_formatter(
            "Receipt:\nItems:\n{x:bullet}\nTotal: {total:bold}",
            x=["Apple", "Banana"],
            total=12.50
        )
        expected = "Receipt:\nItems:\n- Apple\n- Banana\nTotal: **12.5**"
        assert result == expected

    def test_no_style_fallback(self):
        """Tests that standard formatting still works if no styles are applied."""
        result = style_formatter("Hello, {name}", name="World")
        assert result == "Hello, World"

    def test_empty_list_bullet(self):
        """Tests bullet formatting with an empty list."""
        result = style_formatter("Items:\n{x:bullet}", x=[])
        expected = "Items:\n"
        assert result == expected

    def test_numeric_formatting(self):
        """Tests if numbers are formatted properly."""
        result = style_formatter("The value is: {value}", value=42)
        assert result == "The value is: 42"

    def test_style_with_empty_string(self):
        """Tests that styling works even when the string is empty."""
        result = style_formatter("{x:bold}", x="")
        expected = "****"  # Bold empty string should still be two asterisks
        assert result == expected

    def test_positional_arguments(self):
        """Tests positional arguments instead of named arguments."""
        result = style_formatter("Values:\n{0:bullet}", ["A", "B", "C"])
        expected = "Values:\n- A\n- B\n- C"
        assert result == expected

    def test_function_call_style(self):
        """Tests positional arguments instead of named arguments."""
        result = style_formatter("Values:\n{0:bullet('-', 1)}", ["A", "B", "C"])
        expected = "Values:\n - A\n - B\n - C"
        assert result == expected
