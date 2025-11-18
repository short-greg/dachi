"""
This module contains parsers for parsing text responses into
structured data.

The API for this module is as follows:
- Parser: Base class for all parsers.
- forward(val: str | None) -> typing.List | None: Parses the complete response.
- delta(val: str | None, delta_store: typing.Dict, is_last: bool=True) -> typing.List | None: Parses the response incrementally.
- render(data) -> str: Renders the structured data back into a string.

"""

# 1st party 
from abc import abstractmethod
import typing
import csv
import io
from collections import OrderedDict


# local
from ._process import Process
from .. import utils


class Parser(Process):
    """Base class for parsers. 
    It converts the input text
    into a list of objects
    """
    @abstractmethod
    def forward(self, val: str | None) -> typing.List | None:
        """Parse the full response into a list of objects

        Args:
            val (str | None): The value to parse
        Returns:
            typing.List | None: The list of parsed objects or None if not enough data
        """
        pass

    @abstractmethod
    def delta(
        self, 
        val: str | None, 
        delta_store: typing.Dict, 
        is_last: bool=True
    ) -> typing.List | None:
        """Parse the response into a list of objects

        Args:
            val (str | None): The value to parse
            delta_store (typing.Dict): The dictionary to store deltas
            streamed (bool, optional): Whether this is a streamed response. Defaults to False.
            is_last (bool, optional): Whether this is the last chunk. Defaults to True.

        Returns:
            typing.List | None: The list of parsed objects or None if not enough data
        """
        pass

    @abstractmethod
    def render(self, data) -> str:
        pass

PARSER = typing.TypeVar('PARSER', bound=Parser)

# TODO: Look into how to get CSVRowParser
# combined with CSVout. Ensure CSVOut
# can process row by row or if it already
# can
class CSVRowParser(Parser):
    """
    Dynamically parse CSV data, returning new rows as accumulated. 
    The header will be returned along with them if used.
    """

    delimiter: str = ','
    use_header: bool = True

    def forward(
        self, 
        val: str | None,
    ):
        """Parse complete CSV response into rows
        
        Args:
            val (str | None): The complete CSV string to parse
            
        Returns:
            List: Parsed CSV rows as dictionaries (if use_header=True) or lists
        """
        if val is None:
            return []
            
        rows = list(
            csv.reader(io.StringIO(val), delimiter=self.delimiter)
        )
        
        if not rows:
            return []
        
        if self.use_header:
            header = rows[0]
            data_rows = rows[1:]
            return [OrderedDict(zip(header, row)) for row in data_rows]
        
        return rows

    def delta(
        self, 
        val: str | None, 
        delta_store: typing.Dict=None, 
        is_last: bool=True
    ) -> typing.List | None:
        """
        Parses CSV data incrementally using csv.reader.
        """
        # resp = self.handle_null(resp, '')
        delta_store = delta_store if delta_store is not None else {}

        val = utils.acc(delta_store, 'val', val, '')
        row = utils.get_or_set(delta_store, 'row', 0)
        header = utils.get_or_set(
            delta_store, 'header', None
        )
        # Process accumulated data using csv.reader
        # csv_data = io.StringIO(delta_store['val'])

        rows = list(
            csv.reader(io.StringIO(val), delimiter=self.delimiter)
        )
        new_rows = []
        for i, row in enumerate(rows[row:]):  # Only return new rows
            new_rows.append(row)

        if len(new_rows) == 0:
            return utils.UNDEFINED
        
        if not is_last:
            new_rows.pop()

        if len(new_rows) == 0:
            return utils.UNDEFINED

        if (
            self.use_header is True 
            and delta_store['header'] is None
        ):
            delta_store['header'] = new_rows.pop(0)
            utils.acc(delta_store, 'row', 1)

        header = delta_store['header']
        utils.acc(delta_store, 'row', len(new_rows))
        if len(new_rows) == 0:
            return utils.UNDEFINED
        
        if self.use_header:
            return [OrderedDict(zip(header, row)) for row in new_rows]
        return new_rows

    def render(self, data) -> str:
        """

        Args:
            data: An iterable of rows. If header is set to true

        Returns:
            str: the rendered CSV
        """
        output = io.StringIO()
        writer = csv.writer(output, delimiter=self.delimiter, lineterminator='\n')
        
        if self.use_header:
            header = list(data[0].keys())
            writer.writerow(header)
            for row in data:
                writer.writerow(list(row.values()))
        else:
            for row in data:
                writer.writerow(row)
        
        return output.getvalue()

# Review this and see how it differs
# from KVParser
class CharDelimParser(Parser):
    """Parses based on a defined character

    Example:
        parser = CharDelimParser(sep=',')
        resp = Resp(msg=Msg(role='assistant', text='value1,value2,value3,partia'))
        delta_store = {}
        result = parser.forward(resp, delta_store, streamed=True, is_last=False)
        print(result)  # Output: ['value1', 'value2', 'value3']
        
        # Simulate receiving the rest of the data
        resp = Resp(msg=Msg(role='assistant', text='l_value'))
        result = parser.forward(resp, delta_store, streamed=True, is_last=True)
        print(result)  # Output: ['partial_value']
    """
    sep: str = ','

    def forward(self, val: str | None) -> typing.List:
        """Parse complete delimited response
        
        Args:
            val (str | None): The complete string to parse
            
        Returns:
            List: List of parsed values split by separator
        """
        if val is None:
            return []
            
        if val == '':
            return []
            
        return val.split(self.sep)

    def delta(
        self, 
        val, 
        delta_store: typing.Dict=None, 
        is_last: bool=True
    ) -> typing.List | None:
        """Parse the response based on the defined character separator

        Args:
            val: The value to parse
            delta_store (typing.Dict, optional): The delta store. Defaults to None.
            streamed (bool, optional): Whether the response is streamed. Defaults to False.
            is_last (bool, optional): Whether this is the last chunk. Defaults to True.
        Returns:
            typing.List | None: The parsed list or None if not ready.
        """
        delta_store = delta_store if delta_store is not None else {}
        # resp = self.handle_null(resp, '')
        val = val or ''
        val = utils.acc(delta_store, 'val', val)
        res = val.split(self.sep)
        return_val = utils.UNDEFINED
        
        if len(res) > 0:
            if len(val) == 0:
                return_val = []
            elif val[-1] == self.sep:
                return_val = res[:-1]
                delta_store['val'] = ''
            elif is_last:
                return_val = res
                delta_store['val'] = ''
            else:
                return_val = res[:-1]
                delta_store['val'] = res[-1]
                
        return return_val

    def render(self, data) -> str:
        """Renders the data separated by lines

        Args:
            data: The data to render

        Returns:
            str: The rendered data
        """
        
        return f'{self.sep}'.join(data)


class LineParser(Parser):
    """
    Parses line by line. Can have a line continue by putting a backslash at the end of the line.
    Example:
        parser = LineParser()
        resp = Resp(msg=Msg(role='assistant', text='This is line 1\\\nThis is line 2\nThis is line 3\\\nThis is line 4'))
        delta_store = {}
        result = parser.forward(resp, delta_store, streamed=True, is_last=False)
        print(result)  # Output: ['This is line 1This is line 2']
    """
    sep: str = '\n'
    
    def forward(self, val: str | None):
        """Parse complete line-based response handling line continuations
        
        Args:
            val (str | None): The complete string to parse
            
        Returns:
            List: List of parsed lines with continuations handled
        """
        if val is None:
            return []
            
        if val == '':
            return []
            
        lines = val.splitlines()
        result = []
        buffer = []
        
        for line in lines:
            if not line:
                continue  # skip empty lines
                
            if line.endswith("\\"):
                buffer.append(line[:-1])
            else:
                buffer.append(line)
                result.append(''.join(buffer))
                buffer = []
        
        # Handle any remaining buffer content
        if buffer:
            result.append(''.join(buffer))
            
        return result

    def delta(
        self, 
        val: str | None, 
        delta_store: typing.Dict=None, 
        is_last: bool=True
    ) -> typing.List:
        """Parse the response into lines, handling line continuations with backslashes.

        Args:
            val (str | None): The string value to parse.
            delta_store (typing.Dict, optional): The delta store. Defaults to None.
            is_last (bool, optional): Whether this is the last chunk. Defaults to True.

        Returns:
            typing.List: The parsed lines.
        """
        delta_store = delta_store if delta_store is not None else {}
        val = val or ''
        utils.acc(delta_store, 'val', val)
        lines = delta_store['val'].splitlines()
        buffer = []

        if is_last and len(lines) == 0:
            return []
        final_ch = delta_store.get('val', '')[-1] if delta_store.get('val', '') else ''
        buffered_lines = []
        for i, line in enumerate(lines):

            if not line:
                continue  # skip empty lines if that's desired

            if line.endswith("\\"):
                buffer.append(line[:-1])
            else:

                buffer.append(line)
                buffered_lines.append(buffer)
                # logical_line = "".join(buffer)
                # logical_line = "\n".join(buffer)
                # result.append(logical_line)
                buffer = []

        if not is_last and len(buffered_lines) > 0:
            if final_ch == self.sep:
                buffered_lines[-1].append(self.sep)
            delta_store['val'] = ''.join(buffered_lines[-1])
            buffered_lines.pop(-1)

        if len(buffered_lines) == 0:
            return utils.UNDEFINED

        return [
            ''.join(line)
            for line in buffered_lines
        ]

    def render(self, data) -> str:
        """Render the data

        Args:
            data: The data to render

        Returns:
            str: The data rendered in lines
        """
        res = []
        for d in data:
            res.append(d.replace('\n', '\\n'))
        return f'\n'.join(data)
