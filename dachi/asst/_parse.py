# 1st party
import typing
from abc import ABC, abstractmethod
import typing
import csv
import io

# 3rd party

# local
from ..msg._messages import END_TOK, NULL_TOK
from .. import utils


class Parser(ABC):
    """
    Parser is an abstract base class designed to process and parse the output of a 
    large language model (LLM) into discrete units. These units can then be 
    converted into a structured output that is sent to the calling function.
    The class provides a framework for implementing custom parsing logic through 
    the `delta` and `template` methods, which must be defined in subclasses. It 
    also includes a convenience method `__call__` to handle parsing of the entire 
    response in one step.
    """
    @abstractmethod
    def delta(self, resp, delta_store: typing.Dict, is_last: bool=False) -> typing.List | None:
        """Parse the response one by one

        Args:
            resp: The response
            delta_store (typing.Dict): Dictionary to accumulate updates
            last (bool, optional): Whether it is the last value or not. Defaults to False.

        Returns:
            typing.List: Will return a list if value defined else UNDEFINED
        """
        pass

    def handle_null(self, resp, default):
        """
        Handles the case where the response (`resp`) is the `END_TOK`.
        Args:
            resp: The response to evaluate. If it matches `END_TOK`, the function
                  will return the default value and a flag indicating the end.
            default: The default value to return if `resp` is `END_TOK`.
        Returns:
            A tuple containing:
                - The response (`resp`) or the default value if `resp` is `END_TOK`.
                - A boolean flag indicating whether `resp` was `END_TOK` (True) or not (False).
        """

        if resp is NULL_TOK:
            return default
        return resp
    
    def stream(self, resp_iter: typing.Iterable, delta_store: typing.Dict=None, get_msg: bool=False) -> typing.Iterator:
        """
        Parses a stream of responses from the LLM and yields processed results.
        Args:
            resp_iter (typing.Iterable): An iterable of response tuples, where each tuple
                contains a key and a response object.
            delta_store (typing.Dict, optional): A dictionary to store intermediate state
                for delta processing. Defaults to an empty dictionary if not provided.
        Yields:
            typing.Iterator: Processed results from the responses, excluding undefined values.
        Notes:
            - The `delta` method is used to process each response and update the `delta_store`.
            - If the processed result is not `utils.UNDEFINED`, it is yielded.
            - After the iteration, a final call to `delta` is made with `END_TOK` to handle
              any remaining processing.
        """
        delta_store = delta_store or {}
        for msg, resp in resp_iter:
            cur = self.delta(resp, delta_store, False)
            if cur is not utils.UNDEFINED:
                if get_msg:
                    yield msg, cur
                else: yield cur
        cur = self.delta(NULL_TOK, delta_store, True)
        if cur is not utils.UNDEFINED:
            if get_msg:
                yield msg, cur
            else: yield cur

    async def astream(self, resp_iter: typing.AsyncIterable, delta_store: typing.Dict=None, get_msg: bool=False) -> typing.AsyncIterator:
        """
        Parses a stream of responses from the LLM and yields processed results.
        Args:
            resp_iter (typing.Iterable): An iterable of response tuples, where each tuple
                contains a key and a response object.
            delta_store (typing.Dict, optional): A dictionary to store intermediate state
                for delta processing. Defaults to an empty dictionary if not provided.
        Yields:
            typing.Iterator: Processed results from the responses, excluding undefined values.
        Notes:
            - The `delta` method is used to process each response and update the `delta_store`.
            - If the processed result is not `utils.UNDEFINED`, it is yielded.
            - After the iteration, a final call to `delta` is made with `END_TOK` to handle
              any remaining processing.
        """
        delta_store = delta_store or {}
        async for msg, resp in await resp_iter:
            cur = self.delta(resp, delta_store, False)
            if cur is not utils.UNDEFINED:
                if get_msg:
                    yield msg, cur
                else: yield cur
        cur = self.delta(NULL_TOK, delta_store, True)
        if cur is not utils.UNDEFINED:
            if get_msg:
                yield msg, cur
            else: yield cur

    def __call__(self, resp) -> typing.List[typing.Any] | None:
        """Convenience function to parse based on the whole set of data

        Args:
            resp: 

        Returns:
            typing.List[typing.Any]: The parsed values. Will return undefined if
            cannot parse
        """
        return self.delta(
            resp, {}, True
        )

    @abstractmethod
    def render(self, data) -> str:
        pass


class CSVRowParser(Parser):
    """
    Dynamically parse CSV data, returning new rows as accumulated. 
    The header will be returned along with them if used.
    """
    
    def __init__(self, delimiter: str = ',', use_header: bool = True):
        """
        Initializes the CSV parser with the specified delimiter and header usage.
        This class is designed to dynamically parse CSV data, returning new rows 
        as they are accumulated. It supports customization of the delimiter and 
        whether the CSV file includes a header row.
        Args:
            delimiter (str): The character used to separate values in the CSV. 
                             Defaults to ','.
            use_header (bool): Indicates whether the CSV file includes a header row. 
                               Defaults to False.
        """
        self._delimiter = delimiter
        self._use_header = use_header

    def delta(self, resp, delta_store: typing.Dict, is_last: bool=False) -> typing.List | None:
        """
        Parses CSV data incrementally using csv.reader.
        """
        resp = self.handle_null(resp, '')

        val = utils.add(delta_store, 'val', resp, '')
        row = utils.get_or_set(delta_store, 'row', 0)
        header = utils.get_or_set(
            delta_store, 'header', None
        )

        # Process accumulated data using csv.reader
        # csv_data = io.StringIO(delta_store['val'])

        rows = list(
            csv.reader(io.StringIO(val), delimiter=self._delimiter)
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
            self._use_header is True 
            and delta_store['header'] is None
        ):
            delta_store['header'] = new_rows.pop(0)
            utils.add(delta_store, 'row', 1)

        header = delta_store['header']
        utils.add(delta_store, 'row', len(new_rows))
        if len(new_rows) == 0:
            return utils.UNDEFINED
        
        if self._use_header:
            return [(header, row) for row in new_rows]
        return new_rows

    def render(self, data) -> str:
        pass


class CSVCellParser(Parser):
    """This parser assumes the input string is a CSV
    """

    def __init__(self, delimiter: str = ',', use_header: bool = True):
        """
        Initializes the converter for parsing CSV data.
        Args:
            delimiter (str): The character used to separate values in the CSV. Defaults to ','.
            use_header (bool): Indicates whether the CSV includes a header row. If True, 
                the header row will be used to label the cells. Defaults to False.
        This class parses CSV data and returns all cells. If `use_header` is True, 
        the cells will be labeled using the values from the header row.
        """
        self._delimiter = delimiter
        self._use_header = use_header

    def delta(self, resp, delta_store: typing.Dict, is_last: bool=False) -> typing.Any:
        """
        Parses a single-row CSV and returns one cell at a time.
        """
        resp = self.handle_null(resp, '')
        print('Resp: ', resp)

        val = utils.add(delta_store, 'val', resp)
        cur_row = utils.get_or_set(delta_store, 'row', 0)
        header = utils.get_or_set(delta_store, 'header', None)
        data = utils.get_or_set(delta_store, 'data', None)
        cur_col = utils.get_or_set(delta_store, 'col', 0)

        rows = list(csv.reader(io.StringIO(delta_store['val']), delimiter=self._delimiter))
        cells = []

        new_rows = []
        for i, row in enumerate(rows[cur_row:]):  
            # Only return new rows
            new_rows.append(row)

        if (
            len(new_rows) == 0 or 
            (len(new_rows) == 1 and not is_last)
        ):
            return utils.UNDEFINED

        if self._use_header and header is None:
            delta_store['row'] = 1
            delta_store['header'] = header = new_rows.pop(0)

        if len(new_rows) == 0 or (len(new_rows) == 1 and not is_last):
            return utils.UNDEFINED

        # if not is_last:
        #     new_rows[-1].pop(-1)

        cells = []

        for i, row in enumerate(new_rows):

            row = row[cur_col:] if i == 0 else row
            
            for j, cell in enumerate(row):
                if self._use_header:
                    cells.append((header[j], cell))
                else:
                    cells.append(cell)
        if not is_last and len(cells) > 0:
            cells.pop(-1)
        utils.add(delta_store, 'col', j)
        utils.add(delta_store, 'row', i)
        # else:
        #     delta_store['col'] = j
        #     delta_store['row'] = i
        
        if len(cells) == 0:
            return utils.UNDEFINED
        return cells
    
    def render(self, data) -> str:
        
        pass


class CharDelimParser(Parser):
    
    def __init__(self, sep: str=','):
        super().__init__()
        self.sep = sep

    def delta(
        self, resp, delta_store: typing.Dict, is_last: bool=False
    ) -> typing.List | None:
        
        resp = self.handle_null(resp, '')
        val = utils.add(delta_store, 'val', resp)
        res = val.split(self.sep)
        return_val = utils.UNDEFINED
        
        if len(res) > 0:
            if val[-1] == self.sep:
                return_val = res[:-1]
                delta_store['val'] = ''
            elif is_last:
                return_val = res
                delta_store['val'] = ''
            else:
                return_val = res[:-1]
                delta_store['val'] = res[-1]
                
        return return_val

    def render(self, data):
        
        return f'{self.sep}'.join(data)


class NullParser(Parser):
    """
    A parser that does not perform any parsing or transformation on the input.
    Instead, it simply returns the input response as-is.
    """

    def delta(self, resp, delta_store: typing.Dict, is_last: bool=False) -> typing.List:
        
        if resp is NULL_TOK:
            return utils.UNDEFINED
        
        return [resp]
    
    def render(self, data):
        pass


class FullParser(Parser):
    """
    A parser that accumulates input data until the end of a stream is reached.
    """
    
    def delta(self, resp, delta_store: typing.Dict, is_last: bool=False) -> typing.List:
        
        resp = self.handle_null(resp, '')
        utils.add(delta_store, 'val', resp)
        if is_last:
            val = delta_store['val']
            delta_store.clear()
            return [val]
        return utils.UNDEFINED

    def render(self, data):
        
        return str(data)


class LineParser(Parser):
    """
    A parser that accumulates input data until the end of a stream is reached.
    """
    def delta(
        self, resp, delta_store: typing.Dict, is_last: bool=False
    ) -> typing.List:
        """

        Args:
            resp : 
            delta_store (typing.Dict): 
            is_last (bool, optional): . Defaults to False.

        Returns:
            typing.Any: 
        """ 
        resp = self.handle_null(resp, '')
        utils.add(delta_store, 'val', resp)
        lines = delta_store['val'].splitlines()
        result = []
        buffer = []
        
        for i, line in enumerate(lines):

            if not line:
                continue  # skip empty lines if that's desired

            if line.endswith("\\"):
                buffer.append(line[:-1])
            else:
                buffer.append(line)
                logical_line = "\n".join(buffer)
                result.append(logical_line)
                buffer = []

        if buffer:
            result.append("\n".join(buffer))
        
        if not is_last and len(result) > 0:
            delta_store['val'] = result[-1]
            if resp[-1] == '\n':
                delta_store['val'] += '\n'
            result = result[:-1]
        elif is_last:
            delta_store['val'] = ''
        if len(result) == 0:
            return utils.UNDEFINED

        return result

    def render(self, data):
        
        res = []
        for d in data:
            res.append(d.replace('\n', '\\n'))
        return f'\n'.join(data)
