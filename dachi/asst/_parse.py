# 1st party
import typing
import csv
import io
from abc import abstractmethod

# 3rd party

# local
from .. import store
from .. import utils
from ._msg import MsgProc
from collections import OrderedDict
from ..proc import Module


class Parser(Module):
    """Base class for parsers. It converts the input text
    into a list of objects
    """

    def __init__(self, name, from_: str | typing.List[str]='content'):
        """Create the parser

        Args:
            name: The name of the parser
            from_ (str | typing.List[str], optional): The input for the parser. Defaults to 'data'.
        """
        super().__init__(name, from_)

    def forward(self, resp, delta_store: typing.Dict, streamed: bool=False, is_last: bool=False) -> typing.List | None:
        pass

    @abstractmethod
    def render(self, data) -> str:
        pass


class CSVRowParser(Parser):
    """
    Dynamically parse CSV data, returning new rows as accumulated. 
    The header will be returned along with them if used.
    """
    
    def __init__(self, name: str, from_: str ='content', delimiter: str = ',', use_header: bool = True):
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
        super().__init__(name, from_)
        self._delimiter = delimiter
        self._use_header = use_header

    def forward(self, resp, delta_store: typing.Dict, streamed: bool=False, is_last: bool=False) -> typing.List | None:
        """
        Parses CSV data incrementally using csv.reader.
        """
        # resp = self.handle_null(resp, '')

        val = store.acc(delta_store, 'val', resp, '')
        row = store.get_or_set(delta_store, 'row', 0)
        header = store.get_or_set(
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
            store.acc(delta_store, 'row', 1)

        header = delta_store['header']
        store.acc(delta_store, 'row', len(new_rows))
        if len(new_rows) == 0:
            return utils.UNDEFINED
        
        if self._use_header:
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
        writer = csv.writer(output, delimiter=self._delimiter)
        
        if self._use_header:
            header = [key for key, _ in data[0]]
            writer.writerow(header)
            for row in data:
                writer.writerow([value for _, value in row])
        else:
            for row in data:
                writer.writerow(row)
        
        return output.getvalue()


class CharDelimParser(Parser):
    """Parses based on a defined character
    """
    def __init__(self, name: str, from_: str='content', sep: str=','):
        super().__init__(name, from_)
        self.sep = sep

    def forward(
        self, resp, delta_store: typing.Dict, streamed: bool=False, is_last: bool=False
    ) -> typing.List | None:
        
        # resp = self.handle_null(resp, '')
        resp = resp or ''
        val = store.acc(delta_store, 'val', resp)
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
    """
    def __init__(self, name, from_ = 'content'):
        super().__init__(name, from_)

    def forward(
        self, resp, delta_store: typing.Dict, streamed: bool=False, is_last: bool=False
    ) -> typing.List:
        """

        Args:
            resp : 
            delta_store (typing.Dict): 
            is_last (bool, optional): . Defaults to False.

        Returns:
            typing.Any: 
        """ 
        resp = resp or ''
        store.acc(delta_store, 'val', resp)
        lines = delta_store['val'].splitlines()
        result = []
        buffer = []

        if is_last and len(lines) == 0:
            return []

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


class CSVCellParser(MsgProc):
    """This parser assumes the input string is a CSV
    """

    def __init__(self, name: str, from_: str='content', delimiter: str = ',', use_header: bool = True):
        """
        Initializes the converter for parsing CSV data.
        Args:
            delimiter (str): The character used to separate values in the CSV. Defaults to ','.
            use_header (bool): Indicates whether the CSV includes a header row. If True, 
                the header row will be used to label the cells. Defaults to False.
        This class parses CSV data and returns all cells. If `use_header` is True, 
        the cells will be labeled using the values from the header row.
        """
        super().__init__(name, from_)
        self._delimiter = delimiter
        self._use_header = use_header

    def forward(self, resp, delta_store: typing.Dict, streamed: bool=False, is_last: bool=False) -> typing.Any:
        """
        Parses a single-row CSV and returns one cell at a time.
        """
        
        # resp = self.handle_null(resp, '')
        # print('Resp: ', resp)

        resp = resp or ''
        val = store.acc(delta_store, 'val', resp)
        cur_row = store.get_or_set(delta_store, 'row', 0)
        header = store.get_or_set(delta_store, 'header', None)
        data = store.get_or_set(delta_store, 'data', None)
        cur_col = store.get_or_set(delta_store, 'col', 0)

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
            sub_val = 0
        elif self._use_header:
            sub_val = 1
            
        if len(new_rows) == 0 or (len(new_rows) == 1 and not is_last):
            return utils.UNDEFINED

        # if not is_last:
        #     new_rows[-1].pop(-1)

        cells = []

        for i, row in enumerate(new_rows):

            row = row[cur_col:] if i == 0 else row
            
            for j, cell in enumerate(row):
                if self._use_header:
                    cells.append((cur_row + i - sub_val, header[j], cell))
                else:
                    cells.append((cur_row + i, cell))
        if not is_last and len(cells) > 0:
            cells.pop(-1)
        store.acc(delta_store, 'col', j)
        store.acc(delta_store, 'row', i)
        
        if len(cells) == 0:
            return utils.UNDEFINED
        return cells
    
    def render(self, data) -> str:
        
        output = io.StringIO()
        writer = csv.writer(output, delimiter=self._delimiter)

        if self._use_header:
            # Extract the header from the first row of data
            header = [cell[1] for cell in data if cell[0] == 0]
            writer.writerow(header)

            # Group cells by row number and write rows
            rows = {}
            for row_num, col_name, value in data:
                if row_num not in rows:
                    rows[row_num] = {}
                rows[row_num][col_name] = value

            for row_num in sorted(rows.keys()):
                row = [rows[row_num].get(col, '') for col in header]
                writer.writerow(row)
        else:
            # Group cells by row number and write rows
            rows = {}
            for row_num, value in data:
                if row_num not in rows:
                    rows[row_num] = []
                rows[row_num].append(value)

            for row_num in sorted(rows.keys()):
                writer.writerow(rows[row_num])

        return output.getvalue()



# class NullParser(ParseConv):
#     """
#     A parser that does not perform any parsing or transformation on the input.
#     Instead, it simply returns the input response as-is.
#     """

#     def delta(self, resp, delta_store: typing.Dict, streamed: bool=False, is_last: bool=False) -> typing.List:
#         """

#         Args:
#             resp: The response
#             delta_store (typing.Dict): The dictionary
#             streamed (bool, optional): Whether the response is streamed. Defaults to False.
#             is_last (bool, optional): Whether it is the last. Defaults to False.

#         Returns:
#             typing.List: 
#         """
#         resp = resp or ''
#         return [resp]
    
#     def render(self, data):
#         return str(data)

