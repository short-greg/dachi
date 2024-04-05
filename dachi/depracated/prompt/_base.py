from abc import ABC

class Component(object):

    def style(self) -> str:
        pass

    def __call__(self):
        pass


class Styling(object):

    @classmethod
    def style(cls, text, style):
        pass


class Objective(Component):

    def __init__(self, desc: str, style):

        # Your objective is as follows
        # Objective: ...
        
        self.desc = desc
        self.style = style

    def __call__(self):
        
        return Styling.style(self.desc, self.style)


class Role(Component):

    def __init__(self, val: str, style):

        # Your role is as follows.
        # Role: ...

        # # Description
        # # Key
        # # Text
        # # # Show the description
        # # # kdv = '{Key}: {Description} - {Value}'
        # # # style='*{Key}*: {Description} - {Value}'
        # # # KDV / KV / V <- can use any of these
        # # # style(desc=desc, key=key, value=value)
        # # # style.key='Objective'
        # # # style.desc='Here is your objective to achieve'
        # # # style.modifier='
        # # # bold / itallic
        # # 1) Words used
        # # 2) KeyValDesc(key)
        # # 3) KeyVal('role', )
        # # 4) List([], style=style)
        # #    Role(KeyValDesc)

        # # CSVIOTemp(['x', 'y', 'z'], out=True, style=...)
        # # csv_temp.example_list([{}, {}])
        # # csv_temp.example([{}, {}])
        # # ConvExample()
        # # 
        # # KVIOTemp('Output')
        # # ListIOTemp(')
        
        self.val = val
        self.style = style

    def __call__(self):
        
        return self.style(desc=self.desc)

