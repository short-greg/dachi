import typing

from .. import proc as core

# TODO: COMPLETE


# class Turn(core.Module):

#     def __init__(
#         self, model: core.ChatModel, 
#         message: core.Message, 
#         parent: 'Turn'=None
#     ):
#         super().__init__()
#         self.model = model
#         self.parent = parent
#         self.message = message

#     def __getitem__(self, idx: int) -> core.Message:

#         return self.dialog[idx]
    
#     def __setitem__(self, idx: int, message: core.Message) -> core.Message:

#         self.dialog[idx] = message
#         return message

#     def __iter__(self) -> typing.Iterator[core.Message]:

#         for m in self.dialog:
#             return m
        
#     @property
#     def messages(self) -> typing.Iterator[core.Message]:
        
#         for message in self.parent.messages:
#             yield message
#         yield message

#     @property
#     def dialog(self) -> core.Dialog:
#         return core.Dialog(
#             self.messages
#         )

#     def forward(self, text: str='', source: str='user', **data):
#         message = core.Message(source=source, text=text, data=data)
#         turn = Turn()
#         response = self.model(self.dialog.to_messages())
#         cue: core.Cue = self.dialog.cue()
#         self.dialog.append(response)
#         return cue.read(response)
    
#     async def aforward(self, text: str='', source: str='user', **data):
#         message = core.Message(source=source, text=text, data=data)
        
#         response = self.model(self.dialog.to_messages())
#         cue: core.Cue = self.dialog.cue()
#         self.dialog.append(response)
#         return cue.read(response)
    
#     def stream(self, text: str='', source: str='user', **data):
#         message = core.Message(source=source, text=text, data=data)
        
#         response = ''
#         cue: core.Cue = self.dialog.cue()

#         # 1) the cue can be streamed
#         # 2) the cue cannot be streamed
#         # 3) the cue can partially be streamed
#         # 4) the cue doesn't exist

#         # 5) Returns a tool
#         # 6) 

#         # This must tra
#         tokens = []
#         for token in self.model.stream(self.dialog.to_messages()):

#             if token.is_tool():
#                 if token.is_complete():
#                     pass
#             elif token.is_end():
#                 pass
#             elif cue is None:
#                 pass
#             elif cue.multi:
#                 pass
#             else:
#                 try:
#                     cue.read(dx)
#                 except cue.ReadError:
#                     # if the read error fails here assume 
#                     # it is not ready
#                     yield dx
        
#         self.dialog.append(response)
#         return cue.read(response)
    
#     async def astream(self, text: str='', source: str='user', **data):
#         pass
