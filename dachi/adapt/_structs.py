
# class Lookup(StructModule, ABC):

#     score: float
#     content: typing.Dict[str, typing.Any]
#     meta: typing.Dict[str, typing.Any]

#     @abstractmethod
#     def forward(self, key: str, value: typing.Any = ..., get_struct: bool = False) -> typing.Any:
#         return super().forward(key, value, get_struct)


# class LookupList(StructList[Lookup]):

#     @property
#     def lookups(self) -> typing.List[Lookup]:
#         return self.structs

#     def filter(self, threshold: float):

#         return LookupList(
#             lookup for lookup in self.lookups 
#             if lookup.score >= threshold
#         )

