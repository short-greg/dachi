from ._data import (
    Description, Ref   
)
from ._instruct import (
    bullet, formatted, generate_numbered_list,
    numbered, validate_out, fill, head,
    section, cat, join, Inst, inst,
    bold, strike, italic
)
from ._instruct_core import (
    Instruct,
    Cue,
    validate_out,
    IBase,
    InstF,
    SigF,
    FuncDec,
    FuncDecBase,
    AFuncDec,
    StreamDec,
    AStreamDec,
    instructfunc,
    instructmethod,
    signaturefunc,
    signaturemethod
)
