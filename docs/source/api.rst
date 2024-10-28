.. _api:


API Reference
=============

dachi
-----

.. autosummary::
   :toctree: generated

   dachi.Reader
   dachi.Storable
   dachi.render
   dachi.render_multi
   dachi.Module
   dachi.Cue
   dachi.Param
   dachi.MultiRead
   dachi.PrimRead
   dachi.PydanticRead
   dachi.AIModel
   dachi.AIPrompt
   dachi.AIResponse
   dachi.Dialog
   dachi.Message
   dachi.TextMessage
   dachi.Data
   dachi.stream_text
   dachi.Partial
   dachi.ParallelModule
   dachi.parallel_loop
   dachi.processf
   dachi.MultiModule
   dachi.ModuleList
   dachi.Sequential
   dachi.Batched
   dachi.Streamer
   dachi.AsyncModule
   dachi.reduce
   dachi.I
   dachi.P
   dachi.async_map
   dachi.run_thread
   dachi.Runner
   dachi.RunStatus
   dachi.StreamRunner
   dachi.validate_out
   dachi.InstructCall
   dachi.SignatureFunc
   dachi.signaturefunc
   dachi.signaturemethod
   dachi.InstructFunc
   dachi.instructmethod
   dachi.instructfunc


dachi.data
-----------

.. autosummary::
   :toctree: generated

   dachi.data.Term
   dachi.data.Glossary
   dachi.data.Context
   dachi.data.ContextStorage
   dachi.data.Shared
   dachi.data.get_or_set
   dachi.data.get_or_spawn
   dachi.data.SharedBase
   dachi.data.Buffer
   dachi.data.BufferIter
   dachi.data.ContextSpawner
   dachi.data.Media
   dachi.data.Message
   dachi.data.DataList
   dachi.data.MediaMessage


dachi.op
---------

.. autosummary::
   :toctree: generated

   dachi.op.Description
   dachi.op.Ref
   dachi.op.bullet
   dachi.op.formatted
   dachi.op.generate_numbered_list
   dachi.op.numbered
   dachi.op.validate_out
   dachi.op.fill
   dachi.op.head
   dachi.op.section
   dachi.op.cat
   dachi.op.join
   dachi.op.Op
   dachi.op.op
   dachi.op.bold
   dachi.op.strike
   dachi.op.italic

dachi.read
-----------

.. autosummary::
   :toctree: generated

   dachi.read.CSVRead
   dachi.read.KVRead
   dachi.read.StructListRead
   dachi.read.JSONRead

dachi.utils
------------

.. autosummary::
   :toctree: generated

   dachi.utils.get_str_variables
   dachi.utils.escape_curly_braces
   dachi.utils.is_primitive
   dachi.utils.generic_class
   dachi.utils.str_formatter
   dachi.utils.is_nested_model
   dachi.utils.is_undefined
   dachi.utils.UNDEFINED
   dachi.utils.WAITING
   dachi.utils.Renderable
   dachi.utils.model_template
   dachi.utils.struct_template
   dachi.utils.model_to_text
   dachi.utils.model_from_text
   dachi.utils.StructLoadException
   dachi.utils.Templatable
   dachi.utils.TemplateField
