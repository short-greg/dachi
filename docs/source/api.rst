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
   dachi.async_module
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


zenkai.data
-----------

.. autosummary::
   :toctree: generated

   zenkai.data.Term
   zenkai.data.Glossary
   zenkai.data.Context
   zenkai.data.ContextStorage
   zenkai.data.Shared
   zenkai.data.get_or_set
   zenkai.data.get_or_spawn
   zenkai.data.SharedBase
   zenkai.data.Buffer
   zenkai.data.BufferIter
   zenkai.data.ContextSpawner
   zenkai.data.Media
   zenkai.data.Message
   zenkai.data.DataList
   zenkai.data.MediaMessage


zenkai.op
---------

.. autosummary::
   :toctree: generated

   zenkai.op.Description
   zenkai.op.Ref
   zenkai.op.bullet
   zenkai.op.formatted
   zenkai.op.generate_numbered_list
   zenkai.op.numbered
   zenkai.op.validate_out
   zenkai.op.fill
   zenkai.op.head
   zenkai.op.section
   zenkai.op.cat
   zenkai.op.join
   zenkai.op.Op
   zenkai.op.op
   zenkai.op.bold
   zenkai.op.strike
   zenkai.op.italic

zenkai.read
-----------

.. autosummary::
   :toctree: generated

   zenkai.read.CSVRead
   zenkai.read.KVRead
   zenkai.read.StructListRead
   zenkai.read.JSONRead

zenkai.utils
------------

.. autosummary::
   :toctree: generated

   zenkai.utils.get_str_variables
   zenkai.utils.escape_curly_braces
   zenkai.utils.is_primitive
   zenkai.utils.generic_class
   zenkai.utils.str_formatter
   zenkai.utils.is_nested_model
   zenkai.utils.is_undefined
   zenkai.utils.UNDEFINED
   zenkai.utils.WAITING
   zenkai.utils.Renderable
   zenkai.utils.model_template
   zenkai.utils.struct_template
   zenkai.utils.model_to_text
   zenkai.utils.model_from_text
   zenkai.utils.StructLoadException
   zenkai.utils.Templatable
   zenkai.utils.TemplateField
