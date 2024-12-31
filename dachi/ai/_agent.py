from .. import _core as core


class LLMAgent(core.Module):

    def forward(self, *args, **kwargs):
        return super().forward(*args, **kwargs)
    
    def aforward(self, *args, **kwargs):
        return super().aforward(*args, **kwargs)
    
    def stream(self, *args, **kwargs):
        return super().stream(*args, **kwargs)
    
    def astream(self, *args, **kwargs):
        return super().astream(*args, **kwargs)
