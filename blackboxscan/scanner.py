from transformers import AutoModelForCausalLM

class GenerativeModelOutputs:
    def __init__(self, model: AutoModelForCausalLM, inputs: list|str):
        self.model = model
        if isinstance(inputs, str):
            self.inputs = [inputs]
        else:
            self.inputs = inputs

    def log_likelihoods(self, inputs: list|str, words: list|str):
        if isinstance(words, str):
            words = [words]
        pass

    def perplexity(self):
        pass

    def view_topk(self):
        pass

class TokenizerOutputs:
    pass

class EncoderModelOutputs:
    pass