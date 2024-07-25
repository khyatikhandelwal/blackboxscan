import numpy as np
import pandas as pd
import re
import ast
import json
import torch
import difflib
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
from huggingface_hub.hf_api import HfFolder

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