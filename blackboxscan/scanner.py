from collections import defaultdict
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch


class GenerativeModelOutputs:
    def __init__(
        self, model: AutoModelForCausalLM, tokenizer: AutoTokenizer, inputs: list | str
    ):
        self.model = model
        self.tokenizer = tokenizer
        if isinstance(inputs, str):
            self.inputs = inputs.split(" ")
        else:
            self.inputs = inputs

    def log_likelihoods(self, words: list | str):
        otpt = defaultdict()
        if isinstance(words, str):
            words = words.split(" ")
        if self.tokenizer.bos_token:
            bos_tok = self.tokenizer.bos_token
        else:
            bos_tok = " "
        bos_tok = [bos_tok]
        full_input = bos_tok + self.inputs + words
        for i in full_input:
            sentence = str(i)
            input_ids = self.tokenizer.encode(sentence, return_tensors="pt")
            with torch.no_grad():
                outputs = self.model(
                    input_ids, labels=input_ids
                )  # No need to provide labels during inference
                logits = outputs.logits

            # Calculate the negative log likelihood for each token
            neg_log_likelihood = torch.nn.CrossEntropyLoss(reduction="none")(
                logits[:, :-1].contiguous().view(-1, logits.size(-1)),
                input_ids[:, 1:].contiguous().view(-1),
            )

            # Reshape the neg_log_likelihood tensor to match the original input shape
            neg_log_likelihood = neg_log_likelihood.view(input_ids[:, 1:].size())

            # Output the negative log likelihood for each token
            sent = 0
            for k in range(
                neg_log_likelihood.size(1)
            ):  # Iterate over the length of neg_log_likelihood
                # token = self.tokenizer.decode(
                # input_ids[0, k + 1]
                # )  # Decode the token (skipping the first token [CLS])
                nll_token = -neg_log_likelihood[0, k]  # Negate the value
                sent += nll_token
            if i in words:
                # Add the total negative log likelihood to the list
                if isinstance(sent, torch.Tensor):
                    sent = sent.item()
                otpt[i] = sent
        return otpt

    # with torch.no_grad():
    # outputs = model(input_ids, labels=input_ids)  # No need to provide labels during inference
    # logits = outputs.logits

    # # Calculate the negative log likelihood for each token
    # neg_log_likelihood = torch.nn.CrossEntropyLoss(reduction='none')(logits[:, :-1].contiguous().view(-1, logits.size(-1)),
    #                                                                 input_ids[:, 1:].contiguous().view(-1))

    # # Reshape the neg_log_likelihood tensor to match the original input shape
    # neg_log_likelihood = neg_log_likelihood.view(input_ids[:, 1:].size())

    # # Output the negative log likelihood for each token
    # sent = 0
    # for i in range(input_ids.size(1)):  # Exclude the last token as it's not used for labels
    #     token = tokenizer.decode(input_ids[0, i])
    #     nll_token = -neg_log_likelihood[0, i-1]  # Negate the value
    #     sent += nll_token

    def perplexity(self):
        pass

    def view_topk(self):
        pass


class EmbeddingOutputs:
    pass


class EncoderModelOutputs:
    pass
