import sys
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import argparse


class ScannedOutputs:
    def __init__(self):
        self.likelihoods_tokenwise = []

    def add_token_output(self, token, output):
        self.likelihoods_tokenwise.append((token, output))

    def get_total(self):
        return sum([i[1] for i in self.likelihoods_tokenwise])

    def get_tokens(self):
        return self.likelihoods_tokenwise


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

    def sentence_log_likelihoods(self, words: list | str):
        output = ScannedOutputs()
        if isinstance(words, str):
            words = words.split(" ")
        full_input = self.inputs + words
        i = " ".join(full_input)
        sentence = str(i)
        print(sentence)
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
        ):  # Exclude the last token as it's not used for labels
            token = self.tokenizer.decode(input_ids[0, k + 1])
            nll_token = -neg_log_likelihood[0, k]  # Negate the value
            if isinstance(nll_token, torch.Tensor):
                nll_token = nll_token.item()
            output.add_token_output(token=token, output=nll_token)
            sent += nll_token
        return output

    def word_log_likelihoods(self, word: str):
        # if self.tokenizer.bos_token:
        #     bos_tok = self.tokenizer.bos_token
        # else:
        #     bos_tok = " "
        pass

    def perplexity(self):
        pass

    def view_topk(self):
        pass


class EmbeddingOutputs:
    pass


class EncoderModelOutputs:
    pass


def main():
    # Setup argument parsing
    parser = argparse.ArgumentParser(description="Analyze outputs of HuggingFace LLMs.")
    parser.add_argument(
        "--model", type=str, required=True, help="HuggingFace model identifier"
    )
    parser.add_argument(
        "--input", type=str, required=True, help="Input text to analyze"
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=["sentence", "word"],
        required=True,
        help="Mode of analysis",
    )
    parser.add_argument(
        "--words", type=str, nargs="+", help="Words to calculate log likelihood for"
    )

    args = parser.parse_args()

    # Load model and tokenizer
    model = AutoModelForCausalLM.from_pretrained(args.model)
    tokenizer = AutoTokenizer.from_pretrained(args.model)

    # Instantiate the GenerativeModelOutputs class
    generative_outputs = GenerativeModelOutputs(model, tokenizer, args.input)

    if args.mode == "sentence" and args.words:
        output = generative_outputs.sentence_log_likelihoods(args.words)
        print(f"Total Sentence Log Likelihood: {output.get_total()}")
        print(f"Token-wise Likelihoods: {output.get_tokens()}")
    elif args.mode == "word":
        # Implement word-level log likelihood analysis if needed
        pass
    else:
        print("Invalid mode or missing required arguments for the selected mode.")
        sys.exit(1)


if __name__ == "__main__":
    main()
