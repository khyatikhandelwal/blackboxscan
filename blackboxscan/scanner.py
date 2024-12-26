import sys
from typing import Optional
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import argparse
from collections import defaultdict


class ScannedOutputs:
    def __init__(self):
        self.likelihoods_tokenwise = []
        self.embeddings = defaultdict()

    def add_token_output(self, token, output):
        self.likelihoods_tokenwise.append((token, output))

    def get_total(self):
        return sum([i[1] for i in self.likelihoods_tokenwise])

    def get_tokens(self):
        return self.likelihoods_tokenwise

    def get_perplexity(self):
        return self.get_total / len(self.likelihoods_tokenwise)

    def get_top_tokens(self, greedy_top_tokens, greedy_score_list):
        return {
            greedy_top_tokens[i]: greedy_score_list[i]
            for i in range(len(greedy_top_tokens))
        }

    def word_embeddings(self, word_embeds):
        for k, v in word_embeds.items():
            self.embeddings[k] = v
        return self.embeddings


class GenerativeModelOutputs:
    def __init__(self, model: AutoModelForCausalLM, tokenizer: AutoTokenizer):
        self.model = model
        self.tokenizer = tokenizer

    def sentence_log_likelihoods(self, words: list | str) -> ScannedOutputs:
        """
        Outputs the likelihoods of the next tokens predicted by the models.
        words: context of words given to the model.
        """
        output = ScannedOutputs()
        input_ids = self.tokenizer.encode(words, return_tensors="pt")
        with torch.no_grad():
            outputs = self.model(
                input_ids, labels=input_ids
            )  # No need to provide labels during inference
            logits = outputs.logits

        # Calculate the negative log likelihood for each token
        neg_log_likelihood = torch.nn.NLLLoss(reduction="none")(
            logits[:, :-1].contiguous().view(-1, logits.size(-1)),
            input_ids[:, 1:].contiguous().view(-1),
        )

        # Reshape the neg_log_likelihood tensor to match the original input shape
        neg_log_likelihood = neg_log_likelihood.view(input_ids[:, 1:].size())

        # Output the negative log likelihood for each token
        sent = 0
        for k in range(neg_log_likelihood.size(1)):
            token = self.tokenizer.decode(input_ids[0, k + 1])
            nll_token = -neg_log_likelihood[0, k]  # Negate the value
            if isinstance(nll_token, torch.Tensor):
                nll_token = nll_token.item()
            output.add_token_output(token=token, output=nll_token)
            sent += nll_token
        return output

    def view_topk(
        self, input_sentence: str, k: int, get_plot: Optional[bool] = False
    ) -> ScannedOutputs:
        """
        https://medium.com/@meoungjun.k/analyzing-token-scores-in-text-generation-with-hugging-face-transformers-c2b3d5b2bece
        """
        inputs = self.tokenizer(input_sentence, return_tensors="pt")
        greedy_short_outputs = self.model.generate(
            **inputs,
            max_new_tokens=1,
            top_k=4,
            return_dict_in_generate=True,
            output_scores=True,
        )
        greedy_score = greedy_short_outputs.scores[0]
        greedy_score_list = greedy_score.topk(k, dim=1)[0].tolist()[0]
        greedy_top_tokens = self.tokenizer.batch_decode(
            greedy_score.topk(k, dim=1).indices
        )[0].split()
        if get_plot:
            self.plot_topk(scores=greedy_score_list, tokens=greedy_top_tokens)
        dict = ScannedOutputs()
        return dict.get_top_tokens(greedy_top_tokens, greedy_score_list)

    def plot_topk(self, scores: list[float], tokens: list[str]) -> None:
        """
        To display an interactive plot of the top k output tokens using Plotly.
        """
        # Original data
        data = scores

        # Calculate the minimum value in the list
        min_value = min(data)

        # Apply the transformation: val - min(list)
        transformed_data = [x - min_value for x in data]

        # Create the bar chart using Plotly
        fig = go.Figure(
            data=[
                go.Bar(
                    x=tokens,
                    y=transformed_data,
                    marker_color="skyblue",
                    text=transformed_data,
                    textposition="auto",
                )
            ]
        )

        # Update layout for better visualization
        fig.update_layout(
            title="Histogram of top k tokens with Transformation: val - min(list)",
            xaxis_title="Tokens",
            yaxis_title="Transformed Likelihood Score (curr_val - min(top k vals))",
            xaxis_tickangle=90,
            template="plotly_white",
        )

        # Show the plot inline
        fig.show()


class EmbeddingOutputs:
    """
    This class helps visualise, and analyze word embeddings of BERT-like encoder models.
    """

    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer

    def get_embeddings_output(self, words_list: list):
        # Input word
        dic = defaultdict()

        for word in words_list:
            tokens = tokenizer.tokenize(word)
            input_ids = tokenizer.convert_tokens_to_ids(tokens)
            input_ids = torch.tensor(input_ids).unsqueeze(0)

            # Forward pass through the model
            outputs = model(input_ids)

            # Extract the word embedding from the last layer
            last_layer_embedding = outputs.last_hidden_state.squeeze(0)
            last_layer_embedding = normalize(
                last_layer_embedding.detach().numpy(), norm="l2", axis=1
            )

            dic[word] = last_layer_embedding

        final_output = ScannedOutputs()
        final_output = final_output.word_embeddings(dic)
        return final_output

    def visualise_embeddings(self, embed_dict: ScannedOutputs):
        pass


class EncoderModelOutputs:
    """
    This class will help to easily visualise the attention mechanism of encoder models.
    """

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
    # main()
    model_name = "gpt2"
    tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side="left")
    model = AutoModelForCausalLM.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    s = GenerativeModelOutputs(model=model, tokenizer=tokenizer)
    d = s.view_topk(input_sentence="Hello! I am an", k=10, get_plot=True)
    print(d)
