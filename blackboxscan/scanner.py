import sys
from typing import Optional
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import argparse
import uuid
from collections import defaultdict
from transformers import AutoModel
from typing import Dict
import matplotlib.pyplot as plt
from sklearn.preprocessing import normalize
import plotly.graph_objects as go
import os


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
        return self.get_total() / len(self.likelihoods_tokenwise)

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
    ) -> dict:
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

        temp_output = ScannedOutputs()

        if get_plot:
            image_url = self.plot_topk(
                scores=greedy_score_list, tokens=greedy_top_tokens
            )
            return {
                "topk_tokens": temp_output.get_top_tokens(
                    greedy_top_tokens, greedy_score_list
                ),
                "plot_url": image_url,
            }
        else:
            return temp_output.get_top_tokens(greedy_top_tokens, greedy_score_list)

    def plot_topk(self, scores: list[float], tokens: list[str]) -> str:
        """
        To display an interactive plot of the top k output tokens using Plotly.
        Also saves the plot to a file and returns the public URL.
        """
        data = scores
        min_value = min(data)
        transformed_data = [x - min_value for x in data]

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

        fig.update_layout(
            title="Histogram of top k tokens with Transformation: val - min(list)",
            xaxis_title="Tokens",
            yaxis_title="Transformed Likelihood Score (curr_val - min(top k vals))",
            xaxis_tickangle=90,
            template="plotly_white",
        )

        # Save to static folder
        os.makedirs("static", exist_ok=True)
        filename = f"{uuid.uuid4().hex}.png"
        filepath = os.path.join("static", filename)
        fig.write_image(filepath)

        # Return public URL (assumes /static/ is served by Flask)
        return f"/static/{filename}"


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
            tokens = self.tokenizer.tokenize(word)
            input_ids = self.tokenizer.convert_tokens_to_ids(tokens)
            input_ids = torch.tensor(input_ids).unsqueeze(0)

            # Forward pass through the model
            outputs = self.model(input_ids)

            # Extract the word embedding from the last layer
            last_layer_embedding = outputs.last_hidden_state.squeeze(0)
            last_layer_embedding = normalize(
                last_layer_embedding.detach().numpy(), norm="l2", axis=1
            )

            dic[word] = last_layer_embedding

        final_output = ScannedOutputs()
        final_output = final_output.word_embeddings(dic)
        return final_output

    def visualise_embeddings(self, embed_dict: ScannedOutputs, xword: str, yword: str):
        """
        This is a function which will take 2 words ('xword' and 'yword'), as well as the embeddings dictionary from above.
        Then, it will visualise a scatter plot showing position of different words in the embeddings dictionary WRT the
        two xwords and ywords.
        This helps us visualise the closeness of different words with the other in the mebedding space.
        """
        pass


class Attention:
    """
    Visualize attention mechanisms of encoder models like BERT, or open-sourced decoder models like GPT 2.
    Supports both vanilla and fine-tuned models.
    This has been tested with Huggingface Models.
    """

    def __init__(self, model_name: str = "gpt2", gen: bool = True):
        """
        Initialize the encoder model and tokenizer.

        :param model_name: Hugging Face model identifier
        """
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        self.model.eval()  # Set to evaluation mode
        self.gen = gen  # For generative models (as opposed to encoder bert-like models)

    def attention_scores(self, sentence: str) -> Dict[str, float]:
        """
        Extract attention scores for each token in the sentence.

        :param sentence: Input sentence to analyze
        :return: Dictionary mapping tokens to their attention scores
        """
        if self.gen:
            # Tokenize
            inputs = self.tokenizer(
                sentence, return_tensors="pt", add_special_tokens=True
            )

            # Forward pass
            with torch.no_grad():
                outputs = self.model(**inputs, output_attentions=True)

            last_layer_attentions = outputs.attentions[-1].squeeze(
                0
            )  # Shape: (num_heads, seq_len, seq_len)

            avg_attentions = torch.mean(
                last_layer_attentions, dim=0
            )  # Shape: (seq_len, seq_len)

            # Get tokens and their corresponding attention scores
            tokens = self.tokenizer.convert_ids_to_tokens(inputs.input_ids[0])
            tokens = [i.replace("Ġ", "") for i in tokens]

            token_attentions = avg_attentions.mean(dim=0)  # Shape: (seq_len,)

            attention_dict = {
                token: score.item() for token, score in zip(tokens, token_attentions)
            }

            return attention_dict
        else:
            # Tokenize
            inputs = self.tokenizer(
                sentence, return_tensors="pt", add_special_tokens=True
            )

            # Forward pass
            with torch.no_grad():
                outputs = self.model(**inputs, output_attentions=True)

            last_layer_attentions = outputs.attentions[-1].squeeze(0)

            avg_attentions = torch.mean(last_layer_attentions, dim=0)

            # Get tokens and their corresponding attention scores
            tokens = self.tokenizer.convert_ids_to_tokens(inputs.input_ids[0])
            # Removing the whitespace token for better viz
            tokens = [i.replace("Ġ", "") for i in tokens]

            token_attentions = avg_attentions.mean(dim=0)

            attention_dict = {
                token: score.item() for token, score in zip(tokens, token_attentions)
            }

            return attention_dict

    def view_attention(self, sentence: str, graph: bool = True):
        """
        Visualize attention scores with a simple print representation.

        :param sentence: Input sentence to analyze
        """
        attention_scores = self.attention_scores(sentence)

        print("Attention Scores:")
        for token, score in sorted(
            attention_scores.items(), key=lambda x: x[1], reverse=True
        ):
            print(f"{token}: {score:.4f}")

        if graph:
            x, y = [], []
            for token, score in sorted(
                attention_scores.items(), key=lambda x: x[1], reverse=True
            ):
                x.append(token)
                y.append(round(score, 4))
            # Plot the attention scores
            plt.figure(figsize=(10, 5))
            plt.bar(x, y, color="skyblue")
            plt.xlabel("Tokens")
            plt.ylabel("Attention Score")
            plt.title("Token Attention Scores")
            plt.xticks(rotation=45, ha="right")
            plt.grid(axis="y", linestyle="--", alpha=0.7)

            # Save the plot
            os.makedirs("static", exist_ok=True)
            filename = f"{uuid.uuid4().hex}.png"
            filepath = os.path.join("static", filename)
            plt.savefig(filepath, bbox_inches="tight", dpi=300)
            plt.close()  # Add this line to close the figure and free memory

            return f"/static/{filename}"
        else:
            return attention_scores


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
        choices=["sentence"],
        required=True,
        help="Mode of analysis",
    )
    parser.add_argument(
        "--words",
        type=str,
        nargs="+",
        required=False,
        help="Words to calculate log likelihood for (required for 'sentence' mode)",
    )

    args = parser.parse_args()

    # Load model and tokenizer
    model = AutoModelForCausalLM.from_pretrained(args.model)
    tokenizer = AutoTokenizer.from_pretrained(args.model)

    # Instantiate GenerativeModelOutputs
    generative_outputs = GenerativeModelOutputs(model, tokenizer)

    if args.mode == "sentence":
        if not args.words:
            print("Error: --words argument is required in 'sentence' mode.")
            sys.exit(1)

        output = generative_outputs.sentence_log_likelihoods(args.words)

        print("\n===== Sentence Log Likelihood Analysis =====")
        print(f"Total Log Likelihood: {output.get_total():.4f}")
        print(f"Perplexity: {output.get_perplexity():.4f}")
        print("Token-wise Likelihoods:")
        for token, score in output.get_tokens():
            print(f"  {token}: {score:.4f}")

    else:
        print("Invalid mode or missing required arguments for the selected mode.")
        sys.exit(1)


if __name__ == "__main__":
    main()
