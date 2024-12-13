# BlackBoxScanner

Theis library enables you to explore the interpretability of large language models, by exploiting their inner likelihoods, tokens, and other interesting inner workings.

The purpose of this library is to provide a more accessible platform to researchers, academics, and enthusiasts.
Feel free to improve/expand the present codebase through a pull request.

1. GenerativeModelOutputs: this method looks inside your LLM to obtain the inner workings of the llm. 
--> sentence_log_likelihoods : this method will output a ```python ScannedOutputs ``` object, which contains tokens and their respective negative log likelihoods (NLLs) of being outputted by the llm. The NLLs are such that a value closer to zero denotes a higher likelihood of being outputted by the llm. This method is to be used in conjunction with an initial context. For instance, context: "The furry pet is a ..." and the output could be {"dog": -90.556, "cat": -91.663} meaning that dog is more likely to be outputted by the LLM than cat.

--> view_top_k : this method is designed to simple provide a view of the next top k probable tokens which the llm may have outputted given a certain context, with k as an input parameter. For now, this method uses the greedy approach to outline the next tokens, but this can soon be changed in a later version to obtain through a parameter.

