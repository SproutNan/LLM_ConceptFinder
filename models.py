from transformers import AutoModelForCausalLM, AutoTokenizer
from embeddings import Embeddings
from perturbation import Perturbation
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
import functools
import torch

def apply_sft_template(text: str) -> str:
    return f"[INST] <<SYS>>\nYou are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe. Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature.\n\nIf a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information.\n<</SYS>>\n\n{text} [/INST] "

def apply_inst_template(text: str) -> str:
    return f"[INST] {text} [/INST]"

class _Model:
    def __init__(self, model_name: str):
        self.model_name: str = model_name
        self.device: str = 'cuda' if torch.cuda.is_available() else 'cpu'

        self.model = AutoModelForCausalLM.from_pretrained(model_name).to(self.device)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.n_layer = self.model.config.num_hidden_layers
        self.n_dimension = self.model.config.hidden_size

class Model_Extraction(_Model):
    def __init__(self, model_name):
        super().__init__(model_name)

    def collect_embeddings(self, inputs: list[str], message: str = "") -> list[Embeddings]:
        """
        Collect embeddings from the model for given inputs.
        
        Args:
        - inputs: A list of input strings for which embeddings are to be collected.
        - message: An optional message label for the embeddings.
        
        Returns:
        - A list of Embeddings objects, each corresponding to a layer of the model.
        """
        if not inputs:
            raise ValueError("Input list is empty. Please provide valid input strings.")

        embeds_list = [Embeddings(
            origin_model=self.model_name,
            layer_id=i,
            message=message,
        ) for i in range(self.n_layer)]
        for embeddings in embeds_list:
            embeddings.data = torch.zeros(len(inputs), self.n_dimension)

        for input_index, input_text in tqdm(enumerate(inputs)):
            input_text = apply_sft_template(input_text)
            input_ids = self.tokenizer(input_text, return_tensors='pt')['input_ids'].to(self.device)

            with torch.no_grad():  # Disable gradient calculation for inference
                output = self.model(input_ids, output_hidden_states=True)
            hidden_states = output.hidden_states

            for layer in range(self.n_layer):
                embeds_list[layer].data[input_index, :] = hidden_states[layer][:, -1, :].detach().cpu()

        return embeds_list

class Model_Generation(_Model):
    def __init__(self, model_name):
        super().__init__(model_name)
        self.hooks = []
        self._register_hooks()
        self.perturbation = None
        self.embeddings_per_token = []
        self.perturbations_per_token = []

    def set_perturbation(self, perturbation: Perturbation):
        """
        Set the perturbation object that modifies model outputs during generation.
        
        Args:
        - perturbation: An instance of Perturbation class.
        """
        self.perturbation = perturbation

    def _register_hooks(self):
        """
        Register forward hooks on each layer of the model to modify the outputs if needed.
        """
        for i in range(self.n_layer):
            layer = self.model.model.layers[i]  # Update this if model's layer access differs
            hook = layer.register_forward_hook(functools.partial(
                self._modify_layer_output, layer_index=i
            ))
            self.hooks.append(hook)

    def _modify_layer_output(self, module, input, output, layer_index):
        """
        Modify the output of a layer during forward pass based on the perturbation.
        
        Args:
        - module: The layer module.
        - input: Input to the layer.
        - output: Output from the layer.
        - layer_index: Index of the layer being modified.
        
        Returns:
        - Modified output tensor.
        """
        if self.perturbation is not None:
            original_output = output.clone()
            output = self.perturbation.get_perturbation(output, layer_index)
            # FIXME: check this
            perturbation_norm = (output - original_output).norm(dim=-1).mean().item()
            self.perturbations_per_token[layer_index].append(perturbation_norm)
        self.embeddings_per_token[layer_index].append(output[:, -1, :].detach().cpu())
        return output

    def generate(
        self, 
        prompt: str, 
        max_length: int = 1000, 
        output_p: bool = False, 
        output_norm: bool = False,
        output_difference: bool = False,
    ) -> dict:
        """
        Generate text based on the provided prompt using the model.
        
        Args:
        - prompt: The input prompt for the model.
        - max_length: Maximum length of the generated output.
        - output_p: If True, returns classification probabilities for each layer and token.
        - output_norm: If True, returns perturbation norms for each layer and token.
        
        Returns:
        - A dictionary containing generated text, probabilities, and perturbation norms if requested.
        """
        prompt = apply_inst_template(prompt)
        input_ids = self.tokenizer(prompt, return_tensors='pt')['input_ids'].to(self.device)

        # Initialize data structures for collecting embeddings and perturbation norms
        if output_p or output_norm or output_difference:
            self.embeddings_per_token = [[] for _ in range(self.n_layer)]
            self.perturbations_per_token = [[] for _ in range(self.n_layer)]

        output = self.model.generate(
            input_ids,
            max_length=max_length,
            return_dict_in_generate=True,
            output_hidden_states=True,
        )

        generated_text = self.tokenizer.decode(output.sequences[0], skip_special_tokens=True)
        result = {"text": generated_text}

        if output_p:
            probabilities = torch.zeros((self.n_layer, len(self.embeddings_per_token[0])))
            for layer_index in range(self.n_layer):
                for token_index, embds in enumerate(self.embeddings_per_token[layer_index]):
                    prob = self.perturbation.classifier_manager.classifiers[layer_index].predict(embds.unsqueeze(0))
                    probabilities[layer_index, token_index] = prob
            result["probabilities"] = probabilities

        if output_norm:
            perturbation_norms = torch.tensor(self.perturbations_per_token)
            result["perturbation_norms"] = perturbation_norms

        if output_difference:
            self._plot_differences()

        return result

    def _plot_differences(self):
        """
        Plot a large figure containing [n_layer * n_newtoken] subplots.
        Each subplot shows a 2D scatter plot with training and test data,
        and perturbations visualized as arrows.
        """
        n_newtokens = len(self.embeddings_per_token[0])
        fig, axes = plt.subplots(self.n_layer, n_newtokens, figsize=(15, 3 * self.n_layer))

        # Ensure axes is a 2D array even when n_layer or n_newtokens is 1
        if self.n_layer == 1:
            axes = np.expand_dims(axes, axis=0)
        if n_newtokens == 1:
            axes = np.expand_dims(axes, axis=1)

        for layer_index in range(self.n_layer):
            classifier = self.perturbation.classifier_manager.classifiers[layer_index]
            
            # Project training and test data
            pos_train_proj = classifier.project_with_pca(classifier.training_data['pos_train'])
            neg_train_proj = classifier.project_with_pca(classifier.training_data['neg_train'])
            pos_test_proj = classifier.project_with_pca(classifier.training_data['pos_test'])
            neg_test_proj = classifier.project_with_pca(classifier.training_data['neg_test'])

            for token_index in range(n_newtokens):
                ax = axes[layer_index, token_index]
                ax.scatter(pos_train_proj[:, 0], pos_train_proj[:, 1], c='blue', label='Train Positive', s=10)
                ax.scatter(neg_train_proj[:, 0], neg_train_proj[:, 1], c='red', label='Train Negative', s=10)
                ax.scatter(pos_test_proj[:, 0], pos_test_proj[:, 1], c='green', label='Test Positive', s=10)
                ax.scatter(neg_test_proj[:, 0], neg_test_proj[:, 1], c='orange', label='Test Negative', s=10)

                # Project embeddings and perturbations
                embds_proj = classifier.project_with_pca(self.embeddings_per_token[layer_index][token_index])
                perturb_proj = classifier.project_with_pca(
                    self.embeddings_per_token[layer_index][token_index] + self.perturbations_per_token[layer_index][token_index]
                )

                # Plot perturbation as arrows
                ax.arrow(embds_proj[0, 0], embds_proj[0, 1], 
                         perturb_proj[0, 0] - embds_proj[0, 0], 
                         perturb_proj[0, 1] - embds_proj[0, 1], 
                         color='purple', head_width=0.02, length_includes_head=True)

                ax.set_title(f"Layer {layer_index} Token {token_index}")
                ax.set_xlabel("PCA Dimension 1")
                ax.set_ylabel("PCA Dimension 2")
                
                # Only show legends in the first subplot
                if layer_index == 0 and token_index == 0:
                    ax.legend(fontsize='small')
                else:
                    ax.legend().set_visible(False)

        plt.tight_layout()
        plt.savefig("perturbation_differences.png")

    def __del__(self):
        """
        Destructor to remove registered hooks and release resources.
        """
        for hook in self.hooks:
            hook.remove()