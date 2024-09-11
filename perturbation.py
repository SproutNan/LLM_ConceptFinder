from classifier import CAV_ClassifierManager
import torch

class Perturbation:
    def __init__(
        self, 
        classifier_manager: CAV_ClassifierManager, 
        target_probability: float = 0.0001, 
        accuracy_threshold: float = 0.9, 
        mask: list[bool] = None
    ):
        """
        Initialize the Perturbation class.
        
        Args:
        - classifier_manager: An instance of CAV_ClassifierManager.
        - target_probability: The target probability for the perturbation.
        - accuracy_threshold: Threshold accuracy above which perturbation is applied.
        - mask: A list of booleans indicating which layers should be perturbed.
        """
        self.classifier_manager = classifier_manager
        self.mask = mask
        self.target_probability = target_probability
        self.accuracy_threshold = accuracy_threshold

    def get_perturbation(self, output: torch.Tensor, layer: int) -> torch.Tensor:
        """
        Apply perturbation to the embeddings of a specified layer if conditions are met.
        
        Args:
        - output: The output tensor containing embeddings.
        - layer: The index of the layer to apply perturbation.
        
        Returns:
        - The perturbed output tensor.
        """
        if self.mask is None or (self.mask and self.mask[layer]):
            if self.classifier_manager.testacc[layer] > self.accuracy_threshold:
                perturbed_embds = self.classifier_manager.calculate_perturbation(
                    embds=output[0][0, -1, :], 
                    layer=layer, 
                    target_prob=self.target_probability
                )
                output[0][0, -1, :] = perturbed_embds
        return output