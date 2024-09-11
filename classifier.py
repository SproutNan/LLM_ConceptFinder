from sklearn.decomposition import PCA
import torch
from torch.utils.data import DataLoader, TensorDataset
from torch import nn, optim
from torch.optim.lr_scheduler import StepLR
from embeddings import Embeddings
import os
from typing import List

class LayerClassifier:
    """
    Represents a single layer classifier using logistic regression.
    
    Attributes:
    - model: A PyTorch sequential model for logistic regression.
    - criterion: Loss function for the classifier.
    - optimizer: Optimizer for training the classifier.
    - layer_index: Index of the layer this classifier represents.
    """
    def __init__(self, input_dim: int, layer_index: int, lr: float = 0.01):
        self.layer_index = layer_index
        self.model = nn.Sequential(
            nn.Linear(input_dim, 1),
            nn.Sigmoid()
        ).to('cuda')
        self.criterion = nn.BCELoss()
        self.optimizer = optim.SGD(self.model.parameters(), lr=lr)
        self.scheduler = StepLR(self.optimizer, step_size=50, gamma=0.5)  # Learning rate scheduler
        self.pca = PCA(n_components=2)
        self.training_data = {
            "pos_train": None,
            "neg_train": None,
            "pos_test": None,
            "neg_test": None
        }

    def train(self, pos_data: torch.Tensor, neg_data: torch.Tensor, epochs: int = 100, batch_size: int = 64) -> List[float]:
        """Trains the classifier using positive and negative embeddings."""
        X = torch.vstack((pos_data, neg_data)).to('cuda')
        Y = torch.cat((torch.ones(pos_data.size(0)), torch.zeros(neg_data.size(0)))).to('cuda')

        self.training_data["pos_train"] = pos_data.cpu()
        self.training_data["neg_train"] = neg_data.cpu()

        # Create DataLoader for batching
        dataset = TensorDataset(X, Y)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        
        loss_list = []
        for epoch in range(epochs):
            epoch_loss = 0.0
            for batch_X, batch_Y in loader:
                self.optimizer.zero_grad()
                outputs = self.model(batch_X).squeeze()
                loss = self.criterion(outputs, batch_Y)
                loss.backward()
                self.optimizer.step()
                epoch_loss += loss.item()

            # Update learning rate and store the average loss for this epoch
            self.scheduler.step()
            average_loss = epoch_loss / len(loader)
            loss_list.append(average_loss)

        combined_data = torch.vstack((self.training_data["pos_train"], self.training_data["neg_train"]))
        self.pca.fit(combined_data)

        return loss_list

    def predict(self, data: torch.Tensor) -> torch.Tensor:
        """
        Predicts the labels for the given data.
        """
        self.model.eval()
        with torch.no_grad():
            outputs = self.model(data).squeeze()
        return outputs

    def evaluate_accuracy(self, pos_data: torch.Tensor, neg_data: torch.Tensor) -> float:
        """Evaluate the classifier accuracy with positive and negative test data."""
        test_data = torch.vstack((pos_data, neg_data)).to('cuda')
        predictions = self.predict(test_data)
        true_labels = torch.cat((torch.ones(pos_data.size(0)), torch.zeros(neg_data.size(0)))).to('cuda')
        correct_count = ((predictions > 0.5) == true_labels).sum()
        
        self.training_data["pos_test"] = pos_data.cpu()
        self.training_data["neg_test"] = neg_data.cpu()
        
        accuracy = correct_count.item() / len(true_labels)
        return accuracy

    def get_weights_and_bias(self):
        """
        Returns the weights and bias of the trained classifier.
        """
        weights = self.model[0].weight.data.squeeze().cpu()
        bias = self.model[0].bias.data.cpu()
        return weights, bias

    def load_state_dict(self, state_dict):
        self.model.load_state_dict(state_dict["model"])
        self.pca = state_dict["pca"]
        self.training_data = state_dict["training_data"]

    def get_state_dict(self):
        state_dict = {
            "model": self.model.state_dict(),
            "pca": self.pca,
            "training_data": self.training_data
        }
        return state_dict

    def project_with_pca(self, data: torch.Tensor) -> torch.Tensor:
        """
        Projects the data into a 2D space using PCA.
        """
        return torch.from_numpy(self.pca.transform(data.cpu()))

class CAV_ClassifierManager:
    """
    Manages multiple LayerClassifiers to train and evaluate classifiers across all layers.
    
    Attributes:
    - type: Type of classifier, e.g., 'safety'.
    - classifiers: List of LayerClassifier instances.
    - testacc: List of test accuracies for each layer.
    - n_layer: Number of layers (classifiers).
    """
    def __init__(self, classifier_type: str = "safety"):
        self.type = classifier_type
        self.classifiers = []  # Store LayerClassifier instances
        self.testacc = []  # Test accuracy for each layer
        self.n_layer = 0  # Number of classifiers

    def train_classifiers(self, pos_embds_list: List[Embeddings], neg_embds_list: List[Embeddings], lr: float = 0.01, epochs: int = 100, batch_size: int = 64):
        self.n_layer = len(pos_embds_list)
        self.origin_model = pos_embds_list[0].origin_model
        self.message = ""

        print("Training CAV Classifiers...")

        for i in range(self.n_layer):
            print(f"Training layer {i}...")
            input_dim = pos_embds_list[i].data.shape[1]
            layer_classifier = LayerClassifier(input_dim, layer_index=i, lr=lr)
            pos_data, neg_data = pos_embds_list[i].data, neg_embds_list[i].data
            
            layer_classifier.train(pos_data, neg_data, epochs=epochs, batch_size=batch_size)
            self.classifiers.append(layer_classifier)

    def evaluate_test_accuracy(self, pos_embds_test_list: List[Embeddings], neg_embds_test_list: List[Embeddings]):
        if self.n_layer != len(pos_embds_test_list) or self.n_layer != len(neg_embds_test_list):
            raise ValueError("The number of layers in the classifier does not match the number of test embeddings provided.")
        
        print("Calculating test accuracy...")

        for i in range(self.n_layer):
            accuracy = self.classifiers[i].evaluate_accuracy(
                pos_embds_test_list[i].data, 
                neg_embds_test_list[i].data
            )
            self.testacc.append(accuracy)
            print(f"Layer {i}, Accuracy: {accuracy:.4f}")

    def train_and_evaluate_from_embeddings(
        self, 
        train_pos_embds_list: List[Embeddings],
        train_neg_embds_list: List[Embeddings],
        test_pos_embds_list: List[Embeddings],
        test_neg_embds_list: List[Embeddings],
        lr: float = 0.01,
        epochs: int = 100,
        batch_size: int = 64,
    ):
        self.train_classifiers(
            train_pos_embds_list, 
            train_neg_embds_list,
            lr=lr,
            epochs=epochs,
            batch_size=batch_size,
        )
        self.evaluate_test_accuracy(test_pos_embds_list, test_neg_embds_list)
        return self

    def save(self, relative_path: str):
        """Save the classifier weights and test accuracies."""
        try:
            save_path = os.path.join(relative_path, f"{self.type}_{self.origin_model}_{self.message}.pt")
            
            # Combine all classifier states into a single state_dict
            state_dict = {
                "n_layer": self.n_layer,
                "testacc": self.testacc,
                "classifiers": {f"layer_{i}": classifier.get_state_dict() for i, classifier in enumerate(self.classifiers)}
            }
            torch.save(state_dict, save_path)
            print(f"ClassifierManager saved to {save_path}")
        except Exception as e:
            print("Error when saving CAV ClassifierManager:", e)

    def load_from_file(self, file_name: str):
        """Load the classifier manager from a saved file."""
        try:
            state_dict = torch.load(file_name)
            self.n_layer = state_dict["n_layer"]
            self.testacc = state_dict["testacc"]
            self.classifiers = []
            for i in range(self.n_layer):
                input_dim = state_dict["classifiers"][f"layer_{i}"]['model']["0.weight"].shape[1]
                classifier = LayerClassifier(input_dim, i)  # Initialize with correct input_dim
                classifier.load_state_dict(state_dict["classifiers"][f"layer_{i}"])
                self.classifiers.append(classifier)
            print(f"Loaded CAV ClassifierManager from {file_name}")
        except Exception as e:
            print("Error when loading CAV ClassifierManager:", e)

    def predict(self, embds_data: torch.Tensor, layer: int):
        """Predict labels using the classifier for the specified layer."""
        return self.classifiers[layer].predict(embds_data)

    def calculate_perturbation(self, embds: torch.Tensor, layer: int, target_prob: float = 0.001):
        """Calculate the perturbation vector for the given embeddings using the specified layer's classifier."""
        classifier = self.classifiers[layer]
        weights, bias = classifier.get_weights_and_bias()
        logit_target = torch.log(torch.tensor(target_prob / (1 - target_prob)))
        weights_norm = torch.norm(weights)

        epsilon = (logit_target - bias - torch.matmul(embds.detach().cpu(), weights)) / weights_norm
        perturbation_direction = weights / weights_norm
        perturbation_vector = epsilon * perturbation_direction

        return embds + perturbation_vector.to(embds.device)
