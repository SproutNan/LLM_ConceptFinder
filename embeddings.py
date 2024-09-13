import torch
import os
from typing import Optional, List

class Embeddings:
    """
    Embeddings class to handle the loading and saving of embeddings from transformer layers.
    
    Attributes:
    - origin_model: The name of the original model.
    - layer_id: The layer index from which the embeddings were extracted.
    - message: An optional message or label for the embeddings.
    - data: A tensor containing the embeddings data.
    """

    def __init__(
        self, 
        origin_model: Optional[str] = None, 
        layer_id: Optional[int] = None, 
        message: str = "", 
        data: Optional[torch.Tensor] = None
    ):
        self.origin_model = os.path.basename(origin_model) if origin_model else None
        self.layer_id = layer_id
        self.message = message
        self.data = data

    def load(self, path: str):
        """
        Initialize Embeddings from a file.
        
        Args:
        - path: The path to the embeddings file.
        
        Returns:
        - self: Returns the instance to support method chaining.
        """
        if not path:
            print("Warning: Empty path provided for loading embeddings, skipping.")
            return self  # 如果路径为空，直接返回未修改的对象
        
        try:
            self.data = torch.load(path, weights_only=True)
        except Exception as e:
            print(f"Error loading file: {e}")
            return self

        # 自动从文件名中提取模型信息和层号
        file_name = os.path.basename(path)
        embds_info = file_name.split("_")
        self.origin_model = embds_info[0]
        if embds_info[1].isdigit():
            # [model]_[layer]_[message].pt
            self.layer_id = int(embds_info[1])
            self.message = embds_info[2].split(".")[0]
        else:
            # [model]_[layer].pt
            self.layer_id = int(embds_info[1].split(".")[0])
            self.message = ""
        
        return self

    def save(self, relative_path: str):
        """
        Save the embeddings data to a file.
        
        Args:
        - relative_path: The directory to save the embeddings.
        """
        if self.data is None or not isinstance(self.data, torch.Tensor):
            print("Error: No valid data to save.")
            return
        
        if not os.path.exists(relative_path):
            os.makedirs(relative_path)

        save_path = os.path.join(relative_path, f"{self.origin_model}_{self.layer_id}.pt")
        if self.message:
            save_path = os.path.join(relative_path, f"{self.origin_model}_{self.layer_id}_{self.message}.pt")
        
        try:
            torch.save(self.data, save_path)
            print(f"Embeddings saved to {save_path}")
        except Exception as e:
            print(f"Error saving embeddings: {e}")

def merge_embeddings(embeddings_list: List[Embeddings]) -> Embeddings:
    """
    Merge a list of Embeddings objects by concatenating their data along the first dimension.
    
    Args:
    - embeddings_list: A list of Embeddings objects to be merged. All objects must have the same origin_model and layer_id.
    
    Returns:
    - An Embeddings object with concatenated data from all provided Embeddings objects.
    """
    
    if not embeddings_list:
        raise ValueError("The embeddings list is empty, nothing to merge.")

    base_model = embeddings_list[0].origin_model
    base_layer = embeddings_list[0].layer_id
    base_shape = embeddings_list[0].data.shape[1] if embeddings_list[0].data is not None else None

    for idx, emb in enumerate(embeddings_list):
        if emb.origin_model != base_model or emb.layer_id != base_layer:
            raise ValueError(
                f"Embedding at index {idx} is from model {emb.origin_model} "
                f"or layer {emb.layer_id}, which does not match the base model {base_model} and layer {base_layer}."
            )
        if emb.data is None or emb.data.shape[1] != base_shape:
            raise ValueError(
                f"Embedding at index {idx} has incompatible dimensions: expected {base_shape}, "
                f"but got {emb.data.shape if emb.data is not None else 'None'}."
            )

    merged_data = torch.cat([emb.data for emb in embeddings_list], dim=0)

    return Embeddings(
        origin_model=base_model,
        layer_id=base_layer,
        message="merged",
        data=merged_data
    )

def load_embedding_list_all_layer_by_message(
    base_model: str,
    n_layer: int,
    message: str,
    directory: str,
) -> List[Embeddings]:
    """
    Load embeddings from multiple layers based on the base model name and message, within a specified directory.
    
    Args:
    - base_model: The base model name used in the saved embeddings filenames.
    - n_layer: The number of layers for which embeddings need to be loaded.
    - message: The message part of the embedding filenames.
    - directory: The root directory where the embedding files are stored.
    
    Returns:
    - A list of Embeddings objects, each corresponding to a specific layer.
    """

    base_model = os.path.basename(base_model)

    embedding_lists = [
        Embeddings().load(os.path.join(directory, f"{base_model}_{i}_{message}.pt"))
        for i in range(n_layer)
    ]

    return embedding_lists