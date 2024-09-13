from pipeline import *
from classifier import *
from embeddings import *
from query_dataset import *
from models import *
import os

origin_model = "meta-llama/Meta-Llama-3-8B-Instruct"
task_name = "RepE"

# Step 1: Extract embeddings
if not os.path.exists(f"./dataset/embeddings/{task_name}"):
    embeddings_lists = extract_embeddings(
        origin_model=origin_model,
        task_name=task_name,
        train_size=0.6,
        label_list=["Malicious", "Safe"],
    )

    os.makedirs(f"./dataset/embeddings/{task_name}")
    for embeddings_list in embeddings_lists:
        for embeddings in embeddings_list:
            embeddings.save(f"./dataset/embeddings/{task_name}")
else:
    print("Embeddings folder already exist, skipping...")

# Step 2: Train and evaluate classifiers
if not os.path.exists(f"./dataset/classifiers/{task_name}"):
    pos_embds_list_train = load_embedding_list_all_layer_by_message(
        base_model=origin_model,
        n_layer=32,
        message="MaliciousTrain",
        directory=f"./dataset/embeddings/{task_name}",
    )
    neg_embds_list_train = load_embedding_list_all_layer_by_message(
        base_model=origin_model,
        n_layer=32,
        message="SafeTrain",
        directory=f"./dataset/embeddings/{task_name}",
    )
    pos_embds_list_test = load_embedding_list_all_layer_by_message(
        base_model=origin_model,
        n_layer=32,
        message="MaliciousTest",
        directory=f"./dataset/embeddings/{task_name}",
    )
    neg_embds_list_test = load_embedding_list_all_layer_by_message(
        base_model=origin_model,
        n_layer=32,
        message="SafeTest",
        directory=f"./dataset/embeddings/{task_name}",
    )

    clsfr = train_classifier(
        classifier_type="safety",
        pos_embds_list_train=pos_embds_list_train,
        neg_embds_list_train=neg_embds_list_train,
        pos_embds_list_test=pos_embds_list_test,
        neg_embds_list_test=neg_embds_list_test,
    )

    os.makedirs(f"./dataset/classifiers/{task_name}")
    clsfr.save(f"./dataset/classifiers/{task_name}")
else:
    print("Classifiers folder already exist, skipping...")