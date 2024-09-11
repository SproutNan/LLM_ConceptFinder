from classifier import *
from embeddings import *
import matplotlib.pyplot as plt

clsfr = CAV_ClassifierManager(
    classifier_type="safety",
)

pos_embds_list = load_embedding_list_all_layer_by_message(
    base_model="Llama-2-7b-chat-hf",
    n_layer=32,
    message="MaliciousTrain",
    directory="./dataset/embeddings/RepE",
)

neg_embds_list = load_embedding_list_all_layer_by_message(
    base_model="Llama-2-7b-chat-hf",
    n_layer=32,
    message="SafeTrain",
    directory="./dataset/embeddings/RepE",
)

pos_embds_list_test = load_embedding_list_all_layer_by_message(
    base_model="Llama-2-7b-chat-hf",
    n_layer=32,
    message="MaliciousTest",
    directory="./dataset/embeddings/RepE",
)

neg_embds_list_test = load_embedding_list_all_layer_by_message(
    base_model="Llama-2-7b-chat-hf",
    n_layer=32,
    message="SafeTest",
    directory="./dataset/embeddings/RepE",
)

clsfr.train_and_evaluate_from_embeddings(
    train_pos_embds_list=pos_embds_list,
    train_neg_embds_list=neg_embds_list,
    test_pos_embds_list=pos_embds_list_test,
    test_neg_embds_list=neg_embds_list_test,
)

# check LayerClassifier
print(len(clsfr.classifiers))
layerClsfr = clsfr.classifiers[7]

# check LayerClassifier attributes
print(layerClsfr.layer_index)
for k, v in layerClsfr.training_data.items():
    print(k, v.shape)

# PCA
pos_train_pca = layerClsfr.project_with_pca(layerClsfr.training_data["pos_train"])
neg_train_pca = layerClsfr.project_with_pca(layerClsfr.training_data["neg_train"])
pos_test_pca = layerClsfr.project_with_pca(layerClsfr.training_data["pos_test"])
neg_test_pca = layerClsfr.project_with_pca(layerClsfr.training_data["neg_test"])

print(pos_train_pca.shape)

fig = plt.figure(figsize=(10, 10))
plt.scatter(pos_train_pca[:, 0], pos_train_pca[:, 1], c="red", label="Malicious Train")
plt.scatter(neg_train_pca[:, 0], neg_train_pca[:, 1], c="blue", label="Safe Train")
plt.scatter(pos_test_pca[:, 0], pos_test_pca[:, 1], c="orange", label="Malicious Test")
plt.scatter(neg_test_pca[:, 0], neg_test_pca[:, 1], c="green", label="Safe Test")
plt.legend()
plt.savefig("test_classifier_pca.png")

# test load and save
clsfr.save("./dataset/classifiers/RepE")

clsfr2 = CAV_ClassifierManager(
    classifier_type="safety",
)

clsfr2.load_from_file("./dataset/classifiers/RepE/safety_Llama-2-7b-chat-hf_.pt")

print(len(clsfr2.classifiers))
print(clsfr2.testacc)
print(clsfr2.n_layer)

layerClsfr2 = clsfr2.classifiers[7]
print(layerClsfr2.layer_index)

pos_train_pca2 = layerClsfr2.project_with_pca(layerClsfr2.training_data["pos_train"])
neg_train_pca2 = layerClsfr2.project_with_pca(layerClsfr2.training_data["neg_train"])
pos_test_pca2 = layerClsfr2.project_with_pca(layerClsfr2.training_data["pos_test"])
neg_test_pca2 = layerClsfr2.project_with_pca(layerClsfr2.training_data["neg_test"])

print(pos_train_pca2.shape)

fig = plt.figure(figsize=(10, 10))
plt.scatter(pos_train_pca2[:, 0], pos_train_pca2[:, 1], c="red", label="Malicious Train")
plt.scatter(neg_train_pca2[:, 0], neg_train_pca2[:, 1], c="blue", label="Safe Train")
plt.scatter(pos_test_pca2[:, 0], pos_test_pca2[:, 1], c="orange", label="Malicious Test")
plt.scatter(neg_test_pca2[:, 0], neg_test_pca2[:, 1], c="green", label="Safe Test")

plt.legend()
plt.savefig("test_classifier_pca2.png")

acc = layerClsfr2.evaluate_accuracy(
    layerClsfr2.training_data["pos_test"],
    layerClsfr2.training_data["neg_test"],
)

print(acc)

print("PASSED")
