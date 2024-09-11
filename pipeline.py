from embeddings import *
from classifier import *
from query_dataset import *
from models import *
import matplotlib.pyplot as plt

# 根据任务名提取指令数据集
# 放到指定模型中提取指令的 embeddings
def extract_embeddings(
    origin_model: str,
    task_name: str,
    train_size: float = 1.0,
    label_list: List[str] = ["Malicious", "Safe"],
):
    model_extract = Model_Extraction(origin_model)
    if train_size == 1.0:
        inst_lists = load_instructions_by_dataset_name_only_train(
            name=task_name,
            label_list=label_list,
        )
        pos_inst_list = inst_lists[0]
        neg_inst_list = inst_lists[1]

        pos_embds_list = model_extract.collect_embeddings(pos_inst_list, message=label_list[0])
        neg_embds_list = model_extract.collect_embeddings(neg_inst_list, message=label_list[1])

        return pos_embds_list, neg_embds_list
    
    else:
        inst_lists = load_instructions_by_dataset_name(
            name=task_name,
            label_list=label_list,
            shuffle=True,
            train_size=train_size,
        )
        label_list = [
            f"{label_list[0]}Train",
            f"{label_list[0]}Test",
            f"{label_list[1]}Train",
            f"{label_list[1]}Test",
        ]
        pos_inst_list_train = inst_lists[0]
        pos_inst_list_test = inst_lists[1]
        neg_inst_list_train = inst_lists[2]
        neg_inst_list_test = inst_lists[3]

        pos_embds_list_train = model_extract.collect_embeddings(pos_inst_list_train, message=label_list[0])
        pos_embds_list_test = model_extract.collect_embeddings(pos_inst_list_test, message=label_list[1])
        neg_embds_list_train = model_extract.collect_embeddings(neg_inst_list_train, message=label_list[2])
        neg_embds_list_test = model_extract.collect_embeddings(neg_inst_list_test, message=label_list[3])

        return pos_embds_list_train, pos_embds_list_test, neg_embds_list_train, neg_embds_list_test

# 训练分类器（所有层）
# 如果不提供测试集，则只训练分类器，不测试精度
def train_classifier(
    classifier_type: str,
    pos_embds_list_train: List[Embeddings],
    neg_embds_list_train: List[Embeddings],
    pos_embds_list_test: List[Embeddings]=None,
    neg_embds_list_test: List[Embeddings]=None,
    lr: float = 0.001,
    epochs: int = 100,
    batch_size: int = 32,
):
    clsfr = CAV_ClassifierManager(classifier_type)
    if pos_embds_list_test is None or neg_embds_list_test is None:
        clsfr.train_classifiers(
            pos_embds_list_train,
            neg_embds_list_train,
            lr=lr,
            epochs=epochs,
            batch_size=batch_size,
        )
        clsfr.message = "OnlyTrain"
    else:
        clsfr.train_and_evaluate_from_embeddings(
            pos_embds_list_train,
            neg_embds_list_train,
            pos_embds_list_test,
            neg_embds_list_test,
            lr=lr,
            epochs=epochs,
            batch_size=batch_size,
        )
        clsfr.message = "Complete"
    return clsfr

# 降维并绘制图像（单类）
def reduction_single_class(
    clsfr: CAV_ClassifierManager,
    embeddings_to_reduce: Embeddings,
    output_name: str="reduction.png",
):
    assert embeddings_to_reduce.layer_id >= 0, "Layer ID must be provided."

    layer = embeddings_to_reduce.layer_id
    layerClsfr: LayerClassifier = clsfr.classifiers[layer]

    plt.figure(figsize=(5, 5))
    pos_train_pca = layerClsfr.project_with_pca(layerClsfr.training_data["pos_train"])
    neg_train_pca = layerClsfr.project_with_pca(layerClsfr.training_data["neg_train"])

    reduced_embds = layerClsfr.project_with_pca(embeddings_to_reduce.data)
    plt.scatter(pos_train_pca[:, 0], pos_train_pca[:, 1], c="red", label="TrainClass1")
    plt.scatter(neg_train_pca[:, 0], neg_train_pca[:, 1], c="blue", label="TrainClass2")
    plt.scatter(reduced_embds[:, 0], reduced_embds[:, 1], c="green", label="ReducedEmbds")
    plt.title(f"Layer {layer} Customized Reduction")
    plt.legend()
    plt.savefig(output_name)
    plt.close()

# 降维并绘制图像（两类），返回测试精度
def reduction_two_class(
    clsfr: CAV_ClassifierManager,
    pos_embds: Embeddings,
    neg_embds: Embeddings,
    output_name: str="reduction.png",
) -> float:
    assert pos_embds.layer_id >= 0, "Layer ID must be provided."
    assert neg_embds.layer_id >= 0, "Layer ID must be provided."
    assert pos_embds.layer_id == neg_embds.layer_id, "Layer ID must be the same."

    layer = pos_embds.layer_id
    layerClsfr: LayerClassifier = clsfr.classifiers[layer]

    plt.figure(figsize=(5, 5))
    pos_train_pca = layerClsfr.project_with_pca(layerClsfr.training_data["pos_train"])
    neg_train_pca = layerClsfr.project_with_pca(layerClsfr.training_data["neg_train"])

    pos_reduced_embds = layerClsfr.project_with_pca(pos_embds.data)
    neg_reduced_embds = layerClsfr.project_with_pca(neg_embds.data)
    plt.scatter(pos_train_pca[:, 0], pos_train_pca[:, 1], c="red", label="TrainClass1")
    plt.scatter(neg_train_pca[:, 0], neg_train_pca[:, 1], c="blue", label="TrainClass2")
    plt.scatter(pos_reduced_embds[:, 0], pos_reduced_embds[:, 1], c="green", label="PosEmbds")
    plt.scatter(neg_reduced_embds[:, 0], neg_reduced_embds[:, 1], c="orange", label="NegEmbds")
    plt.title(f"Layer {layer} Customized Reduction")
    plt.legend()
    plt.savefig(output_name)
    plt.close()

    test_acc = layerClsfr.evaluate_accuracy(pos_embds.data, neg_embds.data)

    return max(test_acc, 1 - test_acc)

# 根据指定模型生成问题的答案
def generation_with_perturbation(
    model_generate: Model_Generation,
    prompt: str,
    perturbation: Perturbation,
    max_length: int = 1000,
    output_p: bool = False, 
    output_norm: bool = False,
    output_difference: bool = False,
):
    model_generate.set_perturbation(perturbation)
    return model_generate.generate(
        prompt,
        max_length=max_length,
        output_p=output_p,
        output_norm=output_norm,
        output_difference=output_difference,
    )

def generation_without_perturbation(
    model_generate: Model_Generation,
    prompt: str,
    max_length: int = 1000,
):
    model_generate.set_perturbation(None)
    return model_generate.generate(
        prompt,
        max_length=max_length,
    )