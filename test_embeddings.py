from embeddings import *

# init
embd = Embeddings(
    origin_model="model",
    layer_id=1,
    message="message",
    data=torch.randn(10, 768)
)

print(embd.origin_model)  # model
print(embd.layer_id)  # 1
print(embd.message)  # message
print(embd.data.shape)  # torch.Size([10, 768])

# save
embd.save("./dataset/embeddings/test")

# load
embd2 = Embeddings().load("./dataset/embeddings/test/model_1_message.pt")

print(embd2.origin_model)  # model
print(embd2.layer_id)  # 1
print(embd2.message)  # message
print(embd2.data.shape)  # torch.Size([10, 768])

# check if the data is the same
print(torch.allclose(embd.data, embd2.data))  # True

# merge the two embeddings
embd3 = merge_embeddings([embd, embd2])

print(embd3.origin_model)  # model
print(embd3.layer_id)  # 1
print(embd3.message)  # merged
print(embd3.data.shape)  # torch.Size([20, 768])

embd3.layer_id = 0
embd3.message = "message"
embd3.save("./dataset/embeddings/test")

# load list all layer by message
embd_list = load_embedding_list_all_layer_by_message("model", 2, "message", "./dataset/embeddings/test")

print(len(embd_list))  # 2
print(embd_list[0].origin_model)  # model
print(embd_list[0].layer_id)  # 0
print(embd_list[0].message)  # message
print(embd_list[0].data.shape)  # torch.Size([20, 768])

print(embd_list[1].origin_model)  # model
print(embd_list[1].layer_id)  # 1
print(embd_list[1].message)  # message
print(embd_list[1].data.shape)  # torch.Size([10, 768])

print("PASSED")