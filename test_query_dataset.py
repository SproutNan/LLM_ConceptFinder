from query_dataset import *
import pandas as pd

list1 = load_instructions_by_dataset_name_only_train(name="Advbench", label_list=['Malicious', 'Safe'])

print("len(list1):", len(list1))
print("len(list1[0]):", len(list1[0]))
print(list1[0][0])
print("len(list1[1]):", len(list1[1]))
print(list1[1][0])

list2 = load_instructions_by_dataset_name_only_train(name="Advbench", label_list=['yes', 'no'])

print("len(list2):", len(list2))
print("len(list2[0]):", len(list2[0]))
print("len(list2[1]):", len(list2[1]))

list3 = load_instructions_by_dataset_name(
    name="RepE",
    label_list=['Malicious', 'Safe'],
    shuffle=False,
    train_size=0.8 # 不起作用，因为已经标记了 TrainTestLabel
)

print("len(list3):", len(list3))
print("len(list3[0]):", len(list3[0]))
print("len(list3[1]):", len(list3[1]))
print("len(list3[2]):", len(list3[2]))
print("len(list3[3]):", len(list3[3]))

list4 = load_instructions_by_dataset_name(
    name="FrenchConcept",
    label_list=['English', 'French'],
    shuffle=True,
    train_size=0.8
)

print("len(list4):", len(list4))
print("len(list4[0]):", len(list4[0]))
print(list4[0][0])
print("len(list4[1]):", len(list4[1]))
print(list4[1][0])
print("len(list4[2]):", len(list4[2]))
print(list4[2][0])
print("len(list4[3]):", len(list4[3]))
print(list4[3][0])

print("PASSED")