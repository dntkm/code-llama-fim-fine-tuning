from datasets import load_from_disk

dataset_path = '/Users/wangdongyang/Workspace/Projects/OtherProject/CodeLLamaFinetuning/TangSengDaoDaoiOS_dataset'
dataset = load_from_disk(dataset_path)

for data in dataset:
    print(data["filepath"], data["content"])
    break

