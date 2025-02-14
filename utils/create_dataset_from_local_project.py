import os
from datasets import Dataset
from tqdm import tqdm  # 用于显示进度条（可选）

def create_objc_dataset(repo_path, output_path=".\objc_dataset"):
    """
    从本地Git仓库创建Objective-C数据集
    
    参数：
    repo_path: 本地git仓库路径
    output_path: 数据集输出目录
    """
    # 递归查找所有Objective-C文件
    objc_files = []
    for root, _, files in os.walk(repo_path):
        for file in files:
            if file.endswith(('.m', '.h', '.mm')):
                full_path = os.path.join(root, file)
                objc_files.append(full_path)

    # 构建数据集字典
    dataset_dict = {
        "filepath": [],
        "content": []
    }

    # 读取文件内容
    for filepath in tqdm(objc_files, desc="Processing files"):
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                content = f.read()
                dataset_dict["filepath"].append(filepath)
                dataset_dict["content"].append(content)
        except Exception as e:
            print(f"跳过文件 {filepath}，读取失败：{str(e)}")

    # 创建HuggingFace数据集
    dataset = Dataset.from_dict(dataset_dict)
    
    # 保存数据集
    os.makedirs(output_path, exist_ok=True)
    dataset.save_to_disk(output_path)
    
    print(f"数据集已保存至 {output_path}，包含 {len(dataset)} 个样本")
    return dataset

# 使用示例
if __name__ == "__main__":
    create_objc_dataset(
        repo_path="/Users/wangdongyang/Workspace/Trash/llm/LLmDataSource/TangSengDaoDaoiOS",  # 替换为你的本地仓库路径
        output_path="./TangSengDaoDaoiOS_dataset"
    )