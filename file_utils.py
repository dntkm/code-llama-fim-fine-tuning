import numpy as np
import re
import random

def remove_comments(file_content):
    # 单行注释
    single_line_pattern = r'//.*$'
    new_content = re.sub(single_line_pattern, '', file_content, flags=re.MULTILINE)

    # 多行注释
    multi_line_pattern = r'/\*.*?\*/'
    new_content = re.sub(multi_line_pattern, '', new_content, flags=re.DOTALL)
    
    return new_content

def split_file(file_content):
    # 获取依赖的文件名
    pattern = r'#import\s+(["<])(.*?)(.h[">])'
    matches = re.findall(pattern, file_content)

    filenames = list(set([match[1] for match in matches]))

    # 获取content
    new_content = re.sub(r'^#import.*$', '', file_content, flags=re.MULTILINE)
    
    # 删除连续的换行符
    new_content = re.sub(r'\n{2,}', '\n', new_content)
    return sorted(filenames, key=lambda x: len(x)), new_content

def get_random_file_dependencies_by_max_length(dependencies, max_length):
    result = []
    remaining_length = max_length
    list_index = list(range(len(dependencies)))
    total_length = sum([len(name) for name in dependencies]) + len(dependencies) - 1
    
    # 尾部文件名超长
    while len(list_index) and len(dependencies[list_index[-1]]) > remaining_length:
        total_length = total_length - len(dependencies[list_index[-1]]) - 1
        list_index.pop()
    
    while len(list_index) and remaining_length > 0 and total_length > remaining_length:
        # 获取随机头文件名
        random_index = random.randint(0, len(list_index) - 1)
        random_string = dependencies[list_index[random_index]]
        
        result.append(random_string)
        remaining_length = remaining_length - len(random_string) - 1
        total_length = total_length - len(random_string) - 1
        
        # 处理可用index 以及 剩下的文件名总长
        list_index.remove(list_index[random_index])
        while len(list_index) and len(dependencies[list_index[-1]]) > remaining_length:
            total_length = total_length - len(dependencies[list_index[-1]]) - 1
            list_index.pop()
    
    # 剩余文件名可以全部放入
    if total_length <= remaining_length and len(list_index):
        result += dependencies[:list_index[-1]+1]
    
    result = list(set(result))
    random.shuffle(result)
    result_str = " ".join(set(result))
    assert len(result_str) <= max_length, "Warning: dependency string length exceeded."
    return result_str

def split_contents(contents, np_rng):
    try:
        # A boundary can be =0 (prefix will be empty)
        # a boundary can be =len(contents) (suffix will be empty)
        # The two boundaries can be equal (middle will be empty)
        boundaries = list(np_rng.randint(low=0, high=len(contents) + 1, size=2))
        boundaries.sort()
    except ValueError as e:
        print(len(contents), contents)
        print(e)
        raise e

    prefix = contents[: boundaries[0]]
    middle = contents[boundaries[0] : boundaries[1]]
    suffix = contents[boundaries[1] :]
    return [prefix, middle, suffix, np_rng]

# Adapted from https://github.com/bigcode-project/Megatron-LM/blob/6c4bf908df8fd86b4977f54bf5b8bd4b521003d1/megatron/data/gpt_dataset.py#L491
def permute_char_level(
    sample,
    dependencies,
    dependencies_max_length,
    dependencies_on_prefix,
    np_rng,
    fim_rate,
    fim_spm_rate,
    suffix_tok_id,
    prefix_tok_id,
    middle_tok_id,
    pad_tok_id,
    tokenizer,
    truncate_or_pad=-1,
):
    """
    Take in a sample (np array w/ size (0,chunklength)) and perform a FIM transformation on it.
    Maintain the same sample length (if transform creates a few extra tokens, drop them).
    """

    # 对 fim_rate 比例的 data 内容进行 fim 切分， OpenAI 论文经验值为 0.9
    if np_rng.binomial(1, fim_rate):  # sample bernoulli dist
        # 获取满足max_length的随机file_dependencies
        random_dependencies = get_random_file_dependencies_by_max_length(dependencies, dependencies_max_length)
        
        # format 为 SPM or PSM 格式， 据 OpenAI 论文， 50-50 的 PSM 和 SPM 混训效果较好
        use_PSM = np_rng.binomial(1, fim_spm_rate)
        
        # 这里 decode 了输入的 token 序列，而后进行了 char level 的 fim 划分
        contents = tokenizer.decode(sample, skip_special_tokens=True)
        [prefix, middle, suffix, np_rng] = split_contents(contents, np_rng)
        
        # dependencies_on_prefix = True  则将prefix固定在prefix前
        # dependencies_on_prefix = False 则将prefix固定在整个结构前
        if use_PSM or dependencies_on_prefix:
            prefix = random_dependencies + prefix
        else:
            suffix = random_dependencies + suffix

        # By adding and removing the <MID> token, we ensure that the tokenizer doesn't add extra leading whitespace
        # The prefix whitespace doesn't matter as also mentioned in the Code Llama paper
        # L-TODO: 论文提及的 special_token，用于优化输出内容格式？ 还没搞明白原理
        # 不过在模型应用场景中，应该也需要在生成prompt时添加该 sp token吧
        special_token = "▔"
        special_token_id = tokenizer.encode(
            special_token, add_special_tokens=False, return_tensors="np"
        )[0]
        special_token_id_len = special_token_id.shape[0]

        prefix = tokenizer.encode(
            random_dependencies + prefix, add_special_tokens=False, return_tensors="np"
        )[0]
        middle = tokenizer.encode(
            special_token + middle, add_special_tokens=False, return_tensors="np"
        )[0][special_token_id_len:]
        suffix = tokenizer.encode(
            special_token + suffix, add_special_tokens=False, return_tensors="np"
        )[0][special_token_id_len:]

        # here we truncate each given segment to fit the same length as it was before
        # A consequence is that we never reach the end of a file?
        # we should rather truncate at the context-level
        if truncate_or_pad >= 0:
            # need to make same length as the input. Take the 3 sentinel tokens into account. truncate_or_pad is the number of tokens added after the transformation
            new_length = (
                suffix.shape[0]
                + prefix.shape[0]
                + middle.shape[0]
                + 3
                + truncate_or_pad
            )
            diff = new_length - sample.shape[0]
            if diff > 0:  # too long
                if (
                    suffix.shape[0] <= diff
                ):  # if there's no space to truncate the suffix: stop and report it.
                    print("suffix too short", diff, suffix.shape[0])
                    return sample[: sample.shape[0] - truncate_or_pad], np_rng
                suffix = suffix[: suffix.shape[0] - diff]
            elif diff < 0:  # too short
                suffix = np.concatenate([suffix, np.full((-1 * diff), pad_tok_id)])

        if not use_PSM:
            # SPM
            # 此处好像是根据 OpenAI 论文使用的 SPM 格式: <PRE> <SUF> {suffix} <MID> <prefix> <middle> <EOT>
            # magic format? why??
            new_sample = np.concatenate(
                [
                    [prefix_tok_id, suffix_tok_id],
                    suffix,
                    [middle_tok_id],
                    prefix,
                    middle,
                ]
            )
        else:
            # PSM
            new_sample = np.concatenate(
                [
                    [prefix_tok_id],
                    prefix,
                    [suffix_tok_id],
                    suffix,
                    [middle_tok_id],
                    middle,
                ]
            )

    else:
        # don't do FIM preproc
        new_sample = sample[: sample.shape[0] - truncate_or_pad]

    return new_sample, np_rng
