import numpy as np


def compute_average_loss(loss_value, num_interation, model_list):
    num_model = len(model_list)
    avg_list, max_list, min_list, max_len = [], [], [], []

    for model_idx in range(num_model):
        # 找出当前模型中最长的loss或norm列表的长度
        max_length = max(len(subitem) for subitem in loss_value[model_idx])

        # 将每个实验中的loss列表填充到相同长度
        # 填充使用的是edge，是将实验的loss or norm列表将使用其最后一个值进行填充，而不是使用0
        loss_or_norm = [np.pad(trial, (0, max_length - len(trial)), 'edge') for trial in loss_value[model_idx]]
        # 将填充后的列表转换为NumPy数组，方便后续np.mean, np.max, np.min
        loss_or_norm = np.array(loss_or_norm)

        # 对应取出来的数组为一个二维数组，用axis=0沿着每一列进行操作
        avg_list.append(np.mean(loss_or_norm, axis=0).tolist())
        max_list.append(np.max(loss_or_norm, axis=0).tolist())
        min_list.append(np.min(loss_or_norm, axis=0).tolist())

        # 找出最长的epoch列表
        longest_epoch = max(num_interation[model_idx], key=len)
        max_len.append(longest_epoch)

    return avg_list, max_list, min_list, max_len