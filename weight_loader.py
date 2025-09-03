from ultralytics.utils import LOGGER
import torch


def load_pretrained_weights(model, weights_path):
    """
    自定义权重加载函数，处理新增自定义模块时的权重加载

    Args:
        model: 目标模型
        weights_path: 预训练权重路径

    Returns:
        加载了预训练权重的模型
    """
    # 加载预训练权重
    LOGGER.info(f"正在加载预训练权重: {weights_path}")
    pretrained = torch.load(weights_path, map_location='cpu')

    # 获取预训练权重的state_dict
    if isinstance(pretrained, dict):
        if 'model' in pretrained:
            pretrained = pretrained['model']  # 提取模型权重
        state_dict = pretrained.float().state_dict() if hasattr(pretrained, 'state_dict') else pretrained
    else:
        state_dict = pretrained.float().state_dict()

    # 获取当前模型的state_dict
    model_state_dict = model.state_dict()

    # 创建要加载的新state_dict
    new_state_dict = {}
    skip_layers = []
    transferred = 0

    # 遍历预训练权重的每个键
    for k, v in state_dict.items():
        if k in model_state_dict and model_state_dict[k].shape == v.shape:
            new_state_dict[k] = v
            transferred += 1
        else:
            # 检查键名是否包含自定义模块的名称
            if 'Adjust_Transformer' in k:
                skip_layers.append(k)
            else:
                LOGGER.warning(f"形状不匹配的层: {k}")

    # 加载匹配的权重
    model_state_dict.update(new_state_dict)
    model.load_state_dict(model_state_dict)

    LOGGER.info(f"成功加载 {transferred}/{len(state_dict)} 个预训练权重项")
    if skip_layers:
        LOGGER.info(f"跳过了自定义层: {skip_layers}")

    return model