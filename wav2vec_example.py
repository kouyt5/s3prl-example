import numpy as np
import torch
import os
from s3prl_updream.interfaces import Featurizer
from s3prl_updream.wav2vec.wav2vec2 import UpstreamExpert
from tqdm import tqdm
import json
import soundfile as sf
import pickle


def convert(source_path:str, target_path:str, model:torch.nn.Module, device=torch.device("cpu")):
    """离线提取特征示例

    Args:
        source_path (str): 源音频
        target_path (str): 目标文件夹
        model (torch.nn.Module): 自监督模型
        device ([type], optional): 设备. Defaults to torch.device("cpu").
    """
    name = source_path.split('/')[-1].replace('.wav', '.pkl')
    mean_path = os.path.join(target_path, 'mean', name)
    last_path = os.path.join(target_path, 'last', name)
    data, sr = sf.read(source_path)
    data = (data - np.mean(data))/np.sqrt((np.var(data)+1e-5))
    data = torch.as_tensor(data, dtype=torch.float32).to(device)
    hidden_states = model([data])["hidden_states"]
    all_hidden_states = torch.zeros((len(hidden_states), *(hidden_states[0].size())),dtype=torch.float32)
    for i in range(len(hidden_states)):
        all_hidden_states[i].copy_(hidden_states[i])
    mean_hidden_state = all_hidden_states.mean(dim=0)
    last_hidden_state = hidden_states[-1]
    mean_hidden_state = mean_hidden_state.detach().cpu().numpy()
    last_hidden_state = last_hidden_state.detach().cpu().numpy()

    pickle.dump(mean_hidden_state, open(mean_path, 'wb'), 1)
    pickle.dump(last_hidden_state, open(last_path, 'wb'), 1)

if __name__ == "__main__":
    # https://dl.fbaipublicfiles.com/fairseq/wav2vec/xlsr_53_56k.pt
    # https://github.com/pytorch/fairseq/tree/main/examples/wav2vec
    wav2vec2_ckpt = "./checkpoints/checkpoint_best.pt"  # 基于fairseq训练的wav2vec2模型文件
    model = UpstreamExpert(ckpt=wav2vec2_ckpt)
    device = torch.device("cpu")
    # model = model.to(device)
    # model.eval()
    # source_wav = "./examples/aishell-dev-example.wav"
    # convert(source_wav, "./examples/", model, device)
    featurizer = Featurizer(
        upstream=model,
        feature_selection="hidden_states",  # last_hidden_state, hidden_state_{0-24}
        upstream_device=device,
        layer_selection=None  # 选择后的第几层特征 0-24
    )
    featurizer2 = Featurizer(
        upstream=model,
        feature_selection="last_hidden_state",  # last_hidden_state, hidden_state_{0-24}
        upstream_device=device,
        layer_selection=None  # 选择后的第几层特征 0-24
    )
    paired_features = model([torch.randn((16000,))])
    feature = featurizer([torch.randn((16000,))], paired_features)  # 提取隐藏层的加权和特征
    feature2 = featurizer2([torch.randn((16000,))], paired_features)  # 提取最后一层特征
    print()
