# s3prl-example

<a href="./LICENSE.txt"><img alt="GitHub" src="https://img.shields.io/github/license/kouyt5/rabbit-rpc-client"></a>
[![Language grade: Python](https://img.shields.io/lgtm/grade/python/g/kouyt5/s3prl-example.svg?logo=lgtm&logoWidth=18)](https://lgtm.com/projects/g/kouyt5/s3prl-example/context:python)
[![Total alerts](https://img.shields.io/lgtm/alerts/g/kouyt5/s3prl-example.svg?logo=lgtm&logoWidth=18)](https://lgtm.com/projects/g/kouyt5/s3prl-example/alerts/)

基于 [s3prl](https://github.com/s3prl/s3prl) 的自监督特征提取

## Q&A

**为什么要把s3prl的代码复制过来创建这个仓库，模型还不全？**

s3prl 是一个比较完善的语音自监督模型库，集成了各种前沿的自监督模型，除了自监督模型外，还集成了基于自监督或非自监督的各种下游任务，可以很方便的复现论文中的结果[<sup>1</sup>](#refer-anchor-1)。

在s3prl中如何调用自监督模型呢？请看如下示例:
```python
import s3prl.hub as hub

model_0 = getattr(hub, 'fbank')()  # use classic FBANK
model_1 = getattr(hub, 'modified_cpc')()  # build 

device = 'cuda'  # or cpu
model_3 = model_3.to(device)
wavs = [torch.randn(160000, dtype=torch.float).to(device) for _ in range(16)]
with torch.no_grad():
    reps = model_3(wavs)["hidden_states"]
```
官方库的通病来了，为了各种模块的兼容性与统一，牺牲了部分灵活性，在这里，它使用模块名字符串调用模块，我们就很难直接知道究竟是调用了哪个模块中的算法，很难直接修改原文件，或者调试模型，难以知道模型初始化的细节。大多数情况下，我们会从已有的项目出发，如果直接使用官方库，又会增加依赖。因此，本仓库尝试对s3prl库进行解耦，牺牲通用性，作成专用。

具体的，取消s3prl库的依赖，最小化依赖，只做自监督特征提取的封装。提供特征提取的调用示例，能够更加方便的引入自己的私有项目。

**目前支持哪些特征？**

+ wav2vec2

## 调用方法示例

在根目录下，`{model_name}_example.py` 中有详细使用方法。

例如：
```python
# https://dl.fbaipublicfiles.com/fairseq/wav2vec/xlsr_53_56k.pt
wav2vec2_ckpt = "./checkpoints/checkpoint_best.pt"
model = UpstreamExpert(ckpt=wav2vec2_ckpt)
device = torch.device("cpu")
featurizer = Featurizer(
    upstream=model,
    feature_selection="hidden_states",  # last_hidden_state, hidden_state_{0-24}
    upstream_device=device,
    layer_selection=None  # 选择后的第几层特征 0-24
)
paired_features = model([torch.randn((16000,))])
feature = featurizer([torch.randn((16000,))], paired_features)  # 提取隐藏层的加权和特征
```

<div id="refer-anchor-1"></div>

- [1] [SUPERB: Speech Processing Universal PERformance Benchmark](https://arxiv.org/abs/2105.01051)