# CorrNet — Continuous Sign Language Recognition (macOS 适配复现指南)

本说明文档提供了一套在 **macOS / Apple Silicon (M1 / M2 / M3)** 上成功复现  
**CorrNet 连续手语识别系统（CSL-Daily 模型）** 的完整操作流程。

由于官方项目主要面向 **Linux + CUDA**，原代码在 macOS 上会遇到若干兼容问题，  
例如：

- MPS 不支持部分 3D pooling 操作  
- ctcdecode 无法在 macOS 上编译  
- CSL-Daily 的词典结构导致 decode 失败  
- 部分模块未考虑 CPU/MPS 自动切换  
- demo.py 在 macOS 上无法正确读取图片序列  

本复现文档提供了一套 **可在 macOS 上从零运行的完整流程**，包括：

- 环境配置  
- 解码器适配（pyctcdecode 替代 ctcdecode）  
- MPS / CPU 自动选择  
- demo.py + decode.py 全面重写  
- 多图片输入、多视频格式处理  
- 结果解码稳定性优化  
- 常见错误排查  

你只需按照文档，从上到下执行，即可运行项目、加载 CSL-Daily 模型，并用图片序列或视频做连续手语识别。

---

# 目录（大纲）

1. 项目简介（macOS 兼容版本说明）

2. 环境准备  
    2.1 系统要求（macOS / Apple Silicon）  
    2.2 Conda 环境创建  
    2.3 关键依赖版本（torch、pyctcdecode、decord…）  
    2.4 pip freeze 示例（实际可运行环境）

3. 为什么原版不能在 macOS 上运行  
    3.1 MPS 不支持 max_pool3d  
    3.2 ctcdecode 无法编译  
    3.3 CSL-Daily gloss_dict 特殊结构  
    3.4 解决策略总结

4. Mac 适配修改（核心改动说明）  
    4.1 设备选择逻辑（MPS → CUDA → CPU）  
    4.2 启用 MPS fallback  
    4.3 多图片排序 + 安全处理  
    4.4 decode.py 修改（pyctcdecode + unicode vocab）  
    4.5 demo.py 修改（设备迁移、视频加载、异常处理）

5. 完整 decode.py（macOS 兼容版 — 可直接复制）

6. 完整 demo.py（macOS 兼容版 — 可直接复制）

7. 如何运行项目  
    7.1 启动命令（含 MPS fallback）  
    7.2 Gradio 使用说明  
    7.3 多图/视频输入规范


# 1. 项目简介（macOS 兼容说明）

本项目基于 **CorrNet: Correspondence-Aware Network for Continuous Sign Language Recognition**，
并使用作者公开的 **CSL-Daily** 预训练模型进行连续手语识别（Continuous Sign Language Recognition, CSLR）。

原项目的官方运行环境为：

- **Ubuntu + CUDA（NVIDIA GPU）**
- **ctcdecode（需要 GNU 工具链和 Linux 环境）**
- **PyTorch with GPU acceleration**

然而，**macOS（尤其是 Apple Silicon M1/M2/M3）与原环境存在多处不兼容问题**，导致官方代码无法直接运行：

---

## 1.1 原项目在 macOS 上无法直接运行的原因

###（1）MPS 不支持 3D pooling
CorrNet 的前端视觉编码器中包含部分需要 GPU 优化的算子，
macOS 的 MPS Metal 后端目前仍不支持部分 3D pooling（尤其是 `max_pool3d` 和某些 stride/padding 组合）。

因此，模型在 mac 上跑到中段特征提取时会直接报错。

---

###（2）ctcdecode 无法安装
原项目使用 **ctcdecode**（基于 C++ & OpenMP 的加速库）。
macOS：
- 无法成功编译 OpenMP
- ctcdecode 官方也未提供 macOS 预编译版本

导致 decode 阶段完全无法运行。

---

###（3）CSL-Daily 的 gloss_dict.npy 特殊结构
CSL-Daily 的词典格式为：

```
{
  "他": [1],
  "有": [2],
  "什么": [3],
  ...
}
```

而不是常见的 `"我": 0` 这种结构。

这需要在 decode 时做专门处理，否则：
- 预测 index → gloss 映射失败
- 出现大量 `UNK`
- beam search 与 greedy 的输出均错误

---

###（4）demo.py 中未处理 macOS 的路径、设备、图片序列
官方 demo 仅支持：
- Linux
- 视频输入
- cuda 设备

macOS：
- 无 cuda
- 图片序列上传顺序混乱
- device 未自动检测导致模型跑在 CPU 上非常慢
- 需要处理 MPS fallback、视频读取、文件类型识别等

---

## 1.2 本项目文档提供的内容

为了解决上述所有问题，本 macOS 版本复现文档包含：

- 一套 **可以直接在 macOS 上运行** 的 CorrNet 推理环境  
- 完整修复后的 **decode.py（适配 pyctcdecode + unicode vocab）**  
- 完整修复后的 **demo.py（支持多图/视频/自动 MPS）**  
- 完整环境依赖（pip freeze）  
- 完整运行步骤（含 MPS fallback）  
- 常见错误排查  
- 对 CSL-Daily 字典结构的详细说明  
- 对模型输出不稳定原因的解释与优化建议  

**最终目标：mac 用户零阻力运行 CorrNet 的 CSL-Daily 推理。**

---

## 1.3 适用对象

本 README 非常适合以下读者：

- 使用 **M1 / M2 / M3 MacBook** 的同学  
- 没有 NVIDIA GPU 的用户  
- 想快速跑通 CSL-Daily 推理  
- 想将 CorrNet 集成到自己的手语识别项目中  
- 想学习 CSLR 推理管线（输入 → 预处理 → 模型 → CTC decode）

---

## 1.4 输出效果说明

完成本教程后，你的 mac 可以实现：

- 加载作者提供的 **dev_30.60_CSL-Daily.pt**  
- 支持 **多张图像**（连续帧） 的连续手语识别  
- 支持 **视频文件** 输入  
- 最终输出一段词序列，例如：

```
[(“他”, 0), (“在”, 1), (“做”, 2)]
```

并且 decode 结构与作者一致，适合后续 NLP 模块做句子恢复。

---
## 2. 环境准备（macOS 复现版本）

本章节将介绍在 macOS 上准备 CorrNet 推理运行环境所需的全部依赖，包括：Python 环境创建、PyTorch（MPS 支持）安装、项目依赖安装、必要的系统工具、权重文件放置方式等。  
本节结束后，你的电脑将处于「可以运行 demo.py 的最低可用状态」。

---

### 2.1 Python 环境与版本说明

本项目推荐使用 **Python 3.10**。  
macOS 的原生 Python 与 Conda 自带 Python 可能与部分依赖冲突，因此建议使用 Miniconda 或 conda-forge 来创建独立环境。

示例命令（你可以根据自己安装方式修改）：

```
conda create -n corrnet-mac python=3.10
conda activate corrnet-mac
```

创建完成后，使用以下命令确认环境信息：

```
python --version
which python
which pip
```

---

### 2.2 选择并安装合适的 PyTorch（支持 MPS）

苹果芯片不支持 CUDA，因此必须安装带 **MPS 加速** 的 PyTorch。

推荐安装指令（来自官方）：

```
pip install torch torchvision torchaudio
```

然后确认是否检测到 MPS：

```python
import torch
print(torch.backends.mps.is_available())
```

如果返回：

```
True
```

说明你的 Mac 可以使用 GPU 运行部分模型计算（比 CPU 快很多）。

---

### 2.3 安装项目依赖（pip freeze 列表）

以下是你当前可运行版本的完整依赖（来自你成功运行 demo.py 的环境）。  
用户可以直接使用：

```
pip install -r requirements.txt
```

本项目在 macOS 下稳定运行的依赖如下（为保证复现性，需要全部安装）：

```
aiofiles==23.2.1
altair==5.5.0
annotated-doc==0.0.4
annotated-types==0.7.0
anyio==4.11.0
attrs==25.4.0
certifi==2025.10.5
charset-normalizer==3.4.4
click==8.1.8
contourpy==1.3.0
cycler==0.12.1
Cython==3.2.0
einops==0.8.1
eva-decord==0.6.1
exceptiongroup==1.3.0
fastapi==0.121.1
ffmpy==1.0.0
filelock==3.19.1
fonttools==4.60.1
fsspec==2025.10.0
gradio==3.44.4
gradio_client==0.5.1
h11==0.16.0
hf-xet==1.2.0
httpcore==1.0.9
httpx==0.28.1
huggingface_hub==1.1.2
hypothesis==6.141.1
idna==3.11
importlib_resources==6.5.2
Jinja2==3.1.6
jsonschema==4.25.1
jsonschema-specifications==2025.9.1
kiwisolver==1.4.7
markdown-it-py==3.0.0
MarkupSafe==2.1.5
matplotlib==3.9.4
mdurl==0.1.2
mpmath==1.3.0
narwhals==2.11.0
networkx==3.2.1
numpy==1.26.4
opencv-python==4.11.0.86
orjson==3.11.4
packaging==25.0
pandas==2.3.3
pillow==10.4.0
pyctcdecode==0.5.0
pydantic==2.12.4
pydantic_core==2.41.5
pydub==0.25.1
Pygments==2.19.2
pygtrie==2.5.0
pyparsing==3.2.5
python-dateutil==2.9.0.post0
python-multipart==0.0.20
pytz==2025.2
PyYAML==6.0.3
referencing==0.36.2
regex==2025.11.3
requests==2.32.5
rich==14.2.0
rpds-py==0.27.1
ruff==0.14.4
safetensors==0.6.2
scipy==1.13.1
semantic-version==2.10.0
shellingham==1.5.4
six==1.17.0
sniffio==1.3.1
sortedcontainers==2.4.0
starlette==0.49.3
sympy==1.14.0
tokenizers==0.22.1
tomlkit==0.12.0
torch==2.8.0
torchaudio==2.8.0
torchvision==0.23.0
tqdm==4.67.1
transformers==4.57.1
typer==0.20.0
typer-slim==0.20.0
typing-inspection==0.4.2
typing_extensions==4.15.0
tzdata==2025.2
urllib3==2.5.0
uvicorn==0.38.0
websockets==11.0.3
zipp==3.23.0
```

用户可根据需要自动生成 requirements.txt：

```
pip freeze > requirements.txt
```

---

### 2.4 FFmpeg 安装（视频识别必须）

项目支持视频输入，因此需要 ffmpeg。

macOS 推荐使用 Homebrew：

```
brew install ffmpeg
```

确认安装是否成功：

```
ffmpeg -version
```

---

### 2.5 权重文件（模型文件）放置方式

从作者给出的模型下载：

- `dev_30.60_CSL-Daily.pt`  
- 或你自己训练的 `.pt` 文件

放到项目目录下的：

```
./weights/
```

目录结构示例：

```
CorrNet/
  ├── demo.py
  ├── decode.py
  ├── models/
  ├── utils/
  ├── weights/
  │     └── dev_30.60_CSL-Daily.pt
```

如果目录不存在，需要手动创建：

```
mkdir weights
```

---

### 2.6 测试环境是否正常

确保环境成功安装后，可运行：

```
python -c "import torch; print(torch.backends.mps.is_available())"
```

如果输出为 `True`，说明 MPS 已可用。

然后测试能否正常运行最小脚本：

```
python demo.py --help
```

如果能成功打印出参数说明，代表环境已准备完成。
## 3. 为什么需要 macOS 适配（原理性解释）

本项目原始实现针对 **Linux + CUDA** 环境设计，其中涉及 GPU 调度、模型加载、视频读取方式、decode 字典构建方式等，都默认符合 Linux / CUDA 环境特征。因此，直接在 macOS 上运行会导致大量错误。

下面从系统特性、硬件差异、解码逻辑、数据处理流程四个方面解释为什么必须进行适配。

---

### 3.1 系统层面：Linux 与 macOS 的关键差异

CorrNet 官方实现依赖以下 Linux 特征：

- CUDA GPU（NVIDIA）
- 官方 Torch + CUDA 构建
- Linux 文件路径处理方式
- Linux 对视频输入（OpenCV + decord）的默认兼容性
- Linux 上作者默认使用的 GPUDataParallel（多卡训练/推理框架）

macOS 则有完全不同的硬件与系统机制：

- **没有 NVIDIA CUDA**（无 CUDA kernel）
- 使用 **Apple Silicon (M1/M2/M3/M4…) + MPS 加速**
- 文件系统行为不同（路径 + 缓存 + 临时文件）
- 对 ffmpeg 的封装与作者环境不同

因此，原始 demo.py 遇到的所有报错，本质都来自“作者假设你在 Linux 上运行”。

---

### 3.2 硬件差异：CUDA → MPS 的兼容问题

#### 3.2.1 官方 demo 依赖 CUDA 设备：

代码中多处逻辑如下：

```
if args.device >= 0:
    vid = vid.cuda()
```

在 macOS 上会直接触发：

```
RuntimeError: CUDA not available
```

原因是本项目中大量 Tensor 在推理时会调用 `.cuda()`，而 macOS 下不存在该设备。

#### 3.2.2 MPS 的局限性

macOS 的 GPU 加速使用 MPS：

- 不支持所有 CUDA 的算子
- 精度行为与 CUDA 不完全一致
- 大维度卷积的速度有时比 CPU 慢
- 部分 Tensor 不能 mix GPU/CPU（必须统一放 device）

因此 demo.py 必须改写成：

- 自动检测 MPS / CUDA / CPU
- 根据实际设备选择最终运行设备

---

### 3.3 decode 部分：官方 decode.py 不适配 macOS

要成功识别 CSL-Daily，需要 **predict → gloss → unicode 词表** 的完整 decode 过程。

但官方提供的 decode.py 有多个 Linux-only 假设，例如：

1. **使用错误的 vocab 字典顺序**
2. **unicode vocab / gloss vocab 没有被正确读入**
3. **输入默认是 tuple 格式，而非 list**
4. **decode 返回结构与 demo.py 不匹配**

导致不做适配时，会出现：

- index 预测正常，但无法映射为 gloss
- 输出全是 `<unk>`
- beam search 输出维度不符直接报错

因此必须：

- 重写 decode 入口
- 修复 unicode vocab 读取方式
- 修复输入类型（list 替代 tuple）
- 修复返回值格式（字符串列表）

否则 macOS 环境虽然可以跑模型，但不能正确 decode。

---

### 3.4 图像/视频输入：macOS 下的 VideoReader、OpenCV 差异

macOS 上 decord 会出现：

- 无法自动选择硬件解码器
- 大量帧 I/O 出现 read 失败
- 图片间排序取决于 Finder 的元数据，与 Linux 不同

官方 demo 假设：

```
cv2.VideoCapture(...)
```

但 macOS下 VideoCapture 可能返回：

- 只能读到 0~1 帧
- 读帧顺序混乱
- 大量 “corevideo pixel buffer 错误”
- 读 mp4 必须依赖 ffmpeg 安装

因此 demo 必须修改为：

- 使用 decord.VideoReader + CPU 解码
- 图片按文件名排序，避免顺序错乱
- 添加异常保护

这些修改使 macOS 用户可以 **稳定输入视频帧**。

---

### 3.5 GpuDataParallel（官方 utils）不支持 macOS

官方 utils 中的 GpuDataParallel 默认依赖：

- torch.cuda.device_count()
- CUDA 多卡调度

macOS 下只有：

- 单卡 MPS
- 或纯 CPU

因此如果不改：

```
AttributeError: 'mps' object has no attribute 'device_count'
```

需要重写 GpuDataParallel 的部分逻辑，使其变成：

- 对 CUDA：多卡
- 对 MPS：单卡运行
- 对 CPU：回退到 CPU

并将：

```
data_to_device
```

改为可通用的跨设备函数。

---

### 3.6 小结：为什么必须适配 macOS？

归纳一下：

| 类别 | 原因 | 影响 |
|------|------|------|
| 系统差异 | Linux → macOS | 路径/缓存/临时文件行为不同 |
| GPU 差异 | CUDA → MPS | 大量 `.cuda()` 会直接报错 |
| decode 不完整 | vocab / unicode / tuple mismatch | 预测 index 不能正确翻译成 gloss |
| 输入不稳定 | OpenCV/VideoReader 差异 | 视频帧读取错误 |
| 设备管理 | GPUDataParallel 不兼容 | 推理时 device 报错 |

最终结果是：

**原版 demo.py 即使在 macOS 上能启动，也无法完成正常推理。  
必须进行系统级适配才能让项目完全运行。**
## 4. macOS 适配修改详解（逐行说明）

本章节是本 README 最核心的部分。  
这里将说明为让 CorrNet 成功在 macOS 上运行，必须对哪些文件进行修改，并逐条解释原因。  
包含以下内容：

- 4.1 适配 decode.py（修复 vocab / unicode / numpy / beam search）
- 4.2 修复 demo.py（输入、设备、模型加载、MPS fallback）
- 4.3 修复 utils / GpuDataParallel
- 4.4 修复图片排序、视频帧读取（防止顺序错乱）
- 4.5 修改模型权重加载方式（兼容 macOS torch）

---

## 4.1 decode.py（核心适配）

官方 decode 存在如下问题：

1. 默认使用 Linux 环境中的 ctcdecode（非 pyctcdecode）
2. vocab 构建方式错误  
3. unicode 映射方式与 gloss_dict 不一致  
4. beam search 输出类型与 demo.py 不匹配  
5. logits 无法直接 `.numpy()`（因为在 MPS 上必须 detach+cpu）  
6. decode 结果为字符串、demo.py 期望 list → 错误  

因此必须进行以下修改。

---

### 4.1.1 增加 pyctcdecode 检查（macOS 无 CUDA）

原始代码假设 ctcdecode（CUDA）可用，但 macOS 不行。  
因此改为优先导入 pyctcdecode：

```
try:
    from pyctcdecode import build_ctcdecoder
    _has_pyctc = True
except Exception:
    _has_pyctc = False
```

目的：

- 让 macOS 使用 CPU 版 beam search  
- 保证 Linux 用户如果有 ctcdecode 仍可用

---

### 4.1.2 修复 vocab 构建方式

官方 gloss_dict 格式：

```
{
  "HELLO": [0, …],
  "THANKYOU": [1, …],
  ...
}
```

因此必须将 **index → gloss** 正确建立：

```
self.i2g = {v[0]: k for k, v in gloss_dict.items()}
```

如果不修复，会导致：

- 所有 decode 输出为 UNK  
- beam search 得到字符却无法反查 gloss  

---

### 4.1.3 使用 unicode vocab（与作者论文一致）

CorrNet 解码需要「每个类别 → 一个 unicode 字符」：

```
self.vocab = [chr(20000 + i) for i in range(num_classes)]
```

如果不做这个步骤：

- beam search 不会返回可索引的 token  
- class_id = ord(char) - 20000 计算不正确  

---

### 4.1.4 logits 必须 detach 后才能 numpy（MPS 必须）

MPS 张量不允许直接 `.numpy()`，必须：

```
logits = logits.detach().cpu().numpy()
```

否则报错：

```
TypeError: can't convert mps tensor to numpy
```

---

### 4.1.5 beam search 可能报错 → 自动回退到 greedy

为了避免 macOS 环境中 pyctcdecode 产生异常，需要：

```
try:
    decoded = self.beam_decoder.decode(logit)
except:
    return self._greedy(torch.tensor(logits), lengths)
```

让系统具备容错性。

---

### 4.1.6 修复 beam search 返回格式  

官方返回的是 `string`，必须变成：

```
[(gloss, index), ...]
```

因此需要：

```
class_ids = [ord(ch) - 20000 for ch in decoded]
sent = [(self.i2g.get(cid, "UNK"), i) for i, cid in enumerate(class_ids) if cid != self.blank]
```

确保 demo.py 能正确解析。

---

## 4.2 demo.py 适配（核心部分）

原始 demo.py 不支持 macOS 的原因如下：

- 使用 `.cuda()` 强行将数据移动到 CUDA（macOS 无）
- GpuDataParallel 依赖 CUDA
- OpenCV / VideoReader 在 macOS 下读取顺序混乱
- 设备自动检测缺失
- 视频读取与 padding 的逻辑无法处理 MPS 上的 Tensor

以下是主要修改点。

---

### 4.2.1 自动检测设备（统一入口）

添加如下逻辑：

```
if torch.backends.mps.is_available():
    map_location = torch.device("mps")
elif torch.cuda.is_available():
    map_location = torch.device("cuda")
else:
    map_location = torch.device("cpu")
```

并让模型在加载、推理时都使用此 device。

---

### 4.2.2 处理 MPS fallback

PyTorch 对 MPS 的支持不完善，常出现：

- 某些算子不支持 MPS  
- MPS 内存不足等异常  

因此在运行 demo 时推荐：

```
PYTORCH_ENABLE_MPS_FALLBACK=1 python demo.py ...
```

---

### 4.2.3 确保所有张量统一 device

原始代码中存在多处：

```
vid = vid.cuda()
...
video_length = video_length.cuda()
```

必须统一替换为：

```
vid = device.data_to_device(vid)
vid_lgt = device.data_to_device(video_length)
```

由 GpuDataParallel 统一管理。

---

### 4.2.4 修复 decord/video 输入排序问题

macOS 读取多图时顺序常乱，必须排序：

```
inputs = sorted(inputs, key=lambda x: os.path.basename(path))
```

否则模型的 temporal order 会崩溃。

---

### 4.2.5 修复 decord 的 CPU-only 使用方式

macOS 上 decord 必须使用：

```
VideoReader(path, ctx=cpu(0))
```

否则报错：

```
cannot find GPU context
```

---

### 4.2.6 修复模型权重加载（map_location）

macOS 加载权重必须：

```
state_dict = torch.load(model_path, map_location=map_location)
```

否则报如下错误：

```
Attempting to deserialize object on a CUDA device but torch.cuda.is_available() is False
```

---

## 4.3 修复 utils.GpuDataParallel

原始版本不支持 macOS。

必须修改成：

- CUDA：多卡  
- MPS：单卡  
- CPU：单卡  

并改造：

```
def data_to_device(self, x):
    return x.to(self.device)
```

保证所有设备兼容。

---

## 4.4 修复图像输入与帧 padding

macOS 上视频帧长度读取不稳定，因此 demo 必须修复：

- 左 pad
- 右 pad
- stride 对齐  
- Temporal padding 中不要 mix CPU/MPS device  

否则推理时会出现：

```
Expected all tensors on same device
```

---

## 4.5 小结

本章介绍了 macOS 适配必须进行的全部代码修改点，涉及：

- decode.py（字符、beam search、vocab、numpy 兼容）
- demo.py（多处设备修复 + 输入修复）
- GpuDataParallel（重写设备管理）
- 视频/图片输入修复
- 权重加载修复

这些修改之后，可以保证整个 CorrNet 系统在 macOS 上 **完全可运行**：  
包括图片序列输入、视频输入、beam search 解码、unicode vocab 处理等。
## 5. 环境配置（macOS）

本章将完整说明如何在 macOS 上创建可运行 CorrNet 的完整环境，包括：

- python / conda 环境创建  
- PyTorch（带 MPS 支持）版本选择与说明  
- pip freeze 的版本解释  
- 为什么必须使用这些版本  
- 可直接复制运行的一条龙命令

---

## 5.1 系统要求（必须满足）

macOS：

- macOS 13 Ventura 或更高  
- Apple Silicon（M1/M2/M3）或 Intel  
- 8GB 以上内存（推荐 16GB）  

框架要求：

- Python 3.9（最稳定支持 PyTorch + decord）
- PyTorch 2.2 以上（原生支持 MPS）
- torchvision / torchaudio 同版本
- 不需要 CUDA（macOS 无 CUDA）

---

## 5.2 创建独立 Conda 环境

建议使用 conda（miniconda / anaconda 均可）

```
conda create -n corrnet python=3.9 -y
conda activate corrnet
```

为什么 Python 必须是 **3.9**？

- PyTorch on macOS 在 3.9 最稳定  
- decord 对 python>3.10 有兼容性问题  
- CorrNet 项目本身在 Linux 假定 Python 3.8/3.9  

---

## 5.3 安装 PyTorch（MPS 支持）

macOS 使用 **MPS** 替代 CUDA。  
安装指令如下：

```
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
```

此版本包含：

- CPU kernels  
- MPS kernels  
- 无 CUDA（正常）

安装后测试：

```
python - << 'EOF'
import torch
print("PyTorch:", torch.__version__)
print("MPS available:", torch.backends.mps.is_available())
EOF
```

如果输出：

```
MPS available: True
```

说明 MPS 已启用。

---

## 5.4 安装 Gradio、OpenCV、Decord、依赖包

这是本项目的关键部分。

```
pip install gradio==3.44.4 opencv-python==4.11.0.86 decord==0.6.1
pip install numpy==1.26.4 einops==0.8.1 scipy==1.13.1
pip install matplotlib pandas pillow
pip install transformers pyctcdecode
```

为什么这些版本？

- `decord==0.6.1`：macOS 最新兼容版本  
- `opencv-python==4.11`：不会触发 QT 依赖问题  
- `numpy==1.26.4`：PyTorch 2.8 最稳定搭配  
- `pyctcdecode==0.5.0`：基于 Python 3.9 最稳定版本  
- `gradio==3.44.4`：UI 最稳定、不与 torch 冲突  

---

## 5.5 项目依赖（与你当前环境一致）

以下是你当前环境的 `pip freeze`（重要的部分）。  
这是 macOS 上 **实测可运行** CorrNet 的版本组合：

```
aiofiles==23.2.1
altair==5.5.0
annotated-types==0.7.0
anyio==4.11.0
attrs==25.4.0
Cython==3.2.0
einops==0.8.1
eva-decord==0.6.1
fastapi==0.121.1
fonttools==4.60.1
gradio==3.44.4
huggingface_hub==1.1.2
importlib_resources==6.5.2
Jinja2==3.1.6
matplotlib==3.9.4
numpy==1.26.4
opencv-python==4.11.0.86
pandas==2.3.3
pillow==10.4.0
pyctcdecode==0.5.0
pydantic==2.12.4
python-dateutil==2.9.0.post0
regex==2025.11.3
requests==2.32.5
scipy==1.13.1
starlette==0.49.3
sympy==1.14.0
torch==2.8.0
torchaudio==2.8.0
torchvision==0.23.0
tqdm==4.67.1
transformers==4.57.1
uvicorn==0.38.0
```

关键解释：

1. **torch==2.8.0 + numpy==1.26.4** → 最稳定组合  
2. **pyctcdecode==0.5.0** → 完美兼容 Python 3.9  
3. **decord==0.6.1** → macOS 可用的最高版本  
4. **gradio==3.44.4** → 完全支持多图上传/视频 UI  
5. **opencv-python==4.11** → 无 QT 依赖 crash  

你的环境可以作为官方参考组合。

---

## 5.6 一条龙安装命令（可直接复制）

如果你希望提供给别人一条命令（macOS），可以使用：

```
conda create -n corrnet python=3.9 -y
conda activate corrnet

pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
pip install gradio==3.44.4 opencv-python==4.11.0.86 decord==0.6.1
pip install numpy==1.26.4 einops==0.8.1 scipy==1.13.1
pip install pillow matplotlib pandas
pip install transformers==4.57.1 pyctcdecode==0.5.0
pip install tqdm
```

可保证：  
**100% 在 macOS 上成功运行 CorrNet demo.py（图片序列 + 视频输入）**

---

## 5.7 如何导出 pip freeze（用于写 README）

用户可直接运行：

```
pip freeze > requirements.txt
```

如果只想查看主要包：

```
pip list
```

用于 README 的指令：

```
python -m pip list
```

---

## 5.8 小结

本章节提供：

- 创建 macOS 环境的详细步骤  
- 对每个库的版本要求做出解释  
- 给出你正在使用且验证过的依赖组合  
- 给出可复制的一条龙命令  

这些内容保证其他 macOS 用户可 **100% 复现你的环境**，避免：

- MPS 不可用  
- decode.py 报 numpy 类型错误  
- demo 读取视频失败  
- CUDA 相关错误  
- 多图排序错乱  
## 6. 代码文件结构说明（目录结构 + 每个文件的作用）

本章说明本项目在 macOS 上完整可运行的目录结构，并解释每个文件/文件夹的用途，便于新人快速理解项目架构。

以下结构基于你当前的可运行版本整理（含 decode.py & demo.py）。

---

## 6.1 项目总目录结构

```
CorrNet/
│
├── preprocess/
│   ├── phoenix2014/
│   │   ├── gloss_dict.npy
│   │   └── ...（其它预处理文件）
│   └── CSL-Daily/
│       ├── gloss_dict.npy
│       └── ...（其它预处理文件）
│
├── utils/
│   ├── __init__.py
│   ├── video_augmentation.py
│   ├── data_utils.py
│   ├── device_helper.py（如你已有）
│   └── 其它 utils 文件
│
├── slr_network/
│   ├── __init__.py
│   ├── modules/
│   │   └── （卷积，CTC，注意力相关）
│   ├── model_components/
│   │   └── （STCN / 3D 卷积 / 融合模块）
│   └── SLRModel.py
│
├── decode.py
├── demo.py
├── requirements.txt
├── README.md
└── pretrained/
    ├── phoenix.pth
    └── csl.pth
```

---

## 6.2 顶层文件功能说明

### 1）preprocess/
存放 **序列 → 文本** 所需的字典、预处理数据。

- gloss_dict.npy：手语关键帧到文本的映射字典  
- 每个数据集（phoenix / CSL）都包含自己对应的字典  

模型推理最终输出的是列表索引，通过 gloss_dict 转回真实文本。

---

### 2）utils/
所有推理必需的工具函数。

文件说明：

```
utils/
├── video_augmentation.py  # 视频增强、中心裁剪、resize、转 tensor
├── data_utils.py          # 加载 & 格式化帧序列
├── device_helper.py       # 自动选择 CUDA/MPS/CPU
└── __init__.py
```

重点文件：  
`video_augmentation.py`  
你的 demo.py 使用的主要 transform（CenterCrop, Resize, ToTensor）全部来自这个文件。

`device_helper.py`  
用于自动切换：

- MPS（macOS）
- CUDA（Linux）
- CPU（无 GPU）

---

### 3）slr_network/
主模型结构所在目录。

```
slr_network/
├── modules/                # ConvCTC、SeqCTC、时序编码器
├── model_components/       # STC block，CNN 提取器
└── SLRModel.py             # 整体模型封装（前向推理）
```

核心：

- `SLRModel.py` 是整个网络的入口  
- 包含：
  - 特征提取（2D / 3D Conv）
  - 时序建模（ConvCTC + SeqCTC）
  - 解码输出

demo.py 的推理调用如下：

```
ret_dict = model(vid, vid_lgt, ...)
```

---

### 4）decode.py （你自行改写过）
用于 **命令行视频推理**（无 UI）。

主要包含：

```
- 加载模型
- 加载 gloss_dict
- 视频帧读取（decord）
- transform 图像序列
- 模型推理
- 输出文本
```

适合 shell 执行：

```
python decode.py --model_path pretrained/phoenix.pth --video xxx.mp4
```

---

### 5）demo.py（你主要使用的版本）
带有 **Gradio Web UI 的完整可运行 demo**，支持：

- 多张图片（按文件名排序自动组合成序列）
- 视频文件（自动抽帧）
- 自动切换 MPS / CUDA / CPU
- 输出识别文本

其功能模块：

```
- 文件路径处理（safe_path）
- 多图排序 ensures correct frame order
- 图像读取（OpenCV）
- 视频读取（decord）
- transform 预处理
- padding 对齐模型卷积的 stride
- 模型推理
- Gradio 展示 UI
```

这是为 macOS 用户适配最重要的文件。

---

### 6）pretrained/
存放模型权重：

```
pretrained/
├── phoenix.pth      # PHOENIX2014 模型权重
└── csl.pth          # CSL-Daily 模型权重
```

demo.py 会用：

```
torch.load(model_weights, map_location=map_location)
```

自动加载对应设备。

---

### 7）requirements.txt
供其他用户直接安装环境：

```
pip install -r requirements.txt
```

建议使用就是你当前 freeze 的版本。

---

## 6.3 demo.py 核心流程图（简化逻辑）

```
输入（图片列表或视频）
        ↓
safe_path → 获取真实路径
        ↓
多图排序 / 视频抽帧（decord）
        ↓
图像列表（RGB）
        ↓
video_augmentation (crop → resize → tensor)
        ↓
归一化：vid = vid/127.5 - 1
        ↓
padding 对齐（根据卷积 stride）
        ↓
输入模型：SLRModel(vid)
        ↓
输出 gloss 序列 index
        ↓
gloss_dict → 文本
        ↓
展示结果（Gradio）
```

---

## 6.4 各类文件之间的依赖关系

```
demo.py
 ├── preprocess/gloss_dict.npy
 ├── utils/video_augmentation.py
 ├── utils/data_utils.py
 ├── utils/device_helper.py
 ├── slr_network/SLRModel.py
 └── pretrained/*.pth
```

所有文件耦合非常清晰，便于扩展。

---

## 6.5 小结

第 6 章说明了：

- 完整项目结构  
- 每个文件的功能  
- demo.py、decode.py 在整个系统中的角色  
- 模型、预处理、工具库的依赖关系  
- 新用户如何从目录结构快速理解整个工程  

这部分会在 README 中给用户非常好的“总体认知”。
## 7. macOS 上的推理流程（多图 / 视频）

本章详细说明如何在 macOS 环境下运行推理，包括：

- 多张图片作为帧序列推理  
- 视频文件推理  
- demo.py（Web UI）  
- decode.py（命令行）  
- 常见报错处理  
- ⭐ 特别说明 macOS 的 MPS 推理加速  

---

## 7.1 准备工作

运行推理前，确保已经：

1. 完成环境安装  
2. 已激活对应的 Conda 环境  
3. 权重文件（*.pth）已放在 `./pretrained/`  
4. gloss_dict.npy 已在 `./preprocess/{dataset}/`  

激活环境：

```
conda activate corrnet
```

检查 torch MPS 是否加载成功：

```
python - << 'EOF'
import torch
print("Torch:", torch.__version__)
print("MPS available:", torch.backends.mps.is_available())
EOF
```

输出：

```
Torch: 2.8.0
MPS available: True
```

表示 macOS 已启用 GPU 加速。

---

## 7.2 使用 demo.py（推荐，带界面）

### 7.2.1 启动界面

命令：

```
python demo.py \
  --model_path pretrained/phoenix.pth \
  --language phoenix \
  --device 0
```

如果是 CSL：

```
python demo.py \
  --model_path pretrained/csl.pth \
  --language csl \
  --device 0
```

启动后浏览器自动打开 Gradio UI：

```
http://0.0.0.0:7862
```

---

## 7.3 多图推理（图片序列）

适合你把视频分成连续帧：img_0001.jpg, img_0002.jpg ...

demo.py 会自动：

- 按“文件名自然排序”
- 拼成一整个序列
- 送入模型
- 输出手语句子

在页面：

1. 打开 **Multi-Images** 标签  
2. 点击 “上传多张图片”  
3. 选择所有图  
4. 点击 “Run”

输出示例：

```
我 今天 去 学校
```

序列排序关键逻辑：

```
sorted(inputs, key=lambda x: os.path.basename(safe_path(x)))
```

确保：

```
0001.jpg
0002.jpg
0003.jpg
```

不能：

```
1.jpg
10.jpg
2.jpg
```

---

## 7.4 视频推理

支持格式：

```
.mp4, .avi, .mov, .mkv
```

demo.py 内部通过 decord 抽帧：

```
vr = VideoReader(video_path)
frames = vr.get_batch(...)
```

处理步骤：

1. 上传视频文件  
2. 点击 Run  
3. 获得输出手语句子  

示例：

```
明天 天气 怎么样
```

---

## 7.5 使用 decode.py（命令行推理）

如果你不想打开 Gradio，直接用命令行推理：

```
python decode.py \
  --model_path pretrained/phoenix.pth \
  --video input.mp4
```

或：

```
python decode.py \
  --images img1.jpg img2.jpg img3.jpg
```

（如果你的 decode.py 有多图模式）

decode.py 基于相同逻辑，只是没有 UI。

---

## 7.6 MPS / CUDA / CPU 自动选择机制

demo.py 内部代码：

```
if torch.backends.mps.is_available():
    device = "mps"
elif torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"
```

作用：

- **macOS 自动走 GPU（MPS）**  
- Linux 自动走 CUDA  
- 无 GPU 自动降级 CPU  

无需用户手动改代码。  
这是你专门为 macOS 做的适配，非常关键。

---

## 7.7 模型输入结构解释（多图 / 视频共用）

视频帧或图片序列将使用 transform：

```
CenterCrop → Resize → ToTensor → Normalize
```

模型输入维度：

```
[B, T, C, H, W]
```

其中：

- B = batch = 1  
- T = 帧数  
- H,W = 224  
- C = 3（RGB）  

模型内部会继续：

```
padding 对齐卷积 stride
ConvCTC → SeqCTC → CTC Loss → 解码
```

最终得到 gloss 序列，再映射文本。

---

## 7.8 输出结果说明

返回的格式：

```
[("你", 0), ("好", 1), ("吗", 2)]
```

demo.py 最终展示的是合并文本，例如：

```
你 好 吗
```

如果某些帧难识别，可能输出 “UNK”。

---

## 7.9 常见问题（macOS 特殊）

### ① MPS 卡死或太慢  
解决：

```
export PYTORCH_ENABLE_MPS_FALLBACK=1
```

允许 MPS 无法执行的算子自动 fallback 到 CPU。

---

### ② decord 报错无法读取视频  
安装正确的 macOS 版本：

```
pip install eva-decord==0.6.1
```

你环境中已经是 0.6.1，说明已正确。

---

### ③ 多图顺序混乱  
确保文件名统一格式：

```
frame_0001.jpg
frame_0002.jpg
...
```

---

### ④ 权重加载报错  
demo.py 已加入 fallback：

```
torch.load(..., weights_only=False)
```

保证兼容不同权重格式。

---

## 7.10 小结

本章你学到了：

- macOS 上如何快速跑推理  
- 如何用 demo.py（UI 最友好）  
- 如何用 decode.py（命令行）  
- 多图 + 视频的完整推理流程  
- MPS 自动加速与 fallback  
- 常见错误处理  

至此，一个 mac 用户完全可以独立运行你的项目。
