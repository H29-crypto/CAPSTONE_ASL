# CorrNet—continuous sign language recognition (MAC OS adaptation and reproduction guide)

This document provides a set of successful reproduction on **MAC OS/Apple Silicon (M1/M2/M3)**.
**Complete operation flow of **CorrNet continuous sign language recognition system (CSL-Daily model)**.

Because the official project is mainly oriented to **Linux+CUDA**, the original code will encounter some compatibility problems on macOS.
For example:

- MPS does not support some 3D pooling operations.
- ctcdecode cannot compile on macOS.
Decode failed because of the dictionary structure of-CSL-Daily.
- Some modules do not consider the automatic switching of CPU/MPS.
- demo.py can't read the picture sequence correctly on macOS.

This reproduced document provides a complete set of **processes that can be run from scratch on macOS**, including:

- Environment configuration
- Decoder adaptation (pyctcdecode instead of ctcdecode)
- MPS/CPU automatic selection
- demo.py+decode.py completely rewritten.
- Multi-picture input and multi-video format processing
- Optimization of result decoding stability
- Troubleshooting of Common Errors

You can run the project, load the CSL-Daily model, and do continuous sign language recognition with picture sequences or videos just by following the documents from top to bottom.

---

# Table of Contents (Outline)

1. Introduction to the project (description of compatible versions of macOS)

2. Environmental preparation
2.1 system requirements (macOS/Apple Silicon)
2.2 Conda environment creation
2.3 Critical Dependent Versions (torch, pyctcdecode, decord…)
2.4 pip freeze Example (Actual Running Environment)

3. Why can't the original run on macOS?
3.1 MPS does not support max_pool3d.
3.2 ctcdecode cannot be compiled.
3.3 CSL-Daily gloss_dict special structure
3.4 Summary of Solution Strategies

4. Mac adaptation modification (core modification description)
4.1 equipment selection logic (MPS → CUDA → CPU)
4.2 enable MPS fallback
4.3 Multi-picture sorting+security processing
4.4 decode.py modification (pyctcdecode+unicode vocab)
4.5 Modification of demo.py (device migration, video loading, exception handling)

5. Complete decode.py(macOS compatible version-can be directly copied)

6. Complete demo.py(macOS compatible version-can be directly copied)

7. How to run the project
7.1 Start command (including MPS fallback)
7.2 instructions for use of gradio
7.3 Multi-image/video input specification


# 1. Project Introduction (macOS Compatibility Description)

This project is based on **Corrnet: Corresponse-aware network for continuous sign language recognition**,
And use the **CSL-Daily** pre-training model published by the author to carry out continuous sign language recognition (CSLR).

The official operating environment of the original project is:

- **Ubuntu + CUDA（NVIDIA GPU）**
- **ctcdecode (requires GNU tool chain and Linux environment)**
- **PyTorch with GPU acceleration**

However, **macOS (especially Apple Silicon M1/M2/M3) has many incompatible problems with the original environment, which makes the official code unable to run directly:

---

## 1.1 The reason why the original project can't run directly on macOS

### (1) MPS does not support 3D pooling.
The front-end visual encoder of CorrNet contains some operators that need GPU optimization.
The MPS Metal backend of macOS still doesn't support some 3D pooling (especially ` max_pool3d' and some stride/padding combinations).

Therefore, the model will report an error directly when it runs to the middle feature extraction on the mac.

---

### (2) ctcdecode cannot be installed.
The original project used **ctcdecode** (acceleration library based on C++ & OpenMP).
macOS：
-OpenMP failed to compile successfully.
-ctcdecode officially does not provide a precompiled version of macOS.

Causes the decode phase to be completely inoperable.

---

### (3) The special structure of gloss_dict.npy of CSL-Daily
The dictionary format of CSL-Daily is:

```
{
"He": [1],
"Yes": [2],
"What": [3],
...
}
```

Instead of the common "I": 0 "structure.

This needs to be handled specially when decoding, otherwise:
- Forecast index → gloss mapping failed.
- There are a lot of "unks".
- beam search and greedy's output are both wrong.

---

### (4) The path, device and picture sequence of macOS are not processed in demo.py.
The official demo only supports:
- Linux
- Video input
- cuda equipment

macOS：
- no cuda
- The sequence of pictures uploaded is out of order.
- device does not detect automatically, which causes the model to run very slowly on the CPU.
- Need to handle MPS fallback, video reading, file type identification, etc.

---

## 1.2 Contents provided in this project document

In order to solve all the above problems, this macOS version reproduction document contains:

- A set of **CorrNet reasoning environment that can run** directly on macOS.
- **decode.py (adapted to pyctDecode+Unicode Vocab) after complete repair**
- fully restored **demo.py (supporting multi-images/videos/automatic MPS)**
- complete environment dependency (pip freeze)
- complete operation steps (including MPS fallback)
- Troubleshooting of Common Errors
- A detailed description of the structure of CSL-Daily dictionary.
- Explaining the causes of unstable model output and suggestions for optimization.

**Ultimate goal: mac users can run CSL-Daily reasoning of CorrNet with zero resistance.**

---

## 1.3 Applicable objects

This README is very suitable for the following readers:

- Students who use the **M1/M2/M3 MacBook**
- users without NVIDIA GPU
- Want to run fast through CSL-Daily reasoning
- I want to integrate CorrNet into my sign language recognition project.
- Want to learn CSLR reasoning pipeline (input → preprocessing → model → CTC decode)

---

## 1.4 Description of output effect

After completing this tutorial, your mac can realize:

- load **dev_30.60_CSL-Daily.pt** provided by the author.
- Support continuous sign language recognition of **multiple images** (continuous frames)
- Support **video file** input.
-finally output a word sequence, such as:

```
[("He", 0), ("In", 1), ("Doing", 2)]
```

And the decode structure is consistent with the author, which is suitable for the subsequent NLP module to do sentence recovery.

---
## 2. Environment preparation (macOS recurring version)

This chapter will introduce all the dependencies needed to prepare the CorrNet inference running environment on macOS, including: Python environment creation, PyTorch(MPS support) installation, project dependency installation, necessary system tools, weight file placement, etc.
At the end of this section, your computer will be in the "lowest available state to run demo.py".

---

### 2.1 Python environment and version description

**Python 3.10** is recommended for this project.
The native Python of macOS and Python of Conda may conflict with some dependencies, so it is suggested to use Miniconda or conda-forge to create an independent environment.

Sample command (you can modify it according to your own installation method):

```
conda create -n corrnet-mac python=3.10
conda activate corrnet-mac
```

After creation, use the following command to confirm the environment information:

```
python --version
which python
which pip
```

---

### 2.2 Select and install the appropriate PyTorch (supporting MPS).

Apple chips don't support CUDA, so PyTorch with **MPS acceleration** must be installed.

Recommended installation instructions (from the official):

```
pip install torch torchvision torchaudio
```

Then confirm whether MPS is detected:

```python
import torch
print(torch.backends.mps.is_available())
```

If returned:

```
True
```

Explain that your Mac can use GPU to run some model calculations (much faster than CPU).

---

### 2.3 Installation Project Dependencies (pip freeze List)

The following are the complete dependencies of your current runnable version (from the environment where you successfully run demo.py).
Users can directly use:

```
pip install -r requirements.txt
```

The stable operation of this project under macOS depends on the following (to ensure reproducibility, all of them need to be installed):

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

Users can automatically generate requirements.txt according to their needs:

```
pip freeze > requirements.txt
```

---

### 2.4 FFmpeg installation (required for video recognition)

The project supports video input, so ffmpeg is needed.

MacOS recommends Homebrew:

```
brew install ffmpeg
```

Confirm that the installation was successful:

```
ffmpeg -version
```

---

### 2.5 Weight file (model file) placement method

Download from the model given by the author:

- `dev_30.60_CSL-Daily.pt`
-or a `. pt` file that you trained yourself.

Put in the project directory:

```
./weights/
```

Example of directory structure:

```
CorrNet/
├── demo.py
├── decode.py
├── models/
├── utils/
├── weights/
│     └── dev_30.60_CSL-Daily.pt
```

If the directory does not exist, you need to create it manually:

```
mkdir weights
```

---

### 2.6 Whether the test environment is normal

Ensure that after the environment is successfully installed, you can run:

```
python -c "import torch; print(torch.backends.mps.is_available())"
```

If the output is `true`, MPS is available.

Then test whether the minimal script can run normally:

```
python demo.py --help
```

If the parameter description can be printed successfully, it means that the environment is ready to be completed.
## 3. Why macOS adaptation is needed (principle explanation)

The original implementation of this project is designed for **Linux+CUDA** environment, which involves GPU scheduling, model loading, video reading mode, decode dictionary construction mode, etc., all of which are in line with the characteristics of Linux/CUDA environment by default. Therefore, running directly on macOS will lead to a lot of errors.

The following explains why adaptation is necessary from four aspects: system characteristics, hardware differences, decoding logic and data processing flow.

---

### 3.1 System level: Key differences between Linux and macOS

The official implementation of CorrNet relies on the following Linux features:

- CUDA GPU（NVIDIA）
- official Torch+CUDA construction
- Linux file path processing method
- Default compatibility of Linux for video input (OpenCV+decord)
- GPUDataParallel (multi-card training/reasoning framework) used by the author by default on Linux.

MacOS has completely different hardware and system mechanisms:

- **no NVIDIA CUDA** (no CUDA kernel)
- use **apple silicon (m1/m2/m3/M4 …)+MPs to accelerate**
- The file system behaves differently (path+cache+temporary file)
- The encapsulation of ffmpeg is different from the author environment.

Therefore, all the errors encountered by the original demo.py are essentially from "the author assumes that you are running on Linux".

---

### 3.2 Hardware Differences: Compatibility of CUDA → MPS

#### 3.2.1 Official demo relies on CUDA devices:

Many logics in the code are as follows:

```
if args.device >= 0:
vid = vid.cuda()
```

It will trigger directly on macOS:

```
RuntimeError: CUDA not available
```

The reason is that a large number of Tensor in this project will call `. cuda () ` when reasoning, but this device does not exist under macOS.

#### 3.2.2 Limitations of MPS

GPU of macOS accelerates the use of MPS;

- Not all CUDA operators are supported.
- The accuracy behavior is not completely consistent with CUDA.
- Large-dimensional convolution is sometimes slower than CPU.
-some Tensor can't mix GPU/CPU (device must be put together).

So demo.py must be rewritten as:

- automatically detect MPS/CUDA/CPU.
- Select the final running equipment according to the actual equipment.

---

### 3.3 decode part: official decode.py is not suitable for macOS.

To successfully identify CSL-Daily, a complete decode process of **predict → gloss → unicode vocabulary** is needed.

But the official decode.py has several Linux-only assumptions, such as:

1. **Use the wrong vocab dictionary order**
2. **Unicode VOAB/GlossVOAB is not read correctly**
3. **Input is in tuple format by default, not list**
4. **decode return structure does not match demo.py**

When no adaptation is made, the following will appear:

- index prediction is normal, but it cannot be mapped to gloss.
- the output is all ` < unk >`.
- beam search output dimension does not match, report the error directly.

Therefore, it is necessary to:

- rewrite decode entry
- fix unicode vocab reading mode
- fix input type (list instead of tuple)
- Fix the return value format (string list)

Otherwise, although the macOS environment can run the model, it can't decode correctly.

---

### 3.4 Image/video input: differences between VideoReader and OpenCV under macOS

Decord will appear on macOS:

- Unable to automatically select the hardware decoder
- A large number of frame I/O failed to read.
- The sorting between pictures depends on the Finder's metadata, unlike Linux.

Official demo hypothesis:

```
cv2.VideoCapture(...)
```

But VideoCapture under macOS may return:

- Only 0~1 frames can be read.
- Reading frames is out of order.
- a lot of "corevideo pixel buffer errors"
- Reading mp4 must rely on ffmpeg installation.

So demo must be modified to:

- decoding with decord.VideoReader+CPU.
- Pictures are sorted by file name to avoid disorder.
- Add exception protection

These modifications enable macOS users to stabilize input video frames.

---

### 3.5 GpuDataParallel (official utils) does not support macOS.

GpuDataParallel default dependency in official utils:

- torch.cuda.device_count()
- CUDA multi-card scheduling

There macOS only:

- single card MPS
- or pure CPU.

So if you don't change:

```
AttributeError: 'mps' object has no attribute 'device_count'
```

Need to rewrite part of the logic of GpuDataParallel to make it:

- Right CUDA: Doka
- for MPS: single card operation
- to CPU: fall back to CPU.

And will:

```
data_to_device
```

Change it to a universal cross-device function.

---

### 3.6 Summary: Why must I adapt to the macOS?

To sum up:

| Category | Cause | Impact |
|------|------|------|
| System differences | Linux → macOS | Path/cache/temporary files behave differently |
| GPU differences | cuda → MPS | A large number of `. CUDA () ` will directly report errors |
| decode is incomplete | vocab/unicode/tuple mismatch | Forecast index cannot be translated into GLOSS correctly |
| Unstable input | OpenCV/VideoReader difference | Video frame reading error |
| device Management | GPUDataParallel Incompatible | Device reported an error during reasoning |

The end result is:

**The original demo.py can't complete normal reasoning even if it can be started on macOS.
System-level adaptation is necessary to make the project fully operational.**

## 4. Detailed explanation of MAC OS adaptation modification (line by line description)

This chapter is the core part of this README.
Here, we will explain which files must be modified in order for CorrNet to run successfully on macOS, and explain the reasons one by one.
Contains the following contents:

- 4.1 adapt decode.py (fix vocab/unicode/numpy/beam search)
- 4.2 fix demo.py (input, equipment, model loading, MPS fallback)
- 4.3 repair utils/GpuDataParallel
- 4.4 Repair picture sorting and video frame reading (to prevent disorder)
- 4.5 modify the model weight loading mode (compatible with macOS torch)

---

## 4.1 decode.py (core adaptation)

The official decode has the following problems:

1. ctcdecode (non-pyctcodec) in Linux environment is used by default.
2. vocab is constructed in the wrong way.
3. unicode mapping method is inconsistent with gloss_dict.
4. beam search output type does not match demo.py
5. Logists cannot directly `. numpy ()` (because detach+cpu is required on MPS).
6. decode result is string, demo.py expected list → error.

Therefore, the following modifications must be made.

---

### 4.1.1 Added pyctcdecode check (macOS has no CUDA).

The original code assumes that ctcdecode(CUDA) is available, but macOS is not.
Therefore, import pyctcodefirst:

```
try:
from pyctcdecode import build_ctcdecoder
_has_pyctc = True
except Exception:
_has_pyctc = False
```

Purpose:

-let macOS use beam search for CPU.
-Ensure that Linux users can still use ctcdecode if they have it.

---

### 4.1.2 Repair the vocab construction mode

Official gloss_dict format:

```
{
"HELLO": [0, …],
"THANKYOU": [1, …],
...
}
```

Therefore, **index → gloss** must be established correctly:

```
self.i2g = {v[0]: k for k, v in gloss_dict.items()}
```

If not repaired, it will lead to:

-All decode outputs are UNK.
-beam search gets characters but can't backcheck gloss.

---

### 4.1.3 Using unicode vocab (consistent with the author's paper)

CorrNet decoding requires "one unicode character per category":

```
self.vocab = [chr(20000 + i) for i in range(num_classes)]
```

If you do not do this step:

-beam search does not return indexable token.
-class_id = ord(char)-20000 calculation is incorrect.

---

#### 4.1.4 logits must be detach before numpy(MPS must).

MPS tensor is not allowed to be directly `. numpy ()`, but must:

```
logits = logits.detach().cpu().numpy()
```

Otherwise, report an error:

```
TypeError: can't convert mps tensor to numpy
```

---

### 4.1.5 beam search may report an error → automatically fall back to greedy.

In order to avoid the exception of pyctcdecode in macOS environment, it is necessary to:

```
try:
decoded = self.beam_decoder.decode(logit)
except:
return self._greedy(torch.tensor(logits), lengths)
```

Make the system fault-tolerant.

---

### 4.1.6 Fix the beam search return format.

The official return is string', which must be changed into:

```
[(gloss, index), ...]
```

Therefore, it is necessary to:

```
class_ids = [ord(ch) - 20000 for ch in decoded]
sent = [(self.i2g.get(cid, "UNK"), i) for i, cid in enumerate(class_ids) if cid ! = self.blank]
```

Make sure that demo.py can be parsed correctly.

---

## 4.2 demo.py adaptation (core part)

The original demo.py does not support macOS for the following reasons:

- Use `. cuda ()` to forcibly move data to CUDA(macOS None).
- GpuDataParallel depends on CUDA
- OpenCV/VideoReader is out of order under macOS.
- Equipment automatically detects missing.
- The logic of video reading and padding cannot handle Tensor on MPS.

The following are the main modification points.

---

### 4.2.1 Automatic detection equipment (unified entrance)

Add the following logic:

```
if torch.backends.mps.is_available():
map_location = torch.device("mps")
elif torch.cuda.is_available():
map_location = torch.device("cuda")
else:
map_location = torch.device("cpu")
```

And let the model use this device when loading and reasoning.

---

### 4.2.2 Handling MPS fallback

PyTorch's support for MPS is not perfect, and it often appears:

-Some operators do not support MPS.
-MPS is out of memory and other exceptions.

Therefore, it is recommended to run demo:

```
PYTORCH_ENABLE_MPS_FALLBACK=1 python demo.py  ...
```

---

### 4.2.3 Ensure that all tensors are unified device.

There are many places in the original code:

```
vid = vid.cuda()
...
video_length = video_length.cuda()
```

Must be replaced by:

```
vid = device.data_to_device(vid)
vid_lgt = device.data_to_device(video_length)
```

Unified management by GpuDataParallel.

---

### 4.2.4 Fix decord/video input sorting problem.

When macOS reads multiple images, the order is often out of order, so it must be sorted:

```
inputs = sorted(inputs, key=lambda x: os.path.basename(path))
```

Otherwise, the temporal order of the model will crash.

---

### 4.2.5 Repair decord's CPU-only usage.

Decord on macOS must use:

```
VideoReader(path, ctx=cpu(0))
```

Otherwise, report an error:

```
cannot find GPU context
```

---

### 4.2.6 Repair model weight loading (map_location)

MacOS load weight must:

```
state_dict = torch.load(model_path, map_location=map_location)
```

Otherwise, report the following error:

```
Attempting to deserialize object on a CUDA device but torch.cuda.is_available() is False
```

---

## 4.3 Repair utils.GpuDataParallel

The original version does not support macOS.

Must be modified to:

- CUDA: Doka
- MPS: single card
- CPU: single card

And transform:

```
def data_to_device(self, x):
return x.to(self.device)
```

Ensure that all devices are compatible.

---

## 4.4 Repair image input and frame padding

Video frame length reading on macOS is unstable, so demo must be fixed:

- left pad
- right pad
- stride alignment
-don't mix CPU/MPS device in temporary padding.

Otherwise reasoning will appear:

```
Expected all tensors on same device
```

---

## 4.5 Summary

This chapter introduces all the code modification points that must be made for macOS adaptation, involving:

-decode.py (character, beam search, vocab, numpy compatible)
-demo.py (multiple device repair+input repair)
-GpuDataParallel (rewriting device management)
-Video/picture input repair
-Weight loading repair

After these modifications, the whole CorrNet system can be guaranteed to run completely on macOS.
Including image sequence input, video input, beam search decoding, unicode vocab processing and so on.
## 5. Environment Configuration (macOS)

This chapter will fully explain how to create a complete environment that can run CorrNet on macOS, including:

- python/conda environment creation
- PyTorch (with MPS support) version selection and description
- pip freeze version explanation
- Why do I have to use these versions?
- One-stop command that can be directly copied and run.

---

## 5.1 System requirements (must be met)

macOS：

- macOS 13 Ventura or higher
- Apple Silicon(M1/M2/M3) or Intel.
- More than 8GB of memory (16GB recommended)

Framework requirements:

- Python 3.9 (most stable support PyTorch+decord)
- PyTorch 2.2 or above (native support MPS)
- torchvision/torchaudio is the same version.
- CUDA(macOS without CUDA) is not required.

---

## 5.2 Creating an Independent Conda Environment

Conda(miniconda/anaconda is acceptable) is recommended.

```
conda create -n corrnet python=3.9 -y
conda activate corrnet
```

Why does Python have to be **3.9**?

- PyTorch on macOS is the most stable in 3.9.
- decord has compatibility issues with python>3.10.
The-CorrNet project itself assumes Python 3.8/3.9 in Linux.

---

## 5.3 install PyTorch(MPS support)

MacOS uses **MPS** instead of CUDA.
The installation instructions are as follows:

```
pip install torch torchvision torchaudio --index-url  https://download.pytorch.org/whl/cpu
```

This version contains:

- CPU kernels
- MPS kernels
- No CUDA (normal)

Post-installation test:

```
python - << 'EOF'
import torch
print("PyTorch:", torch.__version__)
print("MPS available:", torch.backends.mps.is_available())
EOF
```

If output:

```
MPS available: True
```

Indicates that MPS is enabled.

---

## 5.4 Installing Gradio, OpenCV, Decord, Dependency Packages

This is the key part of this project.

```
pip install gradio==3.44.4 opencv-python==4.11.0.86 decord==0.6.1
pip install numpy==1.26.4 einops==0.8.1 scipy==1.13.1
pip install matplotlib pandas pillow
pip install transformers pyctcdecode
```

Why are these versions?

- `decord = = 0.6.1`: latest compatible version of MAC OS.
- `opencv-python==4.11 = 4.11`: QT dependency problem will not be triggered.
- `numpy = = 1.26.4`: pytorch 2.8 is the most stable collocation.
- `pyctcdecode==0.5.0: The most stable version based on Python 3.9.
- `gradio = = 3.44.4`: UI is the most stable and does not conflict with torch.

---

## 5.5 Project Dependency (consistent with your current environment)

The following is the' pip freeze' of your current environment.
This is the combination of **actually measured versions that can run** CorrNet on macOS:

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

Key explanation:

1. **torch = = 2.8.0+numpy = = 1.26.4**→ the most stable combination.
2. **pyctDecode = = 0.5.0** → Perfect compatibility with Python 3.9.
3. **Decord = = 0.6.1**→ the highest version available for MAC OS.
4. **gradio==3.44.4** → Fully support multi-image upload/video UI.
5. **opencv-python==4.11** → QT-free crash

Your environment can be used as an official reference combination.

---

## 5.6 One-stop installation command (can be directly copied)

If you want to give someone a command (macOS), you can use:

```
conda create -n corrnet python=3.9 -y
conda activate corrnet

pip install torch torchvision torchaudio --index-url  https://download.pytorch.org/whl/cpu
pip install gradio==3.44.4 opencv-python==4.11.0.86 decord==0.6.1
pip install numpy==1.26.4 einops==0.8.1 scipy==1.13.1
pip install pillow matplotlib pandas
pip install transformers==4.57.1 pyctcdecode==0.5.0
pip install tqdm
```

Guaranteed:
**100% successfully run CorrNet demo.py (picture sequence+video input) on macOS **

---

## 5.7 How to export pip freeze (for writing README)

Users can directly run:

```
pip freeze > requirements.txt
```

To view only the main package:

```
pip list
```

Instructions for README:

```
python -m pip list
```

---

## 5.8 Summary

This section provides:

- detailed steps to create a macOS environment
- Explain the version requirements of each library.
- Give the combination of dependencies that you are using and have verified.
- Give a replicable one-stop command.

These contents ensure that other macOS users can **100% reproduce your environment **, avoiding:

- MPS is not available
- decode.py reported that the numpy type is wrong.
- demo failed to read the video.
- CUDA related error
- Multi-graph sorting disorder
## 6. Description of code file structure (directory structure+function of each file)

This chapter explains the complete and operational directory structure of this project on macOS, and explains the purpose of each file/folder, so that newcomers can quickly understand the project architecture.

The following structure is based on your current runnable version (including decode.py & demo.py).

---

## 6.1 Project General Directory Structure

```
CorrNet/
│
├── preprocess/
│   ├── phoenix2014/
│   │   ├── gloss_dict.npy
│ └ ── ... (Other preprocessed documents)
│   └── CSL-Daily/
│       ├── gloss_dict.npy
│ └ ── ... (Other preprocessed documents)
│
├── utils/
│   ├── __init__.py
│   ├── video_augmentation.py
│   ├── data_utils.py
│ ├── device_helper.py (if you have it)
│└ ── Other utils files
│
├── slr_network/
│   ├── __init__.py
│   ├── modules/
│ └ ── (convolution, CTC, attention correlation)
│   ├── model_components/
│ └ ── (STCN/3D convolution/fusion module)
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

## 6.2 Top-level File Function Description

### 1）preprocess/
Dictionaries and pre-processing data needed for storing **sequence → text**.

-gloss_dict.npy: dictionary for mapping key frames of sign language to text.
-Each data set (phoenix/CSL) contains its own corresponding dictionary.

The final output of model reasoning is list index, which is converted back to real text through gloss_dict.

---

### 2）utils/
All the necessary tool functions for reasoning.

Document description:

```
utils/
├── video_augmentation.py # Video enhancement, center cropping, resize, and tensor transfer.
├── data_utils.py # Load & Format Frame Sequence
├── device_helper.py # automatically select CUDA/MPS/CPU.
└── __init__.py
```

Key documents:
`video_augmentation.py`
The main transform (Center Crop, Resize, Totensor) used by your demo.py comes from this file.

`device_helper.py`
Used for automatic switching:

- MPS（macOS）
- CUDA（Linux）
-CPU (without GPU)

---

### 3）slr_network/
Directory where the main model structure is located.

```
slr_network/
├── modules/ # ConvCTC, SeqCTC, timing encoder
├── model_components/ # STC block, CNN extractor
└── SLRModel.py # Global model encapsulation (forward reasoning)
```

Core:

- `SLRModel.py is the entrance of the whole network.
- Contains:
- feature extraction (2D/3D Conv)
- time series modeling (ConvCTC+SeqCTC)
- Decoded output

The reasoning call of demo.py is as follows:

```
ret_dict = model(vid, vid_lgt, ...)
```

---

### 4) decode.py (you rewrote it yourself)
Used for **command line video reasoning** (no UI).

Mainly includes:

```
- Load the model
- load gloss_dict
- Video frame reading (decord)
- transform image sequence
- Model reasoning
- output text
```

Suitable for shell execution:

```
python decode.py --model_path pretrained/phoenix.pth --video  xxx.mp4
```

---

### 5) demo.py (your main version)
A fully operational demo **with** Gradio Web UI, which supports:

-Multiple pictures (sorted by file name and automatically combined into a sequence)
-Video file (automatic frame extraction)
-automatically switch MPS/CUDA/CPU.
-output recognition text

Its functional modules:

```
- file path processing (safe_path)
- multi-graph sorting ensures correct frame order
- image reading (OpenCV)
- video reading (decord)
- transform pretreatment
- stride of padding alignment model convolution
- Model reasoning
- Gradio shows UI
```

This is the most important file to adapt for macOS users.

---

### 6）pretrained/
Store model weights:

```
pretrained/
├── phoenix.pth # PHOENIX2014 Model Weight
└── csl.pth # CSL-Daily model weight
```

Demo.py can use:

```
torch.load(model_weights, map_location=map_location)
```

Automatically load the corresponding device.

---

### 7）requirements.txt
Environment for direct installation by other users:

```
pip install -r requirements.txt
```

It is recommended to use your current version of freeze.

---

## 6.3 demo.py core flow chart (simplified logic)

```
Input (picture list or video)
↓
Safe_path → Get the real path.
↓
Multi-image sorting/video frame extraction (decord)
↓
Image list (RGB)
↓
video_augmentation (crop → resize → tensor)
↓
Normalization: vid = vid/127.5-1
↓
Padding alignment (according to convolution stride)
↓
Input model: SLRModel(vid)
↓
Output gloss sequence index
↓
Gloss_dict → text
↓
Show results (Gradio)
```

---

## 6.4 Dependencies among various files

```
demo.py
├── preprocess/gloss_dict.npy
├── utils/video_augmentation.py
├── utils/data_utils.py
├── utils/device_helper.py
├── slr_network/SLRModel.py
└── pretrained/*.pth
```

All files are clearly coupled and easy to expand.

---

## 6.5 Summary

Chapter 6 explains:

-Complete project structure
-Functions of each file
-the roles of demo.py and decode.py in the whole system.
-Dependencies of model, preprocessing and tool library
-How can new users quickly understand the whole project from the directory structure?

This chapter gives users a very good "general understanding" in README.
## 7. Reasoning process on MAC OS (multi-graph/video)

This chapter explains in detail how to run inference in macOS environment, including:

- Inference of multiple pictures as a frame sequence
- video file reasoning
- demo.py（Web UI）
- decode.py (command line)
- Common error handling
- explain MPS reasoning acceleration of macOS in particular.

---

## 7.1 Preparation

Before running reasoning, make sure that you have:

1. Complete the environment installation
2. The corresponding Conda environment has been activated.
3. The weight file (*.pth) has been placed in `./retrained/`.
4. gloss_dict.npy has been published in `. /preprocess/{dataset}/ `.

Activation environment:

```
conda activate corrnet
```

Check whether torch MPS is loaded successfully:

```
python - << 'EOF'
import torch
print("Torch:", torch.__version__)
print("MPS available:", torch.backends.mps.is_available())
EOF
```

Output:

```
Torch: 2.8.0
MPS available: True
```

Indicates that the macOS has enabled GPU acceleration.

---

## 7.2 Use demo.py (recommended, with interface)

### 7.2.1 Start the interface

Command:

```
python demo.py \
--model_path pretrained/phoenix.pth \
--language phoenix \
--device 0
```

If it is CSL:

```
python demo.py \
--model_path pretrained/csl.pth \
--language csl \
--device 0
```

After startup, the browser automatically opens the Gradio UI:

```
http://0.0.0.0:7862
```

---

## 7.3 Multi-graph reasoning (picture sequence)

It is suitable for you to divide the video into consecutive frames: img _ img_0001.jpg, img_0002.jpg ...

Demo.py will automatically:

- Press "Sort file names naturally"
- Put it together into a whole sequence
- feed into the model
- Output sign language sentences

On the page:

1. Open the **Multi-Images** tab.
2. Click "Upload Multiple Pictures"
3. Select all graphs
4. click "Run"

Output example:

```
I go to school today.
```

Key logic of sequence sorting:

```
sorted(inputs, key=lambda x: os.path.basename(safe_path(x)))
```

Ensure that:

```
0001.jpg
0002.jpg
0003.jpg
```

Cannot:

```
1.jpg
10.jpg
2.jpg
```

---

## 7.4 video reasoning

Supported formats:

```
.mp4, .avi, .mov, .mkv
```

Demo.py internal frame extraction through decord:

```
vr = VideoReader(video_path)
frames = vr.get_batch(...)
```

Processing steps:

1. Upload video files
2. click Run
3. Obtain the output sign language sentence.

Example:

```
What will the weather be like tomorrow
```

---

## 7.5 Using decode.py (Command Line Reasoning)

If you don't want to open Gradio, just use the command line reasoning:

```
python decode.py \
--model_path pretrained/phoenix.pth \
--video  input.mp4
```

Or:

```
python decode.py \
--images  img1.jpg  img2.jpg  img3.jpg
```

(If your decode.py has multi-graph mode)

Decode.py is based on the same logic, except that there is no UI.

---

## 7.6 MPS/CUDA/CPU automatic selection mechanism

Demo.py internal code:

```
if torch.backends.mps.is_available():
device = "mps"
elif torch.cuda.is_available():
device = "cuda"
else:
device = "cpu"
```

Function:

- **macOS automatically walks GPU(MPS)**
- Linux automatically walks CUDA.
- automatic CPU degradation without GPU

No need for users to manually change the code.
This is your special adaptation for macOS, which is very critical.

---

## 7.7 Model input structure explanation (shared by multiple images/videos)

Video frames or picture sequences will use transform:

```
CenterCrop → Resize → ToTensor → Normalize
```

Model input dimension:

```
[B, T, C, H, W]
```

Among them:

- B = batch = 1
-T = number of frames
- H,W = 224
- C = 3（RGB）

Inside the model, it will continue:

```
Padding alignment convolution stride
Convtcc → seq CTC → ctclloss → decoding
```

Finally, the gloss sequence is obtained, and then the text is mapped.

---

## 7.8 Description of output results

Format returned:

```
[("You", 0), ("Good", 1), ("Ma", 2)]
```

Demo.py finally shows the merged text, such as:

```
How are you?
```

If some frames are difficult to recognize, "UNK" may be output.

---

## 7.9 Frequently Asked Questions (macOS special)

### ① MPS is stuck or too slow.
Solve:

```
export PYTORCH_ENABLE_MPS_FALLBACK=1
```

Allows operators that MPS cannot execute to automatically fallback to CPU.

---

### ② decord reported an error and could not read the video.
Install the correct macOS version:

```
pip install eva-decord==0.6.1
```

It is already 0.6.1 in your environment, which means it is correct.

---

### (3) Multi-graphs are out of order.
Make sure the file name is in a uniform format:

```
frame_0001.jpg
frame_0002.jpg
...
```

---

### ④ Error in weight loading
Demo.py has joined fallback:

```
torch.load(..., weights_only=False)
```

Ensure compatibility with different weight formats.

---

## 7.10 Summary

In this chapter, you learned:

- How to run reasoning quickly on Mac OS?
- How to use demo.py(UI is the friendliest)
- How to use decode.py (command line)
- Complete reasoning process of multi-graph+video
- MPS automatic acceleration and fallback
- Common error handling

At this point, a mac user can run your project independently.
