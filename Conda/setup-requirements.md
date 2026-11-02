# Install requirements

## 1. CUDA (NVIDIA only)

CUDA Toolkit 11.8: https://developer.nvidia.com/cuda-11-8-0-download-archive

## 2. PyTorch

NVIDIA:

```bash
pip install torch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 --index-url https://download.pytorch.org/whl/cu118
```

AMD / CPU:

```bash
pip install torch torchvision torchaudio
```

Note: an older Conda guide referenced torch 1.11 + cu113, outdated. Use 2.0.1 + cu118.

## 3. Project requirements

```bash
pip install -r requirements.txt
```

## 4. Conda extras (optional)

Only for Streamlit / extras demos:

```bash
pip install -r Conda/additionalRequirements.txt
```

## 5. Test

```bash
python main.py
```
