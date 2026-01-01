# JudicAIta Kaggle Setup Guide

Complete guide for running JudicAIta in Kaggle notebooks for the Google Tunix Hackathon.

## Prerequisites

Before starting, ensure you have:

- **Kaggle Account**: With notebook access enabled
- **Hugging Face Account**: For model downloads (sign up at huggingface.co)
- **Google Account**: For Colab access if preferred

## Quick Start

### Option 1: Kaggle Notebook

1. **Open the training notebook**:
   - Navigate to [examples/notebooks/train_tunix_reasoning.ipynb](examples/notebooks/train_tunix_reasoning.ipynb)
   - Upload to Kaggle or import directly

2. **Configure runtime**:
   - Settings → Accelerator → TPU VM v3-8
   - Settings → Internet → On

3. **Run the setup cell**:
   ```python
   # Cell 1: Install dependencies
   !pip install -q -r requirements-kaggle.txt
   
   # Install from GitHub for latest versions
   !pip install git+https://github.com/clduab11/judicAIta.git
   !pip install git+https://github.com/google/tunix
   !pip install git+https://github.com/google/qwix
   !pip install git+https://github.com/jax-ml/jax
   !pip install git+https://github.com/google/flax
   ```

4. **Restart runtime** after installation (this is critical!)

5. **Run validation**:
   ```python
   !python scripts/validate_kaggle_env.py
   ```

### Option 2: Google Colab

1. **Open in Colab**: Use the Colab link in the notebook
2. **Set runtime**: Runtime → Change runtime type → TPU
3. Follow the same installation steps as Kaggle

---

## Detailed Installation

### Step 1: Package Installation

Run this cell first:

```python
# Core dependencies (Kaggle-compatible)
!pip install -q pydantic>=2.5.0 pydantic-settings>=2.1.0
!pip install -q python-dotenv loguru typer rich
!pip install -q pdfplumber python-docx pypdf
!pip install -q nest-asyncio aiofiles httpx tqdm

# JudicAIta from GitHub
!pip install -q git+https://github.com/clduab11/judicAIta.git

# Tunix and JAX (from GitHub for TPU support)
!pip install -q git+https://github.com/google/tunix
!pip install -q git+https://github.com/google/qwix
!pip install -q git+https://github.com/jax-ml/jax
!pip uninstall -q flax -y
!pip install git+https://github.com/google/flax

print("✅ Installation complete! Please restart the runtime.")
```

### Step 2: Restart Runtime

**Critical!** You must restart the runtime after installation:

- **Kaggle**: Runtime → Restart Session
- **Colab**: Runtime → Restart runtime

**Do NOT re-run Step 1 after restarting!**

### Step 3: Validate Environment

```python
# Import validation
from judicaita.notebook_utils import NotebookHelper

# Quick test
helper = NotebookHelper(show_progress=True)
print("✅ JudicAIta loaded successfully!")

# Full validation
!python -m scripts.validate_kaggle_env
```

### Step 4: TPU Verification

```python
import jax
print(f"JAX version: {jax.__version__}")
print(f"Devices: {jax.devices()}")
print(f"TPU cores: {len([d for d in jax.devices() if d.platform == 'tpu'])}")
```

Expected output:
```
JAX version: 0.8.x
Devices: [TpuDevice(id=0, ...), TpuDevice(id=1, ...), ...]
TPU cores: 8
```

---

## Cell-by-Cell Walkthrough

### Training Notebook Structure

| Cell | Purpose | Expected Output |
|------|---------|-----------------|
| 1 | Install dependencies | Package installation logs |
| 2 | Restart runtime checkpoint | ⚠️ RESTART REQUIRED |
| 3 | Import libraries | No errors, version info |
| 4 | Initialize TPU | "8 TPU cores detected" |
| 5 | Download model | Model files cached |
| 6 | Load dataset | "X training samples loaded" |
| 7 | Configure GRPO | Configuration display |
| 8 | Define reward function | "Reward function ready" |
| 9 | Initialize trainer | "GRPO trainer initialized" |
| 10 | Run training | Loss values, progress bar |
| 11 | Validate output | Quality metrics |
| 12 | Export adapters | "Saved to ./kaggle_upload/" |

### Expected Training Metrics

After training completes, you should see:

```
Training Progress:
  Step 100/500: loss=0.234, reward=0.67
  Step 200/500: loss=0.198, reward=0.72
  ...

Validation Results:
  - XML Format Compliance: 95%
  - Average Reasoning Tokens: 142
  - Reasoning Quality Score: 0.73
  
✅ Training complete! Ready for submission.
```

---

## Troubleshooting

### Common Errors

#### 1. `ModuleNotFoundError: No module named 'tunix'`

**Cause**: Wrong installation method

**Solution**:
```python
# Use GitHub installation
!pip install git+https://github.com/google/tunix
```

#### 2. `RuntimeError: No TPU found`

**Cause**: Runtime not configured for TPU

**Solution**:
- Kaggle: Settings → Accelerator → TPU VM v3-8
- Colab: Runtime → Change runtime type → TPU

#### 3. `jax_cuda12_plugin` warnings

**Cause**: Normal on Colab/Kaggle

**Solution**: These warnings are harmless and can be ignored.

#### 4. `OutOfMemoryError`

**Cause**: Batch size too large

**Solution**:
```python
# Reduce batch size in config
GRPO_CONFIG = {
    "batch_size": 2,  # Reduce from 4
    "num_generations": 2,  # Reduce from 4
}
```

#### 5. JAX TPU initialization fails

**Cause**: Incompatible JAX version

**Solution**:
```python
# Install from GitHub (not PyPI)
!pip install git+https://github.com/jax-ml/jax
```

#### 6. Imports fail after installation

**Cause**: Runtime not restarted

**Solution**: Restart runtime and DO NOT re-run installation cells.

---

## Kaggle Constraints

### Resource Limits

| Resource | Limit | Notes |
|----------|-------|-------|
| Session Duration | ~9 hours | Plan training accordingly |
| GPU Memory | 16 GB | Use smaller batches if needed |
| TPU Memory | 128 GB (total) | Split across 8 cores |
| Internet | Available | Required for model downloads |
| Disk Space | 20 GB | Be mindful of checkpoints |

### Optimization Tips

1. **Enable Internet**: Required for GitHub installs and model downloads
2. **Use checkpointing**: Save progress every N steps
3. **Monitor memory**: Watch for OOM warnings
4. **Start fresh**: Use new session for submission runs

---

## Submission Package Preparation

After training, prepare your submission:

### 1. Verify Package Structure

```python
from pathlib import Path

kaggle_dir = Path('./kaggle_upload')
required_files = [
    'adapter_config.json',
    'adapter_model.safetensors',
    'tokenizer.json',
    'tokenizer_config.json',
    'README.md'
]

print("Package Verification:")
for f in required_files:
    exists = (kaggle_dir / f).exists()
    print(f"{'✅' if exists else '❌'} {f}")
```

### 2. Create Submission Zip

```python
import zipfile
from pathlib import Path

zip_path = Path('./judicaita_submission.zip')
kaggle_dir = Path('./kaggle_upload')

with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zf:
    for file in kaggle_dir.glob('*'):
        if file.is_file():
            zf.write(file, file.name)

print(f"✅ Created: {zip_path}")
print(f"   Size: {zip_path.stat().st_size / 1024 / 1024:.2f} MB")
```

### 3. Submit to Competition

1. Download the zip file
2. Go to competition submission page
3. Upload or link your dataset
4. Submit and verify on leaderboard

---

## Quick Reference

### Essential Commands

```python
# Install JudicAIta
!pip install git+https://github.com/clduab11/judicAIta.git

# Validate environment
!python scripts/validate_kaggle_env.py

# Import helper
from judicaita.notebook_utils import NotebookHelper

# Check TPU
import jax; print(jax.devices())
```

### Key Links

- **Repository**: https://github.com/clduab11/judicAIta
- **Validation Guide**: docs/COLAB_VALIDATION_GUIDE.md
- **Submission Checklist**: docs/HACKATHON_SUBMISSION_CHECKLIST.md
- **Training Notebook**: examples/notebooks/train_tunix_reasoning.ipynb

---

## Support

If you encounter issues:

1. Check the [Troubleshooting](#troubleshooting) section above
2. Review [COLAB_VALIDATION_GUIDE.md](docs/COLAB_VALIDATION_GUIDE.md)
3. Open an issue: https://github.com/clduab11/judicAIta/issues

---

**Last Updated**: January 2026  
**Version**: 1.0
