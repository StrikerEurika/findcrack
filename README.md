# find-crack

uv add --index https://download.pytorch.org/whl/cu124 torch torchaudio torchvision

uv run python -c "import torch; print(torch.__version__)"

uv run python -c "import torch; print('PyTorch CUDA version:', torch.version.cuda)"

uv run python -c "import torch; print('CUDA available:', torch.cuda.is_available())"