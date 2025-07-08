# Transformer

## Getting started (simple approach)

1. Install uv

2. Run the command
```
uv pip sync uv.lock --system --no-cache --require-hashes --index-strategy unsafe-best-match --extra-index-url https://download.pytorch.org/whl/cu128
```

3. Activate the virtual environment

4. Tweak conf/ to your liking

5. Run the following to train a model:
```
python src/matformer/main.py
```

6. Check the results/ dir for mlflow logs, etc.