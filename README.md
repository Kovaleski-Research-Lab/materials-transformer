# Transformer

## Getting started (simple approach, dev strategy)

1. Install uv

2. Run the following command from the project root
```
uv pip sync requirements.txt --system --no-cache --index-strategy unsafe-best-match --extra-index-url https://download.pytorch.org/whl/cu128
```

3. Activate the virtual environment

4. Tweak conf/ to your liking

5. Run the following to train a model:
```
python src/matformer/main.py
```

6. Check the results/ dir for mlflow logs, etc.