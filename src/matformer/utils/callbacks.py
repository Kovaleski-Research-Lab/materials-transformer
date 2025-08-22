import pytorch_lightning as pl

class SMILES_Prediction_Logger(pl.Callback):
    def on_validation_epoch_end(self, trainer, pl_module):
        """
        This function is called after every validation epoch.
        """
        # verify exist
        if not trainer.datamodule.val_dataloader():
            return

        # 1. Get one batch from the validation set
        val_batch = next(iter(trainer.datamodule.val_dataloader()))
        spectrum, true_tokens = val_batch
        spectrum = spectrum.to(pl_module.device)

        # 2. Select a single sample from the batch to predict
        # We add a batch dimension back with .unsqueeze(0)
        sample_spectrum = spectrum[0].unsqueeze(0)
        sample_true_tokens = true_tokens[0]

        # 3. Use the model's generation method to get the prediction
        tokenizer = trainer.datamodule.tokenizer
        predicted_tokens, _, _ = pl_module.generate_smiles(sample_spectrum, tokenizer)

        # 4. Decode both sequences back into strings
        true_smiles = tokenizer.decode(sample_true_tokens, skip_special_tokens=True)
        predicted_smiles = tokenizer.decode(predicted_tokens, skip_special_tokens=True)

        # 5. Print the results to the console
        print(f"\n\n--- Validation Sample Prediction (Epoch {trainer.current_epoch}) ---")
        print(f"  True SMILES     : {true_smiles}")
        print(f"  Predicted SMILES: {predicted_smiles}")
        print("--------------------------------------------------\n")