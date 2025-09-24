import pytorch_lightning as pl
from rdkit import Chem
from rdkit.Chem import rdRascalMCES

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
        predicted_tokens, _ = pl_module.get_smiles_preds(sample_spectrum)

        # 4. Decode both sequences back into strings
        tokenizer = trainer.datamodule.tokenizer
        true_smiles = tokenizer.decode(sample_true_tokens, skip_special_tokens=True).replace(' ', '')
        predicted_smiles = tokenizer.decode(predicted_tokens.flatten(), skip_special_tokens=True).replace(' ', '')
        
        # 5. determine the tanimoto similarity between the two mols
        mol1 = Chem.MolFromSmiles(true_smiles)
        mol2 = Chem.MolFromSmiles(predicted_smiles)
        similarity = 0.0
        if mol1 == None or mol2 == None:
            print('\nEncountered an invalid SMILES string, continuing...')
        else:
            opts = rdRascalMCES.RascalOptions()
            opts.similarityThreshold = 0.0
            sim_results = rdRascalMCES.FindMCES(mol1, mol2, opts)
            if len(sim_results) > 0:
                similarity = sim_results[0].similarity

        # 5. Print the results to the console
        print(f"\n\n--- Validation Sample Prediction (Epoch {trainer.current_epoch}) ---")
        print(f"  True SMILES        : {true_smiles}")
        print(f"  Predicted SMILES   : {predicted_smiles}")
        print(f"  Tanimoto Similarity: {similarity:.3f}")
        print("--------------------------------------------------\n")
        
        # 6. log the similarity
        pl_module.log('tanimoto_similarity', similarity, 
                      prog_bar=False, on_step=False, on_epoch=True)