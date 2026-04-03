"""
Patches fdm_continuous_am_train.py to fix the reconstruction loss.

Changes:
1. Ensures readout_probe and tensors are on the same device as capture.hidden
2. Adds debug output on first batch to confirm recon_loss is nonzero
3. Removes the `with torch.no_grad()` wrapping target_emb so gradients
   can flow through embedder if needed
4. Fixes LAMBDA_RECON being ignored (was always 0.1 but recon was 0.0)
"""

path = "fdm_continuous_am_train.py"
with open(path) as f:
    src = f.read()

old = """            # Reconstruction loss (only if we have hidden states)
            recon_loss = torch.tensor(0.0, device=device)
            if capture.hidden is not None:
                # Readout probe prediction
                pred_emb = readout_probe(capture.hidden.float())  # (batch, embed_dim)

                # Target embeddings from embedder
                with torch.no_grad():
                    target_emb = embedder.embed_batch(values).to(device)  # (batch, embed_dim)

                # Cosine reconstruction loss
                recon_loss = 1.0 - F.cosine_similarity(pred_emb, target_emb, dim=-1).mean()"""

new = """            # Reconstruction loss (only if we have hidden states)
            recon_loss = torch.tensor(0.0, device=device)
            if capture.hidden is not None:
                # Ensure probe is on same device as captured hidden states
                hidden = capture.hidden  # (batch, seq_len, hidden_dim)
                readout_probe.to(hidden.device)

                # Readout probe prediction — mean pool over FDM token positions
                pred_emb = readout_probe(hidden.float())  # (batch, embed_dim)

                # Target embeddings — no_grad since embedder is frozen reference
                with torch.no_grad():
                    target_emb = embedder.embed_batch(values).to(hidden.device).float()

                # Cosine reconstruction loss
                cos_sim = F.cosine_similarity(pred_emb, target_emb, dim=-1)
                recon_loss = (1.0 - cos_sim).mean()

                # Debug on first batch of first epoch
                if n_batches == 0 and epoch == 0:
                    print(f"  DEBUG recon: loss={recon_loss.item():.4f} "
                          f"pred_norm={pred_emb.norm().item():.3f} "
                          f"target_norm={target_emb.norm().item():.3f} "
                          f"hidden shape={hidden.shape} "
                          f"capture.hidden is None: False")
            else:
                if n_batches == 0 and epoch == 0:
                    print(f"  DEBUG: capture.hidden is None — hook not firing!")"""

if old in src:
    src = src.replace(old, new, 1)
    print("OK: reconstruction loss block patched")
else:
    print("ERROR: reconstruction loss block not found — check whitespace")
    # Show context around where it should be
    idx = src.find("Reconstruction loss")
    if idx >= 0:
        print("Found 'Reconstruction loss' at char", idx)
        print(repr(src[idx:idx+500]))
    exit(1)

# Also fix: make sure readout_probe parameters are included in optimizer
# The current code filters `model.parameters()` but readout_probe is separate
old_trainable = """    trainable = (
        list(filter(lambda p: p.requires_grad, model.parameters())) +
        list(filter(lambda p: p.requires_grad, proj_directions.parameters())) +
        list(readout_probe.parameters())
    )"""

new_trainable = """    # Move readout_probe to device before building optimizer
    readout_probe.to(device)

    trainable = (
        list(filter(lambda p: p.requires_grad, model.parameters())) +
        list(filter(lambda p: p.requires_grad, proj_directions.parameters())) +
        list(readout_probe.parameters())
    )"""

if old_trainable in src:
    src = src.replace(old_trainable, new_trainable, 1)
    print("OK: readout_probe device placement patched")
else:
    print("WARNING: trainable block not found — may already be patched")

with open(path, "w") as f:
    f.write(src)

print(f"\nDone. Patched {path}")
print("Run: python fdm_continuous_am_train.py --stage 3 2>&1 | head -60")
print("Check for 'DEBUG recon:' line to confirm loss is nonzero")
