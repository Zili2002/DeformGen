# Policy Commands

Initialize the optional policy submodule before using these commands:

```bash
git submodule update --init --recursive policy
```

Example command templates:

```bash
cd policy
bash scripts/train_act.sh insert_rope deformgen_rope_act
bash scripts/train_dp.sh pack_sloth deformgen_sloth_dp
bash scripts/train_svla.sh fold_cloth deformgen_cloth_svla
bash scripts/train_pi0.sh pack_sloth deformgen_sloth_pi0 --skip-norm-stats
```
