# UrbanGen-AI

**Synthetic urban-scene generation via class-conditional generative models, evaluated against real downstream recognition.**

UrbanGen-AI trains class-conditional generative models (cVAE, conditional DDPM in latent space, conditional DCGAN) from scratch on BDD100K scene images, then tests whether the synthetic residential-scene images they produce improve residential recall on the real BDD100K test set. It consumes [UrbanMind-AI](https://github.com/janakmallik/UrbanMind-AI)'s published data split manifests and Week 5 ResNet-50 baseline as frozen provenance — it does not modify UrbanMind-AI.

## What it is

A generative-modeling research project. The central question is: **can class-conditional generative models produce synthetic urban-scene images that measurably improve downstream urban-scene recognition, or does conventional augmentation beat them?**

Three generative models trained from scratch:
- Conditional VAE (cVAE) — variational inference, latent representation, class conditioning
- Conditional DDPM — diffusion in the cVAE's latent space (no pretrained Stable Diffusion)
- Conditional DCGAN — adversarial generation, mode collapse analysis

Five downstream ResNet-50 training arms:
- A: real BDD100K train only (baseline)
- B: real + conventional augmentation (crop/flip/jitter)
- C: real + cVAE synthetic residential images
- D: real + DDPM synthetic residential images
- E: real + DCGAN synthetic residential images

Success is measured by residential recall improvement on the real BDD100K test set with 95% bootstrap CIs excluding zero, and macro-F1 not degrading by more than 0.01. A negative result (no generative arm beats conventional augmentation) is a documented outcome, not a failure.

## Provenance

This project consumes the following published artifacts from UrbanMind-AI at commit `dfcfaf7` (tag `v0.10.0-week10-calibration`):

- BDD100K scene split manifests (`data/splits/resnet_{val,test}_{clean,excluded}.json`)
- Week 5 ResNet-50 checkpoint: `bdd_resnet50_best.pt`
  SHA-256: `F760B94C4BB334C95C5D6C3C7884534FE6F51CE397641249610AAB1A98447F84`
- Week 5 evaluation protocol (bootstrap CIs, per-class recall)

See `docs/dataset_cards/bdd100k_generative.md` for the full frozen provenance record.

## Status

Week G0 (setup and scoping) — in progress.

## License

MIT — see [LICENSE](LICENSE).