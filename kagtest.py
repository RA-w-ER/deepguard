from pathlib import Path

real = list(Path("data/processed/real").glob("*.png"))
fake = list(Path("data/processed/fake").glob("*.png"))

ff_real  = [f for f in real if not f.stem.startswith("gan_")]
gan_real = [f for f in real if f.stem.startswith("gan_")]
ff_fake  = [f for f in fake if not f.stem.startswith("gan_")]
gan_fake = [f for f in fake if f.stem.startswith("gan_")]

print(f"real FF++:  {len(ff_real)}")
print(f"real GAN:   {len(gan_real)}")
print(f"fake FF++:  {len(ff_fake)}")
print(f"fake GAN:   {len(gan_fake)}")