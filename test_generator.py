from data.synthetic_generator import PairedTimeSeriesGenerator

gen = PairedTimeSeriesGenerator()
data = gen.generate()

print(data["obs_1"].shape)
print(data["latent_1"].shape)
print(data["regime"].unique())
