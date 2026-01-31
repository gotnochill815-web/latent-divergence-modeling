from data.synthetic_generator import PairedTimeSeriesGenerator
from experiments.train import train

gen = PairedTimeSeriesGenerator(T=300)
data = gen.generate()

encoder, ssm = train(data)
