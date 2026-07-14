from data.synthetic_generator import PairedTimeSeriesGenerator
from experiments.train import train

generator = PairedTimeSeriesGenerator(T=300)
data = generator.generate()

encoder, decoder, ssm = train(data)