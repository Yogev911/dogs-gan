import numpy as np

mu = 1
sigma = 0
Noise = np.random.normal(mu, sigma, 36**2)
Noise2D = np.reshape(Noise, (36,36))

