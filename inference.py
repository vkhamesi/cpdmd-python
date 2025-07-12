import numpy as np

class AdaptiveEWMA():

    def __init__(self, lam, L, burn_in):
        self.lam = lam
        self.L = L
        self.burn_in = burn_in
        self.detected_change = False
        self.r_factor = 1.0
        
    def update_mean_variance(self, data_new):
        self.r_factor = self.r_factor * (1 - self.lam) ** 2
        mu_new = self.mu + (data_new - self.mu) / self.n
        self.sigma = np.sqrt(self.sigma ** 2 + ((data_new - self.mu) * (data_new - mu_new) - self.sigma ** 2) / self.n)
        self.mu = mu_new
        
    def initialise_statistics(self, observations):
        self.n = len(observations)
        self.mu = np.mean(observations)
        self.sigma = np.std(observations)
        self.Z = self.mu
        self.sigma_Z = 0

    def update_statistics(self, data_new):
        self.n += 1
        self.Z = (1 - self.lam) * self.Z + self.lam * data_new
        self.sigma_Z = self.sigma * np.sqrt((self.lam / (2 - self.lam)) * (1 - self.r_factor))
    
    def decision_rule(self):
        if self.Z > self.mu + self.L * self.sigma_Z:
            self.detected_change = True
        elif self.Z < self.mu - self.L * self.sigma_Z:
            self.detected_change = True

    def process(self, data):
        for i, data_new in enumerate(data):
            if i == 0:
                continue
            if i == 1:
                self.initialise_statistics(data[:2])
            else:
                self.update_statistics(data_new)
                self.update_mean_variance(data_new)
                if i >= self.burn_in:
                    self.decision_rule()
                    if self.detected_change:
                        return i
        return None