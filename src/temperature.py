"""
Adaptive sampling-temperature controller: raises temperature on a sustained training-
reward plateau (more exploration), up to a ceiling. Defaults match the paper's recipe
(1.2 -> 1.6, step 0.05).
"""
import numpy as np


class TemperatureController:
    def __init__(self, initial=1.2, max_temp=1.6, step=0.05, patience=500, cooldown=800):
        self.temp = initial
        self.max_temp = max_temp
        self.step = step
        self.patience = patience
        self.cooldown = cooldown
        self.history = []
        self.steps_since_increase = cooldown
        self._history_cap = patience * 3

    def update(self, reward):
        self.history.append(float(reward))
        if len(self.history) > self._history_cap:
            self.history = self.history[-self.patience * 2:]
        self.steps_since_increase += 1
        if (len(self.history) >= self.patience
                and self.steps_since_increase >= self.cooldown
                and self.temp < self.max_temp):
            recent = np.mean(self.history[-self.patience // 2:])
            older = np.mean(self.history[-self.patience: -self.patience // 2])
            if recent - older < 0.002:
                self.temp = min(self.temp + self.step, self.max_temp)
                self.steps_since_increase = 0
                print(f"\n  [TempCtrl] Plateau -> temperature: {self.temp:.2f}")
        return self.temp
