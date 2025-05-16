import torch
import torch.nn as nn
import torch.nn.functional as F

class NeuralTuringMachine(nn.Module):
    def __init__(self, input_size, output_size, controller_size=100, memory_units=128, memory_width=20):
        super(NeuralTuringMachine, self).__init__()
        
        self.input_size = input_size
        self.output_size = output_size
        self.controller_size = controller_size
        self.memory_units = memory_units
        self.memory_width = memory_width

        # Memoria externa
        self.memory = torch.randn(self.memory_units, self.memory_width) * 0.05

        # Controlador (LSTM)
        self.controller = nn.LSTMCell(input_size, controller_size)

        # Parámetros de lectura
        self.read_key = nn.Linear(controller_size, memory_width)
        self.read_beta = nn.Linear(controller_size, 1)

        # Capa de salida
        self.output = nn.Linear(controller_size + memory_width, output_size)

        # Inicializar estado del controlador
        self.h = None
        self.c = None

    def reset_memory(self):
        self.memory = torch.randn(self.memory_units, self.memory_width) * 0.05

    def cosine_similarity(self, key, memory):
        key = key.unsqueeze(0)  # [1, W]
        numerator = F.cosine_similarity(key, memory, dim=1)
        return numerator

    def forward(self, x):
        batch_size = x.size(0)

        if self.h is None:
            self.h = torch.zeros(batch_size, self.controller_size)
            self.c = torch.zeros(batch_size, self.controller_size)

        # Paso del controlador
        self.h, self.c = self.controller(x, (self.h, self.c))

        # Lectura de memoria
        key = self.read_key(self.h)               # [B, W]
        beta = F.softplus(self.read_beta(self.h)) # [B, 1]

        sim = self.cosine_similarity(key, self.memory)  # [N]
        weights = F.softmax(beta * sim, dim=0)          # [N]

        read = torch.matmul(weights.unsqueeze(0), self.memory)  # [1, W]

        # Combinar con la salida del controlador
        combined = torch.cat([self.h, read], dim=1)
        out = self.output(combined)

        return out
