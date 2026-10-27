import torch
import torch.nn as nn
from typing import List

class CNNEncoder(nn.Module):
    def __init__(self, k_list: List[int] = [7, 7, 7, 7], c_list: List[int] = [6, 16, 32, 64, 128],
                        s_list: List[int] = [1, 1, 1, 1], p_list: List[int] = [3, 3, 3, 3]) -> None:
        super(CNNEncoder, self).__init__()
        self.k_list, self.c_list, self.s_list, self.p_list = k_list, c_list, s_list, p_list
        layers: List[nn.Module] = []

        for i in range(len(self.c_list) - 1):
            layers.append(torch.nn.Conv1d(self.c_list[i], self.c_list[i+1], self.k_list[i], \
                stride=self.s_list[i], padding=self.p_list[i]))
            layers.append(torch.nn.BatchNorm1d(self.c_list[i+1]))
            layers.append(torch.nn.GELU())
            layers.append(torch.nn.Dropout(0.1))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

class MotionEncoder(nn.Module):
    def __init__(self, k_list: List[int] = [7, 7, 7, 7], c_list: List[int] = [6, 16, 32, 64, 128],
                        s_list: List[int] = [1, 1, 1, 1], p_list: List[int] = [3, 3, 3, 3]) -> None:
        super().__init__()
        self.k_list, self.c_list, self.s_list, self.p_list = k_list, c_list, s_list, p_list
        layers: List[nn.Module] = []

        for i in range(len(self.c_list) - 1):
            layers.append(torch.nn.Conv1d(self.c_list[i], self.c_list[i+1], self.k_list[i], \
                stride=self.s_list[i], padding=self.p_list[i]))
            layers.append(torch.nn.BatchNorm1d(self.c_list[i+1]))
            layers.append(torch.nn.GELU())
        layers.append(torch.nn.Dropout(0.5))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)
