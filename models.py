"""Small CNN used throughout HW1.

Architecture (from the assignment): input ``3 x S x S``, 100 classes,
sequential (no residual connections), built only from convolution, BatchNorm,
ReLU, pooling and linear layers. Every convolution uses ``padding = k // 2``
and ``bias=False``; every ReLU is ``inplace=True``.

Layer table
-----------
===============================  ==================
Layers                           Output resolution
===============================  ==================
Conv7x7 s2 3->32, MaxPool 3x3 s2 p1   S/4
Conv5x5 32->64                        S/4
Conv3x3 s2 64->128                    S/8
Conv1x1 128->256                      S/8
Conv3x3 s2 256->256                   S/16
Conv1x1 256->512                      S/16
head: GlobalAvgPool, Linear 512->256, ReLU, Linear 256->100   --
===============================  ==================

Runs in ``eval()`` mode, FP32.
"""

from torch import Tensor, nn


class CustomCNN(nn.Module):
    """The HW1 network.

    See the module docstring for the exact layer table.
    """

    def __init__(self, num_classes: int = 100):
        super().__init__()
        self.num_classes = num_classes
        
        self.features = nn.Sequential(
            # Layer 1: Conv7x7 s2 3->32, MaxPool 3x3 s2 p1 -> Out: S/4
            nn.Conv2d(3, 32, kernel_size=7, stride=2, padding=3, bias=False),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            
            # Layer 2: Conv5x5 32->64 -> Out: S/4
            nn.Conv2d(32, 64, kernel_size=5, stride=1, padding=2, bias=False),
            nn.ReLU(inplace=True),
            
            # Layer 3: Conv3x3 s2 64->128 -> Out: S/8
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1, bias=False),
            nn.ReLU(inplace=True),
            
            # Layer 4: Conv1x1 128->256 -> Out: S/8
            nn.Conv2d(128, 256, kernel_size=1, stride=1, padding=0, bias=False),
            nn.ReLU(inplace=True),
            
            # Layer 5: Conv3x3 s2 256->256 -> Out: S/16
            nn.Conv2d(256, 256, kernel_size=3, stride=2, padding=1, bias=False),
            nn.ReLU(inplace=True),
            
            # Layer 6: Conv1x1 256->512 -> Out: S/16
            nn.Conv2d(256, 512, kernel_size=1, stride=1, padding=0, bias=False),
            nn.ReLU(inplace=True),
        )
        
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(512, 256, bias=True),
            nn.ReLU(inplace=True),
            nn.Linear(256, num_classes, bias=True)
        )

    def forward(self, x: Tensor) -> Tensor:
        """Run one forward pass.

        Parameters
        ----------
        x : torch.Tensor
            Input of shape ``(B, 3, S, S)``.

        Returns
        -------
        torch.Tensor
            Logits of shape ``(B, num_classes)``.
        """
        return self.head(self.features(x))


def get_model(num_classes: int = 100) -> nn.Module:
    """Factory function for CustomCNN."""
    return CustomCNN(num_classes=num_classes)
