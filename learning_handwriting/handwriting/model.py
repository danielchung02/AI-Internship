import torch
from torch import nn

class JamoGlyphNet(nn.Module):
    """Conditional MLP + convolutional decoder for a 96×96 handwriting glyph."""
    def __init__(self, other_vocab: int, emb_dim: int = 64, latent: int = 512):
        super().__init__()
        self.initial = nn.Embedding(20, emb_dim)
        self.vowel = nn.Embedding(22, emb_dim)
        self.final = nn.Embedding(28, emb_dim)
        self.other = nn.Embedding(other_vocab, emb_dim)
        self.kind = nn.Embedding(2, emb_dim)
        self.mlp = nn.Sequential(nn.Linear(emb_dim * 2, latent), nn.GELU(), nn.Linear(latent, 512 * 6 * 6), nn.GELU())
        self.decoder = nn.Sequential(
            nn.Unflatten(1, (512, 6, 6)),
            nn.ConvTranspose2d(512, 256, 4, 2, 1), nn.BatchNorm2d(256), nn.GELU(),
            nn.ConvTranspose2d(256, 128, 4, 2, 1), nn.BatchNorm2d(128), nn.GELU(),
            nn.ConvTranspose2d(128, 64, 4, 2, 1), nn.BatchNorm2d(64), nn.GELU(),
            nn.ConvTranspose2d(64, 32, 4, 2, 1), nn.BatchNorm2d(32), nn.GELU(),
            # 6 -> 12 -> 24 -> 48 -> 96. Keep the final glyph at 96×96.
            nn.Conv2d(32, 1, 3, 1, 1), nn.Sigmoid())

    def forward(self, initial, vowel, final, other, is_hangul):
        h = self.initial(initial) + self.vowel(vowel) + self.final(final)
        o = self.other(other)
        base = torch.where(is_hangul[:, None].bool(), h, o)
        z = self.mlp(torch.cat([base, self.kind(is_hangul.long())], dim=1))
        return self.decoder(z)
