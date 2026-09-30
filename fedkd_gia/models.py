"""ViT wrappers that expose per-layer hidden states and attention maps for FedKD distillation."""
import timm
import torch
import torch.nn as nn

TEACHER_ARCH = "vit_base_patch16_224"
STUDENT_ARCH = "vit_small_patch16_224"
EMBEDDING_KEYS = ("patch_embed", "pos_embed", "cls_token")


class ViTWithInternals(nn.Module):
    """timm ViT that records every block's output and attention probabilities during forward."""

    def __init__(self, arch, num_classes, pretrained=True):
        super().__init__()
        self.vit = timm.create_model(arch, pretrained=pretrained, num_classes=num_classes)
        self.num_heads = self.vit.blocks[0].attn.num_heads
        self.embed_dim = self.vit.embed_dim
        self.depth = len(self.vit.blocks)
        self._hidden, self._attn = [], []
        self.record = False
        for blk in self.vit.blocks:
            # Fused SDPA never materializes the attention matrix; the explicit path does.
            blk.attn.fused_attn = False
            blk.attn.attn_drop.register_forward_hook(self._attn_hook)
            blk.register_forward_hook(self._hidden_hook)

    def _attn_hook(self, module, inputs, output):
        if self.record:
            self._attn.append(inputs[0])

    def _hidden_hook(self, module, inputs, output):
        if self.record:
            self._hidden.append(output)

    def forward(self, x, return_internals=False):
        self.record = return_internals
        self._hidden, self._attn = [], []
        logits = self.vit(x)
        self.record = False
        if not return_internals:
            return logits
        hidden, attn = self._hidden, self._attn
        self._hidden, self._attn = [], []
        return logits, hidden, attn


def build_student(num_classes, pretrained=True, freeze_embeddings=True):
    return _build(STUDENT_ARCH, num_classes, pretrained, freeze_embeddings)


def build_teacher(num_classes, pretrained=True, freeze_embeddings=True):
    return _build(TEACHER_ARCH, num_classes, pretrained, freeze_embeddings)


def _build(arch, num_classes, pretrained, freeze_embeddings):
    model = ViTWithInternals(arch, num_classes, pretrained)
    if freeze_embeddings:
        # Official FedKD freezes embedding parameters of both mentor and mentee.
        for name, p in model.named_parameters():
            if any(k in name for k in EMBEDDING_KEYS):
                p.requires_grad = False
    return model


class HiddenProjector(nn.Module):
    """Per-layer linear maps W_h from student hidden size to teacher hidden size (kept local to each client)."""

    def __init__(self, depth, d_student, d_teacher):
        super().__init__()
        self.maps = nn.ModuleList(nn.Linear(d_student, d_teacher) for _ in range(depth))
        for m in self.maps:
            nn.init.xavier_uniform_(m.weight, gain=1)
            nn.init.zeros_(m.bias)

    def forward(self, i, h):
        return self.maps[i](h)


def shared_state(student):
    """Parameters the client uploads: trainable student parameters (frozen embeddings never change)."""
    return {n: p.detach() for n, p in student.named_parameters() if p.requires_grad}
