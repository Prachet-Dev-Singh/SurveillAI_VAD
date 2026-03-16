import torch
import torch.nn as nn
from models.self_distillation import StudentEncoder
from models.decoder import ReconstructionDecoder
from models.vit_branch import ViTSpatialEncoder

class DistilledStudentInference(nn.Module):
    """
    Combines the Teacher's vision with the Student's compression.
    This is what we need to evaluate the student on raw images.
    """
    def __init__(self, latent_dim=256):
        super().__init__()
        # 1. We need the Teacher to convert the Image -> Features
        self.teacher_vision = ViTSpatialEncoder(freeze_blocks=12, embed_dim=384, output_dim=latent_dim)
        # 2. We use your 517KB Student weights
        self.student_brain = StudentEncoder(input_dim=latent_dim, output_dim=latent_dim)
        # 3. We use the saved Decoder
        self.decoder = ReconstructionDecoder(latent_dim=latent_dim)

    def forward(self, x):
        with torch.no_grad():
            feat = self.teacher_vision(x)
            compressed_feat = self.student_brain(feat)
            return self.decoder(compressed_feat)

print("Student Loader helper created!")
