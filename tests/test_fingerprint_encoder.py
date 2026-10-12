import torch

from structure_pretrain.data.molecular_graph import fingerprint_tensor
from structure_pretrain.models.fingerprint_encoder import FingerprintEncoder


def test_fp_forward():
    fingerprints = torch.stack([fingerprint_tensor("CCO"), fingerprint_tensor("OCC")])
    assert fingerprints.shape == (2, 2048)
    torch.testing.assert_close(fingerprints[0], fingerprints[1])
    encoder = FingerprintEncoder().eval()
    output = encoder(fingerprints)
    assert output.shape == (2, 256)
    assert torch.isfinite(output).all()
    torch.testing.assert_close(output[0], output[1])
