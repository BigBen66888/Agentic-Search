import pytest
torch = pytest.importorskip("torch")
from search_r1_refine.rl.advantage import grouped_advantages

def test_grouped_advantage():
    out = grouped_advantages(torch.tensor([1.,0.,1.,0.]), ["a","a","b","b"])
    assert torch.isfinite(out).all()
    assert float(out.abs().max()) <= 5
