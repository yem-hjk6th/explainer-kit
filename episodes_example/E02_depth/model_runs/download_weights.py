"""Download the three weight sets for the E02 model runs from the official HF repos into the default HF cache."""
from huggingface_hub import hf_hub_download, snapshot_download

p = hf_hub_download("apple/DepthPro", "depth_pro.pt")
print("apple/DepthPro", p)
for rid in ["depth-anything/DA3-LARGE-1.1", "depth-anything/DA3METRIC-LARGE"]:
    print(rid, snapshot_download(rid))
