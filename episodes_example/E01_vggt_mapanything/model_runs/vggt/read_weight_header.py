"""Read the VGGT-1B safetensors header (JSON header only, weights are not loaded) to check the 1024 dims / patch embed / DPT shapes."""
import glob, json, os, struct, re
import numpy as np

p = glob.glob(os.path.expanduser(r"~/.cache/huggingface/hub/models--facebook--VGGT-1B/snapshots/*/model.safetensors"))[0]
with open(p, "rb") as f:
    n = struct.unpack("<Q", f.read(8))[0]
    hdr = json.loads(f.read(n))
hdr.pop("__metadata__", None)
print("file", p)
pat = re.compile(r"(patch_embed\.patch_embed\.(proj|pos_embed|cls_token|register_tokens)|patch_embed\.pos_embed|"
                 r"depth_head\.(projects|resize_layers|scratch\.(layer\d_rn|output_conv\d|output_conv2\.\d))|"
                 r"camera_head\.pose_branch|track_head\.feature_extractor\.(projects|scratch\.output_conv2\.\d)|"
                 r"point_head\.scratch\.output_conv2\.\d|aggregator\.camera_token|aggregator\.register_token)")
for k, v in sorted(hdr.items()):
    if pat.search(k) and k.endswith(("weight", "pos_embed", "token", "cls_token", "register_tokens")):
        print(k, v["shape"])
w = [v["shape"] for k, v in hdr.items() if k.endswith("patch_embed.patch_embed.proj.weight") or k == "aggregator.patch_embed.patch_embed.proj.weight"]
print("proj weight shapes", w)
blocks = sorted({int(m.group(1)) for k in hdr for m in [re.search(r"aggregator\.patch_embed\.blocks\.(\d+)\.", k)] if m})
print("dinov2 blocks", blocks[0], "..", blocks[-1], len(blocks))
