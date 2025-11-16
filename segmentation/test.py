import inspect
from segment_anything import sam_model_registry

build_sam_vit_h = sam_model_registry["vit_h"]
print(inspect.getsource(build_sam_vit_h))