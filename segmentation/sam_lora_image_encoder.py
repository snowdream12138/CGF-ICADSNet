from segment_anything import sam_model_registry
import math
import torch
import torch.nn as nn
from torch.nn.parameter import Parameter
from segment_anything.modeling import Sam


class _LoRA_qkv(nn.Module):
    def __init__(
            self,
            qkv: nn.Module,
            linear_a_q: nn.Module,
            linear_b_q: nn.Module,
            linear_a_v: nn.Module,
            linear_b_v: nn.Module,
            linear_a_k: nn.Module,
            linear_b_k: nn.Module,
    ):
        super().__init__()
        self.qkv = qkv
        self.dim = qkv.in_features

        self.linear_a_q = linear_a_q
        self.linear_b_q = linear_b_q
        self.linear_a_v = linear_a_v
        self.linear_b_v = linear_b_v
        self.linear_a_k = linear_a_k
        self.linear_b_k = linear_b_k

        self.alpha = nn.Parameter(torch.tensor(0.5))

    def forward(self, x):
        original_qkv = self.qkv(x)

        lora_q = self.linear_b_q(self.linear_a_q(x))
        lora_v = self.linear_b_v(self.linear_a_v(x))
        lora_k = self.linear_b_k(self.linear_a_k(x))

        original_qkv[..., :self.dim] += self.alpha * lora_q
        original_qkv[..., self.dim:2 * self.dim] += self.alpha * lora_k
        original_qkv[..., 2 * self.dim:] += self.alpha * lora_v

        return original_qkv


class LoRA_Sam(nn.Module):
    def __init__(self, sam_model: Sam, r: int, lora_layer=None):
        super(LoRA_Sam, self).__init__()
        assert r > 0, "LoRA must > 0"

        self.lora_layer = lora_layer if lora_layer else list(
            range(len(sam_model.image_encoder.blocks))
        )

        self.w_As = nn.ModuleList()
        self.w_Bs = nn.ModuleList()
        self.alpha_params = nn.ParameterList()

        for param in sam_model.image_encoder.parameters():
            param.requires_grad = False


        # for param in sam_model.prompt_encoder.parameters():
        #     param.requires_grad = False

        # for param in sam_model.mask_decoder.parameters():
        #     param.requires_grad = False

        self.lora_modules = nn.ModuleList()

        for t_layer_i, blk in enumerate(sam_model.image_encoder.blocks):
            if t_layer_i not in self.lora_layer:
                continue

            w_qkv = blk.attn.qkv
            dim = w_qkv.in_features

            w_a_q = nn.Linear(dim, r, bias=False)
            w_b_q = nn.Linear(r, dim, bias=False)
            w_a_v = nn.Linear(dim, r, bias=False)
            w_b_v = nn.Linear(r, dim, bias=False)
            w_a_k = nn.Linear(dim, r, bias=False)
            w_b_k = nn.Linear(r, dim, bias=False)

            new_qkv = _LoRA_qkv(
                w_qkv,
                w_a_q, w_b_q,
                w_a_v, w_b_v,
                w_a_k, w_b_k
            )
            blk.attn.qkv = new_qkv

            self.lora_modules.append(new_qkv)

            self.w_As.extend([w_a_q, w_a_v, w_a_k])
            self.w_Bs.extend([w_b_q, w_b_v, w_b_k])
            self.alpha_params.append(new_qkv.alpha)

        self.reset_parameters()
        self.sam = sam_model

    def reset_parameters(self):
        for w_A in self.w_As:
            nn.init.orthogonal_(w_A.weight)
        for w_B in self.w_Bs:
            nn.init.normal_(w_B.weight, mean=0, std=0.01)
        for alpha in self.alpha_params:
            nn.init.constant_(alpha, 0.5)

    def save_lora_parameters(self, filename: str) -> None:
        state_dict = {
            **{f"w_a_{i}": p.weight for i, p in enumerate(self.w_As)},
            **{f"w_b_{i}": p.weight for i, p in enumerate(self.w_Bs)},
            **{f"alpha_{i}": p for i, p in enumerate(self.alpha_params)},

            **{k: v for k, v in self.sam.state_dict().items()
               if 'prompt_encoder' in k or 'mask_decoder' in k}
        }
        torch.save(state_dict, filename)

    def load_lora_parameters(self, filename: str) -> None:
        state_dict = torch.load(filename, map_location='cpu')

        for i, w_A in enumerate(self.w_As):
            w_A.weight.data.copy_(state_dict[f"w_a_{i}"])
        for i, w_B in enumerate(self.w_Bs):
            w_B.weight.data.copy_(state_dict[f"w_b_{i}"])
        for i, alpha in enumerate(self.alpha_params):
            alpha.data.copy_(state_dict[f"alpha_{i}"])

        sam_dict = self.sam.state_dict()
        sam_dict.update({
            k: v for k, v in state_dict.items()
            if 'prompt_encoder' in k or 'mask_decoder' in k
        })
        self.sam.load_state_dict(sam_dict)

    def forward(self, batched_input, multimask_output, image_size):
        return self.sam(batched_input, multimask_output, image_size)

