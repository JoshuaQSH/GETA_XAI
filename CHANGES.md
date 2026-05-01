# Changes

`graph.py`: 
- Updated the version check in `_get_trace_graph` to handle PyTorch 2.x. The `torch.onnx._optimize_graph` function was removed in PyTorch 2.x, so for these versions, we now fall back to using `_optimize_trace_graph_no_onnx_operator`.

`utils.py`: 
- Added a compatibility wrapper `_is_caffe2_aten_fallback()` that returns False for PyTorch 2.x (since Caffe2 support was removed) and defers to `symbolic_helper.is_caffe2_aten_fallback()` for older versions.
- Updated the call to `_jit_pass_onnx_unpack_quantized_weights` to use the new 2-argument signature in PyTorch 2.x (the third argument for Caffe2 fallback was removed).

`geta.py` and `xai_geta.py`
- `_d_quant_helper()`: Add safeguards to prevent math range error (`d_quant = math.exp(t_quant * math.log(abs(q_m))) / (2 ** (bit_width - 1) - 1)`). `forget_rate = 0.0` added to the after `TODO` to avoid UnboundLocalError.

- `gradient_descent_step`, `partial_projected_gradient_descent_step_range_wt`: Clamp quantization params to valid ranges. `q_m` and `d_quant` must be positive

`quant_layers.py`:
- Instead of `torch.tensor(self.weight_clip_val, device=weight.device)`, do `weight_clip_val = weight.new_tensor(self.weight_clip_val)`

- Instead of `torch.tensor(0.0, device=weight.device)`, do `q_s = weight.new_zeros(())`

# Issues

- [ ] Optimizer `SGD` and `AdamW` may have quantization instability issues