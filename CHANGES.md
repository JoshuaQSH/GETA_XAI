# Changes

`graph.py`: 
- Updated the version check in `_get_trace_graph` to handle PyTorch 2.x. The `torch.onnx._optimize_graph` function was removed in PyTorch 2.x, so for these versions, we now fall back to using `_optimize_trace_graph_no_onnx_operator`.

`utils.py`: 
- Added a compatibility wrapper `_is_caffe2_aten_fallback()` that returns False for PyTorch 2.x (since Caffe2 support was removed) and defers to `symbolic_helper.is_caffe2_aten_fallback()` for older versions.
- Updated the call to `_jit_pass_onnx_unpack_quantized_weights` to use the new 2-argument signature in PyTorch 2.x (the third argument for Caffe2 fallback was removed).