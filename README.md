# sam-attention-fallback
A modified version of the SAM2 transformer module with a manual implementation of scaled dot-product attention. This removes dependency on CUDA-specific kernels and ensures consistent behavior across all hardware configurations. Includes both a deterministic fallback-free version and a conditional fallback variant.
