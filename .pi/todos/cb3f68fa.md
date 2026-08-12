{
  "id": "cb3f68fa",
  "title": "CUDA Fase 3: linear/MLP/patch (cuore DiT)",
  "tags": [],
  "status": "open",
  "created_at": "2026-08-12T13:05:52.266Z"
}

- linear_bf16 (tile 16x16 shared memory, fma esplicito — stesso ordine), mlp_bf16 fuso (fc1->swiglu->fc2), mlp_nax_bf16, swiglu_bf16
- linear_f32, swiglu_f32, scale_add_f32, copy_bf16/f32 (cudaMemcpy device-to-device)
- patch_linear_bf16 + map (96/32->5376), patch_linear offset views
- Bit-exact con le reference dei test esistenti
