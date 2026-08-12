{
  "id": "bccd4b58",
  "title": "CUDA Fase 6: Video VAE + encoder condition",
  "tags": [],
  "status": "open",
  "created_at": "2026-08-12T13:06:19.123Z"
}

- swiglu_f32, scale_add_f32 (già in Fase 3), video_qkv_rope_f32 (già Fase 5)
- vae_encoder_pad_f32 (reflect), vae_encoder_group_norm_silu_f32 (riduzioni 256), conv3d_f32 (NDHWC/OIDHW, fma)
- Test bit-exact conv/pad, tolleranza norm
