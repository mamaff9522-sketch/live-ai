# Bundled assets

- `rnnoise-std.rnnn` — the standard RNNoise speech-denoising model from Xiph's RNNoise
  (https://github.com/xiph/rnnoise, BSD-3-Clause), in the format of FFmpeg's `arnndn` filter
  (copied from https://github.com/richardpl/arnndn-models, `std.rnnn`). Used by the studio's
  "Dialogue Only" audio mode. Runs locally; no network calls.
