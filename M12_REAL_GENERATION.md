# M12 — Real image generation bridge

LocalImage now has a real txt2img execution bridge through stable-diffusion.cpp instead of a placeholder generator.

- SafeTensors imported through Android SAF are reused directly when the fd maps to a regular file.
- Non-path SAF providers are staged to app-private storage only when required by the backend.
- Model loading uses stable-diffusion.cpp C API.
- Backend selection prefers Vulkan and falls back to CPU.
- Parameters are passed from the Material 3 generation screen to the native runtime.
- Returned RGB/RGBA image data is written to a PNG in app-private storage.
- Failed model loading/inference is reported as a real failure; no fake image is produced.

The build fetches the upstream stable-diffusion.cpp source at CMake configure time. This is required because the existing LocalImage tensor/graph layer is not itself a complete SD model implementation.
