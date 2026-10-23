# Implementation 1 performance record

Measured on 23 October 2026 at 11:42 IST. This folder records the first implementation's performance for comparison with later implementations.

| Backend | CPU threads | Median compute time | Median native call time | Observed Unity Editor FPS |
| --- | ---: | ---: | ---: | ---: |
| CPU single | 1 | 2018.73 ms | 2019.15 ms | 0.49 |
| CPU multi | 12 | 282.76 ms | 283.24 ms | 3.35 |
| CUDA GPU | 0 | 3.63 ms | 4.20 ms | 145.94 |

The CUDA row's CPU thread count refers to the renderer's reported CPU worker count. Unity and the CUDA driver still use the CPU.

## Scene and implementation

- Resolution: 320 × 240, or 76,800 primary rays per frame.
- Geometry: 982 triangles, with Suzanne inside the room and a bright blue back wall.
- Camera: position 0, 2.3, -8.5; pitch 2.69428468 degrees; yaw and roll zero; vertical field of view 42 degrees.
- Algorithm: one ray through each pixel centre, with every ray tested against every triangle. The nearest hit supplies the base colour.
- Shading: flat triangle normals, a fixed light direction, and a constant ambient term. The ceiling panel appears white but does not illuminate the scene.
- No shadow rays, reflections, indirect illumination, anti-aliasing, or acceleration structure.
- All three backends use the same intersection and shading functions. Multicore mode distributes image rows through OpenMP.

## Hardware and measurement

- CPU: AMD Ryzen 5 5600H, with 12 logical processors available.
- GPU: NVIDIA GeForce GTX 1650.
- Unity: 6000.3.26f1, with Vulkan used for Unity's display work. Ray tracing runs in CUDA in GPU mode.
- Game view: 1286 × 799. The rendered image remains 320 × 240.
- VSync: disabled. Target frame rate: unlimited.
- Each mode has 3 warm-up frames followed by 30 consecutive measured frames, in single-thread CPU, multicore CPU, then CUDA order.
- FPS is completed frames divided by elapsed wall time in Unity Editor. It includes frame pacing and Unity overhead. It is not the reciprocal of median compute time and is not standalone player FPS.
- CPU compute time uses a wall clock. CUDA compute time uses CUDA events around the kernel. Native call time includes synchronization and GPU readback, but excludes the later texture upload and display.
- Scene upload and initialization are excluded. These results come from one initial batch. Other applications, power conditions, and Editor activity can affect a later run.

All three captured PNGs are byte-identical. The scene comparison also reports zero differing pixels and a maximum channel difference of zero. The native verification checks passed, and Unity reported zero errors and zero warnings after capture.

## Saved data

- [Raw frame measurements](frames.csv)
- [Full results and settings](results.json)
- Rendered images: [single-thread CPU](backend-0.png), [multicore CPU](backend-1.png), and [CUDA](backend-2.png)
- [Unity Game view](unity-game-view.png)

To capture another run, start Play mode with a fixed camera and run `python3 Tools/benchmark_unity.py scratch/new-performance-run` from the project root. Leave the camera and backend controls untouched until capture completes. Use a fresh output directory and preserve this folder for comparison.
