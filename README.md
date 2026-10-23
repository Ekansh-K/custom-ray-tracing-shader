# Custom ray tracing shader

This is my attempt at a custom ray tracing shader, inspired by [Sebastian Lague's ray tracing videos](https://www.youtube.com/watch?v=Qz0KTGYJtUk) and [his project](https://github.com/SebLague/Ray-Tracing).

I use Unity for the scenes and camera, with C++ and CUDA for the renderer. I'm comparing single-core CPU, multicore CPU, CUDA, and Embree, with progressive rendering, BVH traversal, Sobol sampling, and optional OptiX AI denoising.

![My ray tracer rendering the dragon room](docs/images/dragon-room.jpg)


