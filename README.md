# Having Fun with Ray Tracing

Basing this on the Ray Tracing in One Weekend series:

https://raytracing.github.io/

Adding my own twist to stuff as well, like adding openMP support and CUDA-fying the implementations.

Current progress (anti-aliasing and multiple objects, with normals):

![img](https://github.com/user-attachments/assets/e9aba45e-9351-4d70-a37c-aa426da69758)


Cuda version so far only ray traces a single sphere, no normals, but with antialiasing:

![rendered_image](https://github.com/user-attachments/assets/5cb90166-d9b2-432a-81c2-ceed2d2d3306)

UPDATE: Cuda version now handles normals, and ray traces a single sphere with a gradient background. There is a bug with the antialiasing, but without antialiasing following image is generated:

![rendered_image](https://github.com/user-attachments/assets/d61a02ca-d5a3-412c-a8eb-9649b074aab7)

UPDATE: Following this fire repo: https://github.com/rogerallen/raytracinginoneweekendincuda was able to get the CUDA running really well with classes.

![rendered_image](https://github.com/user-attachments/assets/be5e80f9-f26a-442d-9262-7f881939377e)
