#version 460

layout (location = 0) in vec2 inPos;
layout (location = 1) in vec2 inUV;

out vec2 vUV;

layout (location = 0) uniform mat4 uProjMat;
layout (location = 1) uniform mat4 uModelMat;

void main()
{
    vUV         = inUV.xy;
    gl_Position = uProjMat * uModelMat * vec4(inPos.xy, 0.0, 1.0);
}
