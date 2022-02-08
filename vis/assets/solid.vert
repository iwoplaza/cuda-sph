#version 460

layout (location = 0) in vec3 inPos;

layout (location = 0) uniform mat4 uModelMat;
layout (location = 1) uniform mat4 uProjMat;

void main()
{
    gl_Position = uProjMat * uModelMat * vec4(inPos, 1.0);
}
