#version 460

layout (location = 0) in vec3 inPos;

layout (location = 0) uniform mat4 uProjMat;
layout (location = 1) uniform mat4 uModelMat;
layout (location = 2) uniform mat4 uViewMat;

void main()
{
    gl_Position = uProjMat * uViewMat * uModelMat * vec4(inPos, 1.0);
}
