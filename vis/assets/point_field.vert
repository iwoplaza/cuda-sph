#version 460

layout (location = 0) in vec3 inPos;

layout (location = 0) uniform mat4 uModelMat;
layout (location = 1) uniform mat4 uProjMat;

layout (location = 2) uniform vec3 uOffsets[100];

void main()
{
    vec3 offset = uOffsets[gl_InstanceID];

    gl_Position = uProjMat * uModelMat * vec4(inPos + offset, 1.0);
    gl_PointSize = 5;
}
