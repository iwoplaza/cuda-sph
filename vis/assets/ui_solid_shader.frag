#version 460

layout (location = 2) uniform vec3 uColor;

out vec4 outFragColor;

void main()
{
    outFragColor = vec4(1, 0, 0, 1);
}
