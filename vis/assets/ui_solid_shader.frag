#version 460

layout (location = 2) uniform vec4 uColor;

out vec4 outFragColor;

void main()
{
    outFragColor = uColor;
}
