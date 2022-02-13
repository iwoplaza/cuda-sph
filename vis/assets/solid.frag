#version 460

out vec4 outFragColor;

layout (location = 10) uniform vec4 uColor;

void main()
{
    outFragColor = uColor;
}
