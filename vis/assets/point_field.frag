#version 460

in vec3 vColor;

out vec4 outFragColor;

void main()
{
    outFragColor = vec4(vColor, 0.5);
}
