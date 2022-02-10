#version 460

in vec2 vUV;

layout (binding = 0) uniform sampler2D uTexture;
layout (location = 2) uniform vec3 uTextColor;

out vec4 outFragColor;

void main()
{
    vec2 uv = vUV.xy;
    float text = texture(uTexture, uv).r;
    outFragColor = vec4(uTextColor.rgb * text, text);
}
