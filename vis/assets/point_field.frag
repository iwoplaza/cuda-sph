#version 460

out vec4 outFragColor;

void main()
{
    vec2 circCoord = 2.0 * gl_PointCoord - 1.0;
    if (dot(circCoord, circCoord) > 1.0) {
        discard;
    }

    outFragColor = vec4(1, 0, 0, 1);
}
