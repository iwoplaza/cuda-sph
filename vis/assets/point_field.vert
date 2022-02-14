#version 460

layout (location = 0) in float unusedButNecessary;
layout (location = 1) in vec3 inOffset;

layout (location = 0) uniform mat4 uProjMat;
layout (location = 1) uniform mat4 uModelMat;
layout (location = 2) uniform mat4 uViewMat;

void main()
{
    gl_Position = uProjMat * uViewMat * uModelMat * vec4(inOffset, 1.0);
    // gl_PointSize = min(max(1, 20/gl_Position.z), 13);
    gl_PointSize = 13;
}
