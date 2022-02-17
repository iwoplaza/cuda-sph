#version 460

layout (location = 0) in float unusedButNecessary;
layout (location = 1) in vec3 inOffset;
layout (location = 2) in float inDensity;


layout (location = 0) uniform mat4 uProjMat;
layout (location = 1) uniform mat4 uModelMat;
layout (location = 2) uniform mat4 uViewMat;

out vec3 vColor;

void main()
{
    gl_Position = uProjMat * uViewMat * uModelMat * vec4(inOffset, 1.0);

//    vColor = mix(vec3(0.2, 0.7, 1), vec3(1, 0, 0), inDensity/10.0);
    float t = min(inDensity/10.0, 1);
//    t = 1-pow(1-t, 2); // inv-squared
    vColor = mix(vec3(0.1, 0.7, 0.5), vec3(0.3, 0.4, 1) * 0.7, t);
    gl_PointSize = min(max(1, (40 + (1-t)*20)/gl_Position.z), 13);
}
