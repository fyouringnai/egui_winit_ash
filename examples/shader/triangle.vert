#version 460

layout(location = 0) in vec3 positions;

layout(location = 0) out vec3 fragColor;

layout(push_constant) uniform PushConstants
{
    vec3 color[3];
}
constants;

void main()
{
    fragColor = constants.color[gl_VertexIndex];
    gl_Position = vec4(positions, 1.0);
}
