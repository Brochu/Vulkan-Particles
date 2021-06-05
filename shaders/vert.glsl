#version 450
#extension GL_ARB_separate_shader_objects : enable

layout(binding = 0) uniform UniformBufferObject
{
    mat4 model;
    mat4 view;
    mat4 proj;

    vec4 time;
} ubo;

layout(location = 0) in vec2 inPosition;
layout(location = 1) in vec3 inColor;
layout(location = 2) in mat4 instancedTransform;

layout(location = 1) out vec3 fragColor;

void main()
{
    //gl_Position = ubo.proj * ubo.view * ubo.model * vec4(inPosition, 0.0, 1.0);
    gl_Position = ubo.proj * ubo.view * instancedTransform * ubo.model * vec4(inPosition, 0.0, 1.0);

    fragColor = inColor;
    //fragColor = mix(vec3(1, 1, 1), inColor, abs(cos(ubo.time.y)));
}

