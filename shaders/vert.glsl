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
layout(location = 2) in vec4 instancedTransform;

layout(location = 1) out vec3 fragColor;

void main()
{
    gl_Position = ubo.model * vec4(inPosition, 0.0, 1.0); // Change to world space
    gl_Position = gl_Position + instancedTransform; // Instanced move
    gl_Position = ubo.proj * ubo.view * gl_Position; // Rest of transform to screen space

    fragColor = inColor;
    //fragColor = instancedTransform.xyz;
}

