#include <exception>

#include "VulkanApp.h"

int main(int argc, char** argv)
{
    VulkanApp app;

    try
    {
        app.run();
    }
    catch(std::runtime_error e)
    {
        printf("=> FATAL: %s\n", e.what());
        return EXIT_FAILURE;
    }
    return EXIT_SUCCESS;
}
