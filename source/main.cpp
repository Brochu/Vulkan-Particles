#include <exception>

#include "VulkanApp.h"

int main(int argc, char** argv)
{
    VulkanApp app;

    try
    {
        app.run();
    }
    catch(std::exception e)
    {
        std::cerr << e.what() << std::endl;
        return EXIT_FAILURE;
    }
    return EXIT_SUCCESS;
}
