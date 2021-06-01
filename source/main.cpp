#include <stdio.h>

#include "VulkanInitializers.h"

int main(int argc, char** argv)
{
    auto test = vks::initializers::submitInfo();
    printf("%i\n", test.sType);

    return 0;
}
