#include <cstdlib>
#include <memx/accl/utils/path.h>

namespace fs = std::filesystem;

fs::path MX::Utils::mx_get_home_dir()
{
    fs::path home_path;
    if (const char *env_p = std::getenv("MX_API_HOME"))
    {
        home_path += env_p;
    }
    return home_path;
}

fs::path MX::Utils::mx_get_accl_dir()
{
    fs::path accl_path;
    fs::path home_path = mx_get_home_dir();
    // if (!fs::exists(home_path))
    // {
        accl_path = home_path / "mx_accl";
    // }
    return accl_path;
}
