# package_cyg_app

A little app to make a "portable" app from a cygwin package.

This application takes a specified Cygwin package and downloads that package
and all it's dependencies into a minimal system tree for running that
package's applications. It then adds shims to allow running of executables that
are inside that system tree. The resulting package does not require a separate
Cygwin install to function.

_This project is not affiliated with the Cygwin project in any way._
