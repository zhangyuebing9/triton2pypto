# Submodule 初始化脚本
# 网络问题导致自动添加失败时，请手动执行：

set -e

echo "Adding triton as submodule..."
git submodule add https://github.com/triton-lang/triton.git third_party/triton

echo "Adding pypto as submodule..."
git submodule add https://github.com/hw-native-sys/pypto.git third_party/pypto

echo "Initializing submodules..."
git submodule update --init --recursive

echo "Done!"