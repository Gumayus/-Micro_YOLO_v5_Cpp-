#!/bin/bash
# 修复 Windows 到 WSL 的中文乱码问题

echo "========== 开始修复中文乱码 =========="
echo "检测到从 Windows 迁移到 WSL 导致的 GBK 转 UTF-8 问题"

# 备份当前状态
backup_dir="/tmp/micro_yolo_backup_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$backup_dir"
echo "创建备份到: $backup_dir"
cp -r . "$backup_dir/" 2>/dev/null

# 第一步：检查并转换文件编码
echo "步骤 1/4: 转换文件编码 (GBK → UTF-8)"

# 安装必要的工具
sudo apt-get update >/dev/null 2>&1
sudo apt-get install -y iconv dos2unix >/dev/null 2>&1

# 转换所有源码文件
converted_files=0
failed_files=0

find . -type f \( -name "*.cpp" -o -name "*.h" -o -name "*.hpp" -o -name "*.txt" -o -name "*.md" \) | while read -r file; do
    # 跳过 .git 目录
    if [[ "$file" == *"/.git/"* ]]; then
        continue
    fi
    
    # 检查文件编码
    encoding=$(file -bi "$file" 2>/dev/null | sed -e 's/.*charset=//' -e 's/.*charset=//')
    
    if [[ "$encoding" == "gbk" || "$encoding" == "gb2312" || "$encoding" == "iso-8859-1" ]]; then
        echo "转换: $file (${encoding} → UTF-8)"
        # 备份原始文件
        cp "$file" "${file}.bak"
        
        # 尝试转换
        if iconv -f GBK -t UTF-8 "$file" > "${file}.utf8" 2>/dev/null; then
            mv "${file}.utf8" "$file"
            ((converted_files++))
            
            # 清理 Windows 换行符
            if grep -q $'\r' "$file"; then
                sed -i 's/\r$//' "$file"
            fi
        else
            # 尝试其他编码
            if iconv -f GB2312 -t UTF-8 "$file" > "${file}.utf8" 2>/dev/null; then
                mv "${file}.utf8" "$file"
                ((converted_files++))
            else
                echo "  警告: $file 转换失败，可能不是文本文件"
                ((failed_files++))
            fi
        fi
        rm -f "${file}.bak"
    fi
done

echo "已转换 ${converted_files} 个文件，${failed_files} 个文件失败"

# 第二步：修复 Git 配置
echo "步骤 2/4: 配置 Git 编码设置"

# 确保 Git 使用 UTF-8
git config --local core.quotepath false
git config --local i18n.commitencoding utf-8
git config --local i18n.logoutputencoding utf-8
git config --local core.autocrlf input

# 创建 .gitattributes 文件
cat > .gitattributes << 'EOF'
* text=auto
*.cpp text charset=utf-8 eol=lf
*.h text charset=utf-8 eol=lf
*.hpp text charset=utf-8 eol=lf
*.txt text charset=utf-8 eol=lf
*.md text charset=utf-8 eol=lf
*.py text charset=utf-8 eol=lf
EOF

# 第三步：修复已提交的乱码（如果需要）
echo "步骤 3/4: 检查 Git 历史中的乱码"

# 检查最近提交中是否有乱码
if git log --oneline -10 | grep -P "[\x80-\xFF]" > /dev/null; then
    echo "检测到 Git 历史中有乱码字符"
    echo "需要重写 Git 历史吗？(y/n，注意：这会改变提交哈希)"
    read -r response
    if [[ "$response" =~ ^[Yy]$ ]]; then
        # 使用 git filter-branch 修复历史
        echo "正在重写 Git 历史（这可能需要一些时间）..."
        
        # 临时保存当前修改
        git stash
        
        # 使用 git-filter-repo（如果已安装）
        if command -v git-filter-repo &> /dev/null; then
            git filter-repo --force \
              --path-glob '*.cpp' \
              --path-glob '*.h' \
              --path-glob '*.txt' \
              --blob-callback '
                import sys
                try:
                    content = blob.data.decode("gbk").encode("utf-8")
                    blob.data = content
                except:
                    try:
                        content = blob.data.decode("utf-8")
                        blob.data = content.encode("utf-8")
                    except:
                        pass
                '
        else
            echo "建议安装 git-filter-repo 来重写历史："
            echo "sudo apt-get install -y git-filter-repo"
        fi
        
        # 恢复修改
        git stash pop 2>/dev/null || true
    fi
fi

# 第四步：验证修复
echo "步骤 4/4: 验证修复结果"

# 创建测试文件
cat > test_encoding.cpp << 'EOF'
// 测试文件 - 验证中文编码
// 中文注释：这是 Micro YOLO v5 C++ 项目
// 测试字符：áéíóú àèìòù
// 特殊符号：★☆◆◇
#include <iostream>

int main() {
    std::cout << "中文输出测试" << std::endl;
    return 0;
}
EOF

echo "创建测试文件: test_encoding.cpp"

# 检查编码
echo "测试文件编码: $(file -bi test_encoding.cpp)"
echo "中文显示测试:"
cat test_encoding.cpp | head -5

# 清理
rm -f test_encoding.cpp

echo "========== 修复完成 =========="
echo "已执行的操作："
echo "1. 转换了 ${converted_files} 个文件的编码（GBK → UTF-8）"
echo "2. 配置了 Git 使用 UTF-8 编码"
echo "3. 创建了 .gitattributes 文件"
echo ""
echo "接下来请执行："
echo "1. 检查文件是否还有乱码: cat 某个.cpp文件"
echo "2. 提交修复: git add ."
echo "3. 提交: git commit -m 'fix: 修复Windows到WSL的中文乱码问题'"
echo "4. 推送: git push"
echo ""
echo "备份位置: $backup_dir"
echo "==============================="