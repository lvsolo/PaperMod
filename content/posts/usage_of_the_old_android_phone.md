

手机用andronix+termux+vnc安装ubuntu系统


电脑连接手机上的termux，使用质感文件管理器可以管理termux中的文件
https://www.jianshu.com/p/2e6c8152a2ba

手机andronix的ubuntu22中安装anaconda
https://zhuanlan.zhihu.com/p/608147907

手机中ubuntu安装hugo+papermode theme
```bash
sudo apt install hugo
hugo new site hugo_themes --format yaml
cd hugo_themes
cd themes
git clone git@github.com:lvsolo/PaperMod.git
```