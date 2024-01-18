---
title: "Tips for hugo themes and this blog usage"
author: "lvsolo"
date: "2023-12-09"
tags: ['hugothemes']
---

1. How to auto pull the newest master branch from git?

    use `crontab -e` to set the periodic task, in this case , it's the `auto_pull_from_git.sh` script.


2. The date format?
    the date format must be ****-**-**, for example 2023-12-9 should be 2023-12-09.

3. update the hugo version to latest:
    amd64:
    ```
    wget https://github.com/gohugoio/hugo/releases/download/v0.121.2/hugo_extended_0.121.2_linux-amd64.deb
    sudo dpkg -i hugo_extended_0.121.2_linux-amd64.deb
    ```
    arm64:
    ```
    wget https://github.com/gohugoio/hugo/releases/download/v0.121.2/hugo_extended_0.121.2_linux-arm64.deb
    sudo dpkg -i hugo_extended_0.121.2_linux-arm64.deb
    ```

TODO: [折腾 Hugo & PaperMod 主题](https://dvel.me/posts/hugo-papermod-config/)
