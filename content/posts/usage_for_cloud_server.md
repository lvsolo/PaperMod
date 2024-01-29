---
title: "Usage for cloud server"
date: "2024-01-29"
author: "lvsolo"
tags: ["cloud server", "ubuntu", "linux", "blog", "hugo"]
---

# 1. using frp to expose local port
referring to :

[frp to expose local port](https://blog.csdn.net/weixin_42265958/article/details/106008837)

# 2. using hugo + nginx to build blog
how to install and use hugo theme PaperMod can be found in my later blog essay:

[tip for hugo themes](/posts/tips_for_hugo_themes)

so how to install nginx and set the DNS web name to my server's IP?
referring to :

[hugo and nginx](sulvblog.cn/posts/blog/hugo_deploy/)

install nginx:
version==1.18.0

the can-be-used nginx.conf is as below, should be copied to /etc/nginx/nginx.conf
```
# 要配置的第一个地方，这里的用户要改成root，不然可能会没有权限
user root;

worker_processes auto;
error_log /var/log/nginx/error.log;
pid /run/nginx.pid;

include /usr/share/nginx/modules/*.conf;

events {
    worker_connections 1024;
}

http {
    log_format  main  '$remote_addr - $remote_user [$time_local] "$request" '
                      '$status $body_bytes_sent "$http_referer" '
                      '"$http_user_agent" "$http_x_forwarded_for"';

    access_log  /var/log/nginx/access.log  main;

    sendfile            on;
    tcp_nopush          on;
    tcp_nodelay         on;
    keepalive_timeout   65;
    types_hash_max_size 2048;

    include             /etc/nginx/mime.types;
    default_type        application/octet-stream;

    include /etc/nginx/conf.d/*.conf;
    
    # 配置http
    server {
        # 要配置的第二个地方，80访问端口
        listen       80 default_server; 
        listen       [::]:80 default_server;
        
        # 要配置的第三个地方，域名
        server_name grass-shoes-worm.online;
        rewrite ^(.*) https://$server_name$1 permanent; #自动从http跳转到https
        # 要配置的第四个地方，这里指向public文件夹
        root /home/ubuntu/git/PaperMod;

        include /etc/nginx/default.d/*.conf;
        
        # 要配置的第五个地方
        location / {
            root /home/ubuntu/git/PaperMod;
            index  index.html index.htm;
        }
        
        # 要配置的第六个地方
        error_page 404 /404.html;
        location = /40x.html {
            root /home/ubuntu/git/PaperMod;
        }

        error_page 500 502 503 504 /50x.html;
            location = /50x.html {
        }
    }
    
    # 配置https
     server {
         listen 443 ssl;
         # 要配置的第七个地方
         server_name grass-shoes-worm.online;
         root /home/ubuntu/git/PaperMod;
         
         # 要配置的第八个地方
         ssl_certificate       /etc/nginx/grass-shoes-worm.online_bundle.crt;    #指定证书位置，默认在当前目录寻找
         ssl_certificate_key   /etc/nginx/grass-shoes-worm.online.key;    #指定私钥位置
         #ssl_certificate /etc/nginx/1_sulvblog.cn_bundle.crt;
         #ssl_certificate_key /etc/nginx/2_sulvblog.cn.key;
         #
         # 要配置的第九个地方，可以按照我的写法
         ssl_session_timeout 10m;
         ssl_protocols TLSv1 TLSv1.1 TLSv1.2;
         ssl_ciphers ECDHE-RSA-AES128-GCM-SHA256:HIGH:!aNULL:!MD5:!RC4:!DHE;
         ssl_prefer_server_ciphers on;
         
         # 要配置的第十个地方
         error_page 404 /404.html;
         location = /404.html {
              root /home/ubuntu/git/PaperMod;
         }

         include /etc/nginx/default.d/*.conf;
     }

}

```
