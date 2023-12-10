#set the periodic task
#service cron start && crontab -e 
hugo server --themesDir ../ --disableFastRender --baseURL="https://redfish-regular-cattle.ngrok-free.app/"
