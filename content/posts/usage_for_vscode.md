---
title: "Usage for vscode"
date: "2024-03-14"
author: "lvsolo"
tags: ["ubuntu", "linux", "vscode"]
---
#1. c++ vscode debug 
    1) The "preLaunchTask" tag in launch.json should be the same as the "label" tag in tasks.json.
    2) The "g++ -g" tag means debug mode, which will enable the breakpoints in the cpp file/program.
##1.1 launch.json
```
{
    // Use IntelliSense to learn about possible attributes.
    // Hover to view descriptions of existing attributes.
    // For more information, visit: https://go.microsoft.com/fwlink/?linkid=830387
    "version": "0.2.0",
    "configurations": [
        {
        "name": "C++ Launch",
        "type": "cppdbg",
        "request": "launch",
        "program": "${workspaceRoot}/test",
        "stopAtEntry": false,
        //"customLaunchSetupCommands": [
        //  { "text": "target-run", "description": "run target", "ignoreFailures": false }
        //],
        //"launchCompleteCommand": "exec-run",
        "linux": {
          "MIMode": "gdb",
          "miDebuggerPath": "/usr/bin/gdb"
        },
        //"osx": {
        //  "MIMode": "lldb"
        //},
        //"windows": {
        //  "MIMode": "gdb",
        //  "miDebuggerPath": "C:\\MinGw\\bin\\gdb.exe"
        //}
        "preLaunchTask": "vsdebug", 
        "cwd": "${workspaceRoot}"
}

    ]
}
```
##1.2 tasks.json
```
{
    "version": "2.0.0",
    "tasks": [
        {
            "label": "vsdebug",
            "type": "shell",
            "command": "g++  -g -O0 -o test 2-8.cpp",
            "args": [],
            "group": {
                "kind": "build",
                "isDefault": true
            }
        }
    ]
}
```
