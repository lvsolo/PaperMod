---
title: "Usage for vscode"
date: "2024-03-14"
author: "lvsolo"
tags: ["ubuntu", "linux", "vscode"]
---
# 1. c++ vscode debug 

    1) The "preLaunchTask" tag in launch.json should be the same as the "label" tag in tasks.json.
    2) The "g++ -g" tag means debug mode, which will enable the breakpoints in the cpp file/program.

## 1.1 launch.json
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
## 1.2 tasks.json
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

# 2. python vscode debug 
{
    // Use IntelliSense to learn about possible attributes.
    // Hover to view descriptions of existing attributes.
    // For more information, visit: https://go.microsoft.com/fwlink/?linkid=830387
    "version": "0.2.0",
    "configurations": [
        {
            "name": "TrainPlutoMini",
            "type": "debugpy",
            "request": "launch",
            "python": "/home/wyz/anaconda3/envs/pluto/bin/python",
            "cwd": "${workspaceFolder}/pluto",
            "program": "run_training.py",
            "console": "integratedTerminal",
            "args": [
                "py_func=train",
                "+training=train_pluto",
                "worker=single_machine_thread_pool",
                "worker.max_workers=4",
                "scenario_builder=nuplan_mini",
                "cache.cache_path=/home/wyz/nuplan/exp/sanity_check",
                "cache.use_cache_without_dataset=true",
                "data_loader.params.batch_size=4",
                "data_loader.params.num_workers=1",
                "model.use_hidden_proj=true",
                "+custom_trainer.use_contrast_loss=true"
                // "lightning.trainer.params.strategy=ddp_find_unused_parameters_true"
            ],
            "env": {
                "CUDA_VISIBLE_DEVICES": "0"
            }
        },
        {
            "name": "SimulationPlutoMini",
            "type": "debugpy",
            "request": "launch",
            "python": "/home/wyz/anaconda3/envs/pluto/bin/python",
            "cwd": "${workspaceFolder}/pluto",
            "program": "run_simulation.py",
            "console": "integratedTerminal",
            "args": [
                "+simulation=closed_loop_nonreactive_agents",
                "planner=pluto_planner",
                "scenario_builder=nuplan_mini",
                "scenario_filter=mini_demo_scenario",
                "worker=sequential",
                "verbose=true",
                "experiment_uid=pluto_planner/mini_demo_scenario",
                "planner.pluto_planner.render=true",
                "planner.pluto_planner.planner_ckpt=/home/wyz/workspace/learning/planing/pluto/checkpoints/pluto_1M_aux_cil.ckpt",
                "+planner.pluto_planner.save_dir=output"
            ],
            "env": {
                "CUDA_VISIBLE_DEVICES": "0"
            }
        }
    ]
}