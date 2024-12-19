---
title: "PNC Utils"
date: "2024-12-14"
author: "lvsolo"
tags: ["PNC", "Nuplan", "planning"]
---

1. nuboard UI module create
   ```
    #main_callbacks.on_run_simulation_end()
    map_version = "nuplan-maps-v1.0"
    scenario_mapping = ScenarioMapping(scenario_map=get_scenario_map(), subsample_ratio_override=0.5)
    #builder = cfg.scenario_builder
    builder = NuPlanScenarioBuilder('/mnt/HDD/dataset/nuplan/', '/mnt/HDD/dataset/nuplan/mini/maps', None, None, 'nuplan-maps-v1.1', scenario_mapping=scenario_mapping)
    output_dir = '/mnt/HDD/dataset/nuplan/mini/exp/exp/simulation/closed_loop_nonreactive_agents/pluto_planner/mini_demo_scenario/'
    simulation_file = [str(file) for file in pathlib.Path(output_dir).iterdir() if file.is_file() and file.suffix == '.nuboard']
    nuboard = NuBoard(
        nuboard_paths=simulation_file,
        scenario_builder=builder,#scenario_builder,
        #scenario_builder=runners[0].scenario,#scenario_builder,
        vehicle_parameters=get_pacifica_parameters(),
        port_number=5006
    )
    nuboard.run()

   ```