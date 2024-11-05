# Multi-Agent Dynamic Relational Reasoning for Social Robot Navigation Codebase

The following codebase is the implementation of the paper "Multi-Agent Dynamic Relational Reasoning for Social Robot Navigation". Containing the following files:

```
.
├── trajectory_predictor/ - Contains the code for the trajectory predictor
│   ├── encoder.py - Trajectory encoder and customized MLP implementation
│   ├── decoder.py - Trajectory decoder
│   └── model.py - Main model structure
├── robot_nagivation/ - Contains the code for the robot navigation
│   ├── attn.py - Include the attention mechanism and edge attention implementation
│   ├── embedding.py - Include the human-human, human-robot, embedding layer implementation
│   ├── network.py - Main network structure
│   └── policy.py - Policy network
└── src/ - Contains the code for env, baseline and utils
    ├── env/
    │   ├── env.py - Main environment implementation, including reset, step and utils
    │   ├── states.py - Human and robot state class implementation
    │   └── render.py - Rendering implementation
    ├── orca.py - ORCA implementation
    ├── ppo.py - PPO pipeline implementation
    └── utils.py - General utility functions
```
