# Description
Force-feedback MPC based on online estimation. This repo implements custom residual, action model and estimator in C++ (with Python bindings) based on the Crocoddyl library. This is meant to be used as a plugin to reproduce the work described in [this publication](https://hal.science/hal-04564888v1/document).

# Dependencies
## Core (C++/Python bindings)
- [Crocoddyl](https://github.com/loco-3d/crocoddyl) 
- [Pinocchio](https://github.com/stack-of-tasks/pinocchio)
- [Proxsuite](https://github.com/Simple-Robotics/proxsuite)

## Demos (Python)
- [croco_mpc_utils](https://github.com/machines-in-motion/mim_robots)
- [mim_robots](https://github.com/machines-in-motion/mim_robots)
- [PyBullet](https://pybullet.org/wordpress/)  
- matplotlib 
- PyYAML 

# Install from source
## Setup environment
You can optionally use conda to setup your work environment
`conda create -n force_observer`
`conda activate force_observer`
`conda install -c conda-forge mim-solvers cmake proxsuite`
`conda install conda-forge::pyyaml`
`conda install matplotlib`
`conda install conda-forge::pybullet`
Also check out `environment.yaml` to full conda environment. 

## Build and install
Then clone and build / install the code
`git clone git@github.com:machines-in-motion/force_observer.git`
`git submodule update --init`
`mkdir build && cd build`
`cmake .. -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX={INSTALL_DIR}`
`make && sudo make install`
To install inside the conda environment, you can use `-DCMAKE_INSTALL_PREFIX=$CONDA_PREFIX` (environment must be activated).

# How to use it
In `demos` run the contact or sanding task script, e.g. `python sanding_mpc.py`. You can modify the corresponding config file, e.g. `sanding_mpc.yml`.

Run the unit test from the `build` folder by running `ctest -v`

Import the python bindings of C++ classes with `import force_observer`

# Citing this repo
```
@INPROCEEDINGS{10611156,
  author={Jordana, Armand and Kleff, Sébastien and Carpentier, Justin and Mansard, Nicolas and Righetti, Ludovic},
  booktitle={2024 IEEE International Conference on Robotics and Automation (ICRA)}, 
  title={Force Feedback Model-Predictive Control via Online Estimation}, 
  year={2024},
  volume={},
  number={},
  pages={11503-11509},
  keywords={Torque;Systematics;Force;Force feedback;Estimation;Robot sensing systems;Force sensors},
  doi={10.1109/ICRA57147.2024.10611156}}
```
```
A. Jordana, S. Kleff, J. Carpentier, N. Mansard and L. Righetti, "Force Feedback Model-Predictive Control via Online Estimation," 2024 IEEE International Conference on Robotics and Automation (ICRA), Yokohama, Japan, 2024, pp. 11503-11509, doi: 10.1109/ICRA57147.2024.10611156. keywords: {Torque;Systematics;Force;Force feedback;Estimation;Robot sensing systems;Force sensors},
```