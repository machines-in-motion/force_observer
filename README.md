# Description
MPC using force estimation for contact tasks  

# Dependencies
- [robot_properties_kuka](https://github.com/machines-in-motion/robot_properties_kuka)
- [Crocoddyl](https://github.com/loco-3d/crocoddyl) 
- [Sobec](https://github.com/skleff1994/sobec/tree/devel) (to be removed soon thanks to Crocoddyl v2.0)
- [Pinocchio](https://github.com/stack-of-tasks/pinocchio)
- [PyBullet](https://pybullet.org/wordpress/)  
- [bullet_utils](https://github.com/machines-in-motion/bullet_utils) 

# Install the C++ code (with optional bindings)
`git clone` this repo

`git submodule update --init`

`mkdir build && cd build`

`cmake .. -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX=[put_here_the_install_dir_you_want]`

`make && sudo make install`

# How to use it
In `demos` run the contact or sanding task script, e.g. `python sanding_mpc.py`. You can modify the corresponding config file, e.g. `sanding_mpc.yml`.

Run the unit test from the `build` folder by running `ctest -v`

Import the python bindings of C++ classes with `import force_observer`
