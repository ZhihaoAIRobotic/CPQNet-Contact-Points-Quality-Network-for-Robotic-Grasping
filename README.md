# CPQNet
This is code to reproduce experiments for the paper CPQNet: Contact Points Quality Network for Robotic Grasping.

# Running
The steps to collect and run data are as follows. <br>
Collect data by running python data/t_gene_map.py <br>
Complete the map using python data/complete_map.py <br>
Train the CPQNet by running python train/train.py <br> 
Plan grasp using python grasp planning/main_function_loop.py

On Windows you do not link with a .dll file directly – you must use the accompanying .lib file instead. 
You also must make sure that the .dll file is either in the directory contained by the %PATH% environment variable or the project directory.
If you don't have access to the .lib file, one alternative is to load the .dll manually during runtime using WINAPI functions such as LoadLibrary and GetProcAddress.
step:
1. 将lib文件所在的目录添加到 项目--属性--链接器--常规--附加库目录
2. 将lib文件（.lib）添加到 项目--属性--链接器--输入--附加依赖项
3. 将dll添加到工程目录下
4. 将头文件添加到 项目--属性--c/c++--常规--附加包含目录



