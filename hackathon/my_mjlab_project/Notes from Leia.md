Drew it in Onshape (grabed parts and did the assmbly - click on child first and then parent)
Make sure all of the joint names start with dof_....
Run python code (give access keyt etc) - see Leia's Notion - tab called "Onshape to Robot"
It will pull over masses inertias, joint limits in Onshape - and it will pull them over - or you can edit the xml
XML has PD controllers for your motor (added post Onshape)
Edit tasks.commands has 
Edit ur2_reach_env_cfg to change the reward structure