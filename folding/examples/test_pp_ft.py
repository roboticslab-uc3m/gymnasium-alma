#!/usr/bin/env python

import gymnasium as gym
import gymnasium_playground_folding
import numpy as np
import math



def quaternion_to_euler(q):
    # q es un cuaternión [x, y, z, w]
    x, y, z, w = q

    # Calculando los ángulos de Euler (Roll, Pitch, Yaw)
    # Fórmulas estándar para la conversión de cuaternión a Euler (rotación ZYX)
    roll = math.atan2(2.0 * (w * x + y * z), 1.0 - 2.0 * (x**2 + y**2))
    pitch = np.arcsin(2.0 * (w * y - z * x))
    yaw = math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y**2 + z**2))

    # Convertimos de radianes a grados
    roll = np.degrees(roll)
    pitch = np.degrees(pitch)
    yaw = np.degrees(yaw)

    return np.array([roll, pitch, yaw])

env = gym.make('gymnasium_playground/Folding-v0',
               render_mode='human',  # "human", "text", None
               inFileLabelsStr='../assets/labels.txt',
               inFileImgStr='../assets/gymnasium_playground_Folding-v0.png',
               initX=2,
               initY=2)
ob, _ = env.reset()
env.render()






rest = [0.47807414925570263, -1.1729907312031171, 2.662614000377364, 1.5985649970946019, 1.371503715463566, -1.1855814127187034, -2.156952310975745]


pick = [1.2444611163820585, -0.02854918431637237, 2.59218800826213, 0.6807260978938166, 1.8350994558873004, -1.1662732603877193, -1.9479233135472014]
ppick = [1.2005119838597003, -0.35846680845541423, 2.5544515803371732, 0.8656959547715106, 1.8543853766093832, -1.2457364411381102, -1.7876482344397264]
pplace = [0.8436112797531792, -0.7197547558084954, 2.32800233276473, 1.8398020156099957, 1.921113550727344, -0.7571412565045983, -1.9324130610702297]
place = [0.8910425948523524, -0.30756865079077755, 2.434876191851386, 1.7341246774087653, 1.7688937245785257, -0.5495851031657811, -2.074445252439919]

p_pick = env.chain.fk(np.array(pick))
p_place = env.chain.fk(np.array(place))
p_ppick = env.chain.fk(np.array(ppick))
p_pplace = env.chain.fk(np.array(pplace))
p_rest = env.chain.fk(np.array(rest))

print("p_pick",p_pick)
print("p_place",p_place)
print("p_ppick",p_ppick)
print("p_pplace",p_pplace)
print("p_rest",p_rest)

dnm = 1
phase = 0
ac = np.ones(8)
z_ft = 0
ft_limit = 8

for i in range(dnm):
    ob, *_ =env.step(ac)
    env.render()

for i in range(10000000000000):
    # pre pick
    if phase == 0:
        ac[:7] = p_ppick
        if np.linalg.norm(ob[:7] - p_ppick) < 0.01:
            phase = 1
            z_ft = env.get_ft()[-1]

    # pick
    elif phase == 1:
        new_ft =env.get_ft()[-1]
        ac[:7] = p_pick
        print("new_ft",new_ft, "z_ft", z_ft, "diff", np.abs(new_ft - z_ft))
        if new_ft > 0:
            phase = 11
            ac[:7] = ob[:7]

    elif phase == 11:
        ac[-1] = 0
        if np.abs(ob[-1] - ac[-1]) == 0:
            phase = 12
            cc = 0
    elif phase == 12:
        cc +=1
        if cc > 100:
            phase = 2

    # prepick
    elif phase == 2:
        ac[:7] = p_ppick
        if np.linalg.norm(ob[:7] - p_ppick) < 0.01:
            phase = 3

    # pre place
    elif phase == 3:
        ac[:7] = p_pplace
        if np.linalg.norm(ob[:7] - p_pplace) < 0.01:
            phase = 4
            z_ft = env.get_ft()[-1]

    # place
    elif phase == 4:
        new_ft =env.get_ft()[-1]
        ac[:7] = p_place
        print("new_ft",new_ft, "z_ft", z_ft, "diff", np.abs(new_ft - z_ft))
        if new_ft  >0:
            phase = 44
            ac[:7] = ob[:7]

    elif phase == 44:
        ac[-1] = 1
        if np.abs(ob[-1] - ac[-1]) < 0.001:
            cc = 0
            phase = 45
    elif phase == 45:
        cc +=1
        if cc > 100:
            phase = 5

    # pre place
    elif phase == 5:
        ac[:7] = p_pplace
        if np.linalg.norm(ob[:7] - p_pplace) < 0.01:
            phase = 6

    # rest
    elif phase == 6:
        ac[:7] = p_rest

    
    ob, *_ =env.step(ac)
    # raise
    env.render()




