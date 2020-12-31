import numpy as np
from mctspy.tree.nodes import TwoPlayersGameMonteCarloTreeSearchNode
from mctspy.tree.search import MonteCarloTreeSearch
from mctspy.games.examples.fourinrow import FourInRowGameState
from mctspy.games.examples.fourinrow import FourInRowMove
import threading,time
from functools import partial
from multiprocessing import Pool

import pgzrun
pool = Pool(8)
state = np.zeros((7,7))
board_state = FourInRowGameState(state = state, next_to_move=1)
best_node = TwoPlayersGameMonteCarloTreeSearchNode(state = board_state)
gameturn = 0
HEIGHT =800
WIDTH  =800
TITLE = "7*7四子棋by唐老师"
player = 1


def highbit(n):
    n |= (n >>  1)
    n |= (n >>  2)
    n |= (n >>  4)
    n |= (n >>  8)
    n |= (n >> 16)
    return n - (n >> 1)

def setBg(i,j):
    bg = Actor("4row_board")
    bg.pos = i*100,j*100
    return bg
bgs = [setBg(i,j) for i in range(1,8) for j in range(1,8)]
chs = []
anims=[]
winner = 0
costtime = 0
aiturn = False
def draw():
    screen.fill("skyblue")
    for bg in bgs:
        bg.draw()
    for ch in chs:
        ch.draw()
    if winner == 1:
        screen.draw.text("Black Win~",fontsize = 100,center=(400,400))
    if winner == -1:
        screen.draw.text("Red Win~",fontsize = 100,center=(400,400))
    if aiturn:
        screen.draw.text(f"AI time:{costtime}",fontsize = 40, topleft = (0,10))
   
def add_new(action):
    new = None
    if action.value == 1:
        new = Actor("4row_black")
    else:
        new = Actor("4row_red")
    new.pos = (action.y_coordinate+1)*100,0
    new.target = (action.x_coordinate+1)*100
    new.anim = anim
    chs.append(new)
    anims.append(new)
    clock.schedule(anim,0.05)
     
def anim():
    if anims:
        new = anims.pop(0)
        animate(new,tween="bounce_end",duration = 0.4,y = new.target,
                on_finished = get_result)

def get_result():
    global winner
    winner = best_node.state.game_result

def get_next(playerAction=None,level=1680):
    global best_node, aiturn, costtime
    costtime = 0
    if playerAction:
        found = False
        for child in best_node.children:
            if  child.action == playerAction:
                best_node = child
                best_node.parent = None
                print("成功提取记忆")
                found = True
                break
        if not found:
            best_node = TwoPlayersGameMonteCarloTreeSearchNode(
                best_node.state.move(playerAction))
  
    if not best_node.is_terminal_node():
        mcts = MonteCarloTreeSearch(best_node,pool)
        best_node = mcts.best_action(level)
        add_new(best_node.action)
        #get_next2()
    aiturn = False
    
def get_next2():
    local_node = TwoPlayersGameMonteCarloTreeSearchNode(
            best_node.state)
    if not local_node.is_terminal_node():
        mcts = MonteCarloTreeSearch(local_node,pool)
        local_node = mcts.best_action2(3200)
        add_new(local_node.action)
        get_next(local_node.action)
        
def update(delta):
    #return
    global costtime
    if aiturn:
        costtime += delta

def on_mouse_down(pos):
    global best_node ,aiturn,gameturn
    levels = [1680]
    if not aiturn:
        moves = best_node.state.get_legal_actions()
        col = round(pos[0]/100) - 1
        for move in moves:
            if move.y_coordinate == col:
                playerAction = FourInRowMove(move.x_coordinate,move.y_coordinate,move.value)
                add_new(playerAction)
                if gameturn < len(levels):
                    threading.Thread(target = get_next,args=(playerAction,levels[gameturn])).start()
                else:
                    threading.Thread(target = get_next,args=(playerAction,)).start()
                gameturn+=1
                aiturn = True
                return
aiturn = True
threading.Thread(target = get_next,args=(None,420)).start()
pgzrun.go()
pool.close() 
