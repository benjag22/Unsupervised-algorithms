#include <iostream>
#include <vector>
#include <fstream>
#include "DBscan.h"
#include "k-means.h"
#include <stdlib.h>
#include <cmath>
#include <time.h>

using namespace std;

int height_grid, width_grid, action_taken, action_taken2,current_episode;
int blocked[100][100];
float cum_reward,Qvalues[100][100][4], reward[100][100],finalrw[50000];
int init_x_pos, init_y_pos, goalx, goaly, x_pos,y_pos, prev_x_pos, prev_y_pos,i,j,k;
ofstream reward_output;

//////////////
//Setting value for learning parameters
int action_sel=2; // 1 es greedy, 2 es e-greedy

int environment= 2; // 1 es el grid chico, 2 es el precipicio (Cliff walking)
int algorithm = 2; //1 es Q-learning, 2 es Sarsa
int stochastic_actions=1;

int num_episodes=3000;
float learn_rate=0.1;
float disc_factor=0.99;
float exp_rate=0.05;


void Initialize_environment()
{
    if(environment==1)
    {
        height_grid= 3;
        width_grid=4;
        goalx=3;
        goaly=2;
        init_x_pos=0;
        init_y_pos=0;
    }

    if(environment==2)
    {
        height_grid= 4;
        width_grid=12;
        goalx=11;
        goaly=0;
        init_x_pos=0;
        init_y_pos=0;
    }

    for(i=0; i < width_grid; i++)
    {
        for(j=0; j< height_grid; j++)
        {
            if(environment==1)
            {
                reward[i][j]=-0.04;
                blocked[i][j]=0;
            }

            if(environment==2)
            {
                reward[i][j]=-1;
                blocked[i][j]=0;
            }

            for(k=0; k<4; k++)
            {
                Qvalues[i][j][k]=0;
            }
        }
    }

    if(environment==1)
    {
        reward[goalx][goaly]=1;
        reward[goalx][(goaly-1)]=-1;
        blocked[1][1]=1; // Pared
    }

    if(environment==2)
    {
        reward[goalx][goaly]=1;
        for(int h=1; h<goalx;h++)
        {
            reward[h][0]=-100;
        }
    }
}

int action_selection()
{
    if(action_sel==1)
    {
        float max_q = -99999.0;
        int best_action = 0;
        for(int act=0; act<4; act++){
            if(Qvalues[x_pos][y_pos][act] > max_q){
                max_q = Qvalues[x_pos][y_pos][act];
                best_action = act;
            }
        }
        return best_action;
    }

    if(action_sel==2)// Epsilon-greedy
    {
        float random_val = (float)rand() / RAND_MAX;

        if(random_val < exp_rate)
        {
            return rand()%4;
        }
        else
        {
            float max_q = -99999.0;
            int best_action = 0;
            for(int act=0; act<4; act++){
                if(Qvalues[x_pos][y_pos][act] > max_q){
                    max_q = Qvalues[x_pos][y_pos][act];
                    best_action = act;
                }
            }
            return best_action;
        }
    }
    return 0;
}

void move(int action)
{
    prev_x_pos=x_pos;
    prev_y_pos=y_pos;

    if(stochastic_actions == 1)
    {
        float random_val = (float)rand() / RAND_MAX;

        if (random_val < 0.1)
        {
            action = (action + 1) % 4;
        }
        else if (random_val < 0.2)
        {
            action = (action + 3) % 4;
        }
    }

    if(action==0) // Arriba
    {
        if((y_pos < height_grid-1) && (blocked[x_pos][y_pos+1]==0)) { y_pos=y_pos+1; }
    }
    else if(action==1)  // Derecha
    {
        if((x_pos < width_grid-1) && (blocked[x_pos+1][y_pos]==0)) { x_pos=x_pos+1; }
    }
    else if(action==2)  // Abajo
    {
        if((y_pos > 0) && (blocked[x_pos][y_pos-1]==0)) { y_pos=y_pos-1; }
    }
    else if(action==3)  // Izquierda
    {
        if((x_pos > 0) && (blocked[x_pos-1][y_pos]==0)) { x_pos=x_pos-1; }
    }
}

void update_q_prev_state()
{
    float max_q_current = -99999.0;
    for(int act=0; act<4; act++){
        if(Qvalues[x_pos][y_pos][act] > max_q_current){
            max_q_current = Qvalues[x_pos][y_pos][act];
        }
    }

    float target;
    if(!( ((x_pos==goalx)&&(y_pos==goaly)) ||((environment==1)&&(x_pos==goalx)&&(y_pos==(goaly-1)))||((environment==2)&&(x_pos>0)&&(x_pos<goalx)&&(y_pos==0))) )
    {
        target = reward[x_pos][y_pos] + disc_factor * max_q_current;
    }
    else
    {
        target = reward[x_pos][y_pos];
    }

    float old_q_value = Qvalues[prev_x_pos][prev_y_pos][action_taken];
    float td_error = target - old_q_value; // Error de diferencia temporal
    Qvalues[prev_x_pos][prev_y_pos][action_taken] = old_q_value + learn_rate * td_error;
}

void update_q_prev_state_sarsa()
{
    float q_next_action = Qvalues[x_pos][y_pos][action_taken2];

    float target;
    if(!( ((x_pos==goalx)&&(y_pos==goaly)) ||((environment==1)&&(x_pos==goalx)&&(y_pos==(goaly-1)))||((environment==2)&&(x_pos>0)&&(x_pos<goalx)&&(y_pos==0))     ) )
    {
        target = reward[x_pos][y_pos] + disc_factor * q_next_action;
    }
    else
    {
        target = reward[x_pos][y_pos];
    }

    float old_q_value = Qvalues[prev_x_pos][prev_y_pos][action_taken];
    float td_error = target - old_q_value;
    Qvalues[prev_x_pos][prev_y_pos][action_taken] = old_q_value + learn_rate * td_error;
}

void Qlearning()
{
    action_taken = action_selection();
    move(action_taken);
    cum_reward=cum_reward+reward[x_pos][y_pos];
    update_q_prev_state();
}

void Sarsa()
{
    move(action_taken);
    cum_reward=cum_reward+reward[x_pos][y_pos];
    action_taken2 = action_selection();
    update_q_prev_state_sarsa();
    action_taken = action_taken2;
}

int main(int argc, char* argv[])
{
    srand(time(NULL));
    reward_output.open("sarsa_env2.csv");
    if (!reward_output.is_open()) {
        cout << "Error: No se pudo abrir el archivo Rewards.csv" << endl;
        return 1;
    }

    reward_output << "Episode,Cumulative_Reward" << endl;

    Initialize_environment();

    for(i=0;i<num_episodes;i++)
    {
        x_pos=init_x_pos;
        y_pos=init_y_pos;
        cum_reward=0;

        if(algorithm==2)
        {
            action_taken = action_selection();
        }

        while(!( ((x_pos==goalx)&&(y_pos==goaly)) ||((environment==1)&&(x_pos==goalx)&&(y_pos==(goaly-1)))||((environment==2)&&(x_pos>0)&&(x_pos<goalx)&&(y_pos==0))     ) )
        {
            if(algorithm==1)
            {
                Qlearning();
            }
            if(algorithm==2)
            {
                Sarsa();
            }
        }

        finalrw[i]=cum_reward;

        cout << "Episodio: " << i << ", Recompensa Acumulada: " << finalrw[i] << endl;

        reward_output << i << "," << finalrw[i] << endl;
    }

    reward_output.close();


    std::cout << "=== Ejercicio 1 ===" << std::endl;

    Kmeans kmeans(3);

    kmeans.addPoint('A', 2, 10);
    kmeans.addPoint('B', 2, 5);
    kmeans.addPoint('C', 8, 4);
    kmeans.addPoint('D', 5, 8);
    kmeans.addPoint('E', 7, 5);
    kmeans.addPoint('F', 6, 4);
    kmeans.addPoint('G', 1, 2);
    kmeans.addPoint('H', 4, 9);

    std::vector<Centroid> initialCentroids1 = {
            Centroid(2, 10, 'A'), // Cluster 0
            Centroid(5, 8, 'D'),  // Cluster 1
            Centroid(1, 2, 'G')   // Cluster 2
    };
    kmeans.setInitialCentroids(initialCentroids1);

    kmeans.run(3);

    std::cout << "=== Ejercicio 2 ===" << std::endl;
    DBScan dbscan(2.0, 2);

    dbscan.addPoint('A', 2, 10);
    dbscan.addPoint('B', 2, 5);
    dbscan.addPoint('C', 8, 4);
    dbscan.addPoint('D', 5, 8);
    dbscan.addPoint('E', 7, 5);
    dbscan.addPoint('F', 6, 4);
    dbscan.addPoint('G', 1, 2);
    dbscan.addPoint('H', 4, 9);

    dbscan.run();

    dbscan.runWithDifferentEps(sqrt(10));
    return 0;
}
