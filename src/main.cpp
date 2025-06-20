#include <iostream>
#include <vector>
#include <fstream>
#include <stdlib.h>
#include <math.h>
#include <cmath>
#include <cstdlib>
#include <time.h>
#include <string.h>
#include <algorithm>
#include <random>

using namespace std;

int height_grid, width_grid, action_taken, action_taken2,current_episode;
int maxA[100][100], blocked[100][100];
float maxQ[100][100], cum_reward,Qvalues[100][100][4], reward[100][100],finalrw[50000];
int init_x_pos, init_y_pos, goalx, goaly, x_pos,y_pos, prev_x_pos, prev_y_pos, blockedx, blockedy,i,j,k;


// Random number generator for stochastic actions
random_device rd;
mt19937 gen(rd());
uniform_real_distribution<> dis(0.0, 1.0);

//////////////
//Setting value for learning parameters
int action_sel=2; // 1 is greedy, 2 is e-greedy
int environment= 2; // 1 is small grid, 2 is Cliff walking
int algorithm = 1; //1 is Q-learning, 2 is Sarsa
int stochastic_actions=0; // 0 is deterministic actions, 1 for stochastic actions
int num_episodes=3000; //total learning episodes
float learn_rate=0.1; // how much the agent weights each new sample
float disc_factor=0.99; // how much the agent weights future rewards
float exp_rate=0.05; // how much the agent explores
///////////////


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
                reward[i][j]=-0.04; //-1 if environment 2
                blocked[i][j]=0;

            }


            if(environment==2)
            {
                reward[i][j]=-1;
                blocked[i][j]=0;
            }


            for(k=0; k<4; k++)
            {
                Qvalues[i][j][k]=rand()%10;
                cout << "Initial Q value of cell [" <<i << ", " <<j << "] action " << k << " = " << Qvalues[i][j][k] << "\n";
            }

        }

    }

    if(environment==1)
    {
        reward[goalx][goaly]=100;
        reward[goalx][(goaly-1)]=-100;
        blocked[1][1]=1;
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

int find_max_action(int x, int y)
{
    int max_action = 0;
    float max_value = Qvalues[x][y][0];

    for(int a = 1; a < 4; a++)
    {
        if(Qvalues[x][y][a] > max_value)
        {
            max_value = Qvalues[x][y][a];
            max_action = a;
        }
    }
    return max_action;
}

// Get maximum Q-value for current state
float get_max_q(int x, int y)
{
    float max_value = Qvalues[x][y][0];
    for(int a = 1; a < 4; a++)
    {
        if(Qvalues[x][y][a] > max_value)
        {
            max_value = Qvalues[x][y][a];
        }
    }
    return max_value;
}

int action_selection()
{ // Based on the action selection method chosen, it selects an action to execute next


    if(action_sel==2)//epsilon-greedy, selects the action with the largest Q value with prob (1-exp_rate) and a random action with prob (exp_rate)
    {
        if(dis(gen) < exp_rate) // explore with likely epsilon = (0.05)
        {
            return rand()%4; //Currently returing a random action, need to code the e-greedy strategy
        }else{ // explore - select the better action with likely (1-epsilon = 0.95)
            return find_max_action(x_pos, y_pos);
        }

    }
}

int apply_stochastic_action(int intended_action)
{
    if(!stochastic_actions) return intended_action;

    float prob = dis(gen);

    if(prob < 0.8)        // 80% - desired address
        return intended_action;
    else if(prob < 0.9)   // 10% - right of the desired direction
        return (intended_action + 1) % 4;
    else                  // 10% - left of the desired direction
        return (intended_action + 3) % 4; // +3 ≡ -1 (mod 4)
}

void move(int action)
{
    prev_x_pos=x_pos; //Backup of the current position, which will become past position after this method
    prev_y_pos=y_pos;

    // Apply stochastic action model
    int actual_action = apply_stochastic_action(action);

    //Stochastic transition model (not known by the agent)
    //Assuming a .8 prob that the action will perform as intended, 0.1 prob. of moving instead to the right, 0.1 prob of moving instead to the left

    if(actual_action==0) // Up
    {

        if((y_pos<(height_grid-1))&&(blocked[x_pos][y_pos+1]==0)) //If there is no wall or obstacle Up from the agent
        {
            y_pos=y_pos+1;  //move up
        }

    }


    if(actual_action==1)  //Right
    {

        if((x_pos<(width_grid-1))&&(blocked[x_pos+1][y_pos]==0)) //If there is no wall or obstacle Right from the agent
        {
            x_pos=x_pos+1; //Move right
        }

    }

    if(actual_action==2)  //Down
    {

        if((y_pos>0)&&(blocked[x_pos][y_pos-1]==0)) //If there is no wall or obstacle Down from the agent
        {
            y_pos=y_pos-1; // Move Down
        }

    }

    if(actual_action==3)  //Left
    {

        if((x_pos>0)&&(blocked[x_pos-1][y_pos]==0)) //If there is no wall or obstacle Left from the agent
        {
            x_pos=x_pos-1;//Move Left
        }

    }
}

void update_q_prev_state() //Updates the Q value of the previous state
{
    //Update the Q value of the previous state and action if the agent has not reached a terminal state
    if(!( ((x_pos==goalx)&&(y_pos==goaly)) ||((environment==1)&&(x_pos==goalx)&&(y_pos==(goaly-1)))||((environment==2)&&(x_pos>0)&&(x_pos<goalx)&&(y_pos==0))) )
    {
        // Q-learning: Q(s,a) = Q(s,a) + α[r + γ max_a'Q(s',a') - Q(s,a)]
        float max_q_next = get_max_q(x_pos, y_pos);
        float td_error = reward[x_pos][y_pos] + disc_factor * max_q_next - Qvalues[prev_x_pos][prev_y_pos][action_taken];
        Qvalues[prev_x_pos][prev_y_pos][action_taken] += learn_rate * td_error;
    }
    else//Update the Q value of the previous state and action if the agent has reached a terminal state
    {
        // Terminal state: Q(s,a) = Q(s,a) + α[r - Q(s,a)]
        float td_error = reward[x_pos][y_pos] - Qvalues[prev_x_pos][prev_y_pos][action_taken];
        Qvalues[prev_x_pos][prev_y_pos][action_taken] += learn_rate * td_error;
    }
}

void update_q_prev_state_sarsa()
{
    //Update the Q value of the previous state and action if the agent has not reached a terminal state
    if(!( ((x_pos==goalx)&&(y_pos==goaly)) ||((environment==1)&&(x_pos==goalx)&&(y_pos==(goaly-1)))||((environment==2)&&(x_pos>0)&&(x_pos<goalx)&&(y_pos==0)) ) )
    {
        // SARSA: Q(s,a) = Q(s,a) + α[r + γQ(s',a') - Q(s,a)]
        float td_error = reward[x_pos][y_pos] + disc_factor * Qvalues[x_pos][y_pos][action_taken2] - Qvalues[prev_x_pos][prev_y_pos][action_taken];
        Qvalues[prev_x_pos][prev_y_pos][action_taken] += learn_rate * td_error;
    }
    else//Update the Q value of the previous state and action if the agent has reached a terminal state
    {
        // Terminal state: Q(s,a) = Q(s,a) + α[r - Q(s,a)]
        float td_error = reward[x_pos][y_pos] - Qvalues[prev_x_pos][prev_y_pos][action_taken];
        Qvalues[prev_x_pos][prev_y_pos][action_taken] += learn_rate * td_error;
    }
}

void Qlearning()
{
    action_taken = action_selection();
    move(action_taken);
    cum_reward = cum_reward + reward[x_pos][y_pos];
    update_q_prev_state();
}

void Sarsa()
{
    action_taken2 = action_selection();
    move(action_taken2);
    cum_reward = cum_reward + reward[x_pos][y_pos];
    update_q_prev_state_sarsa();
    action_taken = action_taken2;
}

void Multi_print_grid()
{
    int x, y;

    for(y = (height_grid-1); y >=0 ; --y)
    {
        for (x = 0; x < width_grid; ++x)
        {

            if(blocked[x][y]==1) {
                cout << " \033[42m# \033[0m";

            }else{
                if ((x_pos==x)&&(y_pos==y)){
                    cout << " \033[44m1 \033[0m";

                }else{
                    cout << " \033[31m0 \033[0m";


                }
            }
        }
        printf("\n");
    }
}

int main(int argc, char* argv[])
{
    srand(time(NULL));
    ofstream reward_output;
    reward_output.open("rewards_data.csv");
    if (!reward_output.is_open()) {
        cout << "ERROR: No se pudo abrir el archivo CSV!" << endl;
        return 1;
    }

    reward_output << "Episode,Cumulative_Reward\n";

    Initialize_environment();

    cout << "Running " << ((algorithm == 1) ? "Q-learning" : "SARSA")
         << " on Environment " << environment
         << " with " << ((stochastic_actions) ? "stochastic" : "deterministic") << " actions\n";

    for(i=0;i<num_episodes;i++)
    {
        cout << "Episode " << i;
        current_episode=i;
        x_pos=init_x_pos;
        y_pos=init_y_pos;
        cum_reward=0;

        int steps = 0;

        if(algorithm==2)
        {
            action_taken = action_selection();
        }

        while(!( ((x_pos==goalx)&&(y_pos==goaly)) ||((environment==1)&&(x_pos==goalx)&&(y_pos==(goaly-1)))||((environment==2)&&(x_pos>0)&&(x_pos<goalx)&&(y_pos==0)) ))
        {
            if(algorithm==1)
            {
                Qlearning();
            }
            if(algorithm==2)
            {
                Sarsa();
            }
            steps++;
        }

        finalrw[i]=cum_reward;

        reward_output << i << "," << finalrw[i] << "\n";
        reward_output.flush();

        cout << " - Reward: " << finalrw[i] << " - Steps: " << steps << "\n";

        if(i % 100 == 0) {
            cout << "Progreso: " << (i*100)/num_episodes << "% completado\n";
        }
    }

    reward_output.close();
    cout << "\nArchivo CSV creado: rewards_data.csv\n";

    return 0;
}

/*int main() {
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
}*/
