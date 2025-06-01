#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <math.h>
#include <string.h>

/* #DEFINE'S -----------------------------------------------------------------*/
#define SDELIM "==STAGE %d============================\n"   // stage delimiter
#define MDELIM "-------------------------------------\n"    // delimiter of -'s
#define THEEND "==THE END============================\n"    // end message

#define CH_NULL '\0'    // null byte which ends each string
#define CRTRNC '\r'     // carriage return character
#define ON_CELL '*'     // character representing on cells
#define OFF_CELL '.'    // character representing off cells
#define NBRHDS 8        // number of possible neighborhoods
#define ALL_BIN_STR " 000 001 010 011 100 101 110 111\n" 
                        // string of all binary numbers represented by 3 bits
#define NBRHD_SIZE 3    // size of each neighborhood
#define INITIAL 2       // an initial size used for dynamic memory allocation 
/* TYPE DEFINITIONS ----------------------------------------------------------*/
typedef char cells_t;           // base type to store states of cells
typedef int rule_t[NBRHDS];     // an elementary CA update rule function

typedef struct {                // an elementary CA is defined by
    int             size;       // ... a number of cells,
    int             rule;       // ... the current rule number 
    int             time;       // ... the current time step,
    rule_t          bin_rule;   // ... an update rule function
    cells_t**       all_evols;  // ... an array that stores all CA evolutions   
} CA_t;

/* USEFUL FUNCTIONS ----------------------------------------------------------*/
int mygetchar(void);                // getchar() that skips carriage returns
void read_input(CA_t* ca, int* s1_evol_steps, int cell_n_start_s1[], 
                int cell_n_start_s2[]);
void stage_0(CA_t* ca);
void stage_1(CA_t* ca, int* s1_evol_steps, char all_nbrhds[][NBRHD_SIZE+1], 
             int cell_n_start_s1[]);
void stage_2(CA_t* ca, char all_nbrhds[][NBRHD_SIZE+1], int cell_n_start_s2[], 
             int* s1_evol_steps);
void bin_rule_calc(int* bin_rule, int rule);
void cell_pos_and_time_step(char* curr_line, int idx, int* cell_pos, 
                            int* time_step); 
void ca_evolution_update(CA_t* ca, int req_evol_steps, int start_t_step, 
                         char all_nbrhds[][NBRHD_SIZE+1]);
void on_off_states_calc(CA_t* ca, int cell_pos, int start_time);

/* WHERE IT ALL HAPPENS ------------------------------------------------------*/

int main(int argc, char *argv[]) {
    CA_t ca;
    int s1_evol_steps = 0;
    /* initializes 2 integer arrays to store the cell position and time_step 
    for each corresponding stage's on and off state count. */ 
    int cell_n_start_s1[2], cell_n_start_s2[2];
    
    read_input(&ca, &s1_evol_steps, cell_n_start_s1, cell_n_start_s2);
    stage_0(&ca);
    
    /* creates a 2D array to store all possible neighborhoods that are 
    in ascending order when represented in binary form */
    char all_nbrhds[NBRHDS][NBRHD_SIZE+1] = {
        "...",
        "..*",
        ".*.",
        ".**",
        "*..",
        "*.*",
        "**.",
        "***"
    };
    stage_1(&ca, &s1_evol_steps, all_nbrhds, cell_n_start_s1);
    stage_2(&ca, all_nbrhds, cell_n_start_s2, &s1_evol_steps);
    
    /* frees the dynamically allocated memory of each evolution in all_evols */  
    for (int i = 0; i < ca.time; i++) {
        free(ca.all_evols[i]);
    }
    /* frees the dynamically allocated memory of the array all_evols */ 
    free(ca.all_evols);
    
    return EXIT_SUCCESS;        // algorithms are fun!!!
}

/* USEFUL FUNCTIONS ----------------------------------------------------------*/

// An improved version of getchar(); skips carriage return characters.
// NB: Adapted version of the mygetchar() function by Alistair Moffat
int mygetchar() {
    int c;
    while ((c=getchar())==CRTRNC);          // skip carriage return characters
    return c;
}

/******************************************************************************/

/* reads each line of the input file and stores them for future usage using 
corresponding variables and structures */

void read_input(CA_t *ca, int *s1_evol_steps, int cell_n_start_s1[], 
                int cell_n_start_s2[]){
    
    int ch, line_num = 0;
    /* stores data of each line of the input file into variables or 
    data structures */
    
    while(1){
        int curr_size = INITIAL, idx = 0;
        /* dynamically allocate memory to store the current line string */
        char *curr_line = (char*)malloc(curr_size * sizeof(char));
        assert(curr_line);
        
        /* keeps reading characters until a new line or EOF character is read */ 
        while((ch = mygetchar()) && (ch != '\n') && (ch != EOF)){
            /* if all space allocated to curr_line has been occupied, it is 
            reallocated more memory by doubling its size */
            if (idx == curr_size) {
                curr_size *= 2;
                curr_line = (char*)realloc(curr_line, curr_size * sizeof(char));
                assert(curr_line);
            }
            /* stores the current character in the current line string */
            curr_line[idx] = ch;
            idx++;
        }
        
        /* null terminates the current line string */
        curr_line[idx] = CH_NULL;
        line_num++;
        /* stores cellular automaton size */
        if(line_num == 1){
            ca -> size = atoi(curr_line);
            
        /* stores rule number for cell state updates */
        }else if(line_num == 2){
            ca -> rule = atoi(curr_line);
            /* updates the bin_rule array based on the reversed 
            binary representation of the rule */
            bin_rule_calc(ca -> bin_rule, ca -> rule);
            
        /* stores the initial cellular automaton state */
        }else if(line_num == 3){
            /* dynamically allocates memory for all_evols to store one 
            cellular automaton state */
            ca -> all_evols = (cells_t**)malloc(sizeof(cells_t*));
            assert(ca -> all_evols);
            
            /* dynamically allocates memory to the first element of all_evols 
            based on the cellular automaton size */
            ca -> all_evols[0] = (char*)malloc((ca -> size + 1)* sizeof(char));
            assert(ca -> all_evols[0]);
            strcpy(ca -> all_evols[0], curr_line);
            
        /* stores number of evolutions to compute in stage 1 and re-allocates 
        the required memory to all_evols to store upcoming computations */
        }else if(line_num == 4){
            /* stores the number of evolutions required to compute in stage 1 */
            *s1_evol_steps = atoi(curr_line);
            
            /* Calculate the total number of time steps to compute in both 
            stage 1 an stage 2 */
            int total_t_steps = (*s1_evol_steps) + (ca -> size - 2)/2 + 
                                (ca -> size - 1)/2 + 1;

            /* dynamically allocates memory to store all future computations */
            ca -> all_evols = (cells_t **)realloc(ca -> all_evols, total_t_steps
                                                  * sizeof(cells_t *));
            assert(ca -> all_evols);
            
            /* dynamically allocates memory for each evolution to store a number
            of characters given by the cellular automaton's size */
            for (int i = 1; i < total_t_steps; i++) {
                ca -> all_evols[i] = (char *)malloc((ca -> size + 1)* 
                                                    sizeof(char));
                assert(ca -> all_evols[i]);
            }
            
        }else if(line_num == 5){
            /* stores the cell position and starting time step needed to 
            count the number of on and off states in stage 1 */
            cell_pos_and_time_step(curr_line, idx, &cell_n_start_s1[0], 
                                   &cell_n_start_s1[1]);
        }else{
            /* stores the cell position and starting time step needed to 
            count the number of on and off states in stage 2 */
            cell_pos_and_time_step(curr_line, idx, &cell_n_start_s2[0], 
                                   &cell_n_start_s2[1]);
            free(curr_line);
            break;
        }
        free(curr_line);
    }   
}

/******************************************************************************/

void stage_0(CA_t *ca){
    
    printf(SDELIM, 0); 
    printf("SIZE: %d\n", ca -> size);
    printf("RULE: %d\n", ca -> rule);
    printf(MDELIM);
    
    /* prints all possible binary strings of 3 bits in ascending order */
    printf(ALL_BIN_STR);
    
    /* prints the update binary rule, with each digit present under 
    its corresponding binary string of 3 bits */
    for(int i = 0; i < NBRHDS; i++){
        printf("%3d ", ca -> bin_rule[i]);
    }
    printf("\n");
    printf(MDELIM);
    ca -> time = 0;
    
    /* prints the initial cellular automaton state */
    printf("%4d: %s\n", ca -> time, ca -> all_evols[ca -> time]);
}

/******************************************************************************/

void stage_1(CA_t *ca, int *s1_evol_steps, char all_nbrhds[][NBRHD_SIZE+1], 
             int cell_n_start_s1[]){
    
    printf(SDELIM, 1);
    ca -> time = 1;
    
    /* performs computations for the required number of cellular automaton 
    evolutions and stores these evolutions in the array all_evols. These 
    evolutions are then printed along with their time step */
    int start_t_step = ca-> time - 1, req_evol_steps = *s1_evol_steps + 1;
    ca_evolution_update(ca, req_evol_steps, start_t_step, all_nbrhds);
    printf(MDELIM);
    
    /* calculates the number of on and off states of a particular cell, given 
    the cell position and the starting time step */
    int cell_pos = cell_n_start_s1[0], start_time = cell_n_start_s1[1];
    on_off_states_calc(ca, cell_pos, start_time);
}

/******************************************************************************/

void stage_2(CA_t *ca, char all_nbrhds[][NBRHD_SIZE+1], int cell_n_start_s2[], 
             int *s1_evol_steps){
    
    printf(SDELIM, 2);
    ca -> rule = 184;
    
    /* updates the bin_rule array based on the reversed binary representation 
    of the new rule 184 */
    bin_rule_calc(ca -> bin_rule, ca -> rule);
    
    /* the number of evolutions to compute with the rule 184 */
    int steps_r184 = (ca -> size - 2)/2;
    printf("RULE: %d; STEPS: %d.\n", ca -> rule, steps_r184);
    printf(MDELIM);
    
    /* performs computations for the required number of cellular automaton 
    evolutions and stores these evolutions in the array all_evols. These 
    evolutions are then printed along with their time step */
    int start_t_step = ca -> time - 1, req_evol_steps = ca -> time + steps_r184;
    ca_evolution_update(ca, req_evol_steps, start_t_step, all_nbrhds);
    printf(MDELIM);
    
    ca -> rule = 232;
    
    /* updates the bin_rule array based on the reversed binary representation 
    of the new rule 232 */
    bin_rule_calc(ca -> bin_rule, ca -> rule);
    
    /* the number of evolutions to compute with the rule 232 */
    int steps_r232 = (ca -> size - 1)/2;
    printf("RULE: %d; STEPS: %d.\n", ca -> rule, steps_r232);
    printf(MDELIM);
    
    /* performs computations for the required number of cellular automaton 
    evolutions and stores these evolutions in the array all_evols. These 
    evolutions are then printed along with their time step */
    start_t_step = ca -> time - 1;
    req_evol_steps = ca -> time + steps_r232;
    ca_evolution_update(ca, req_evol_steps, start_t_step, all_nbrhds);
    printf(MDELIM);
    
    /* calculates the number of on and off states of a particular cell, given 
    the cell position and the starting time step */
    int cell_pos = cell_n_start_s2[0], start_time = cell_n_start_s2[1];
    on_off_states_calc(ca, cell_pos, start_time);
    printf(MDELIM);
    
    /* prints the cellular automaton evolution at the beginning of stage 2, 
    for which the density classification problem is being solved */
    printf("%4d: %s\n", *s1_evol_steps, ca -> all_evols[*s1_evol_steps]);
    int num_on = 0, num_off = 0;
    
    /* counts the number of on and off states in the first 2 cells of the 
    final cellular automaton evolution */
    for(int i = 0; i < 2; i++){
        if(ca -> all_evols[ca -> time - 1][i] == ON_CELL){
            num_on++;
        }else{
            num_off++;
        }
    }
    
    /* prints whether there were more, less or equal ocurrences of ones 
    compared to zeros in the cellular automaton evolution at the the beginning 
    of stage 2. This is determined by testing whether there were more, less or 
    equal ocurrences of ones compared to zeros in the first 2 cells of the final 
    cellular automaton evolution */
    if(num_on > num_off){
        printf("AT T=%d: #ON/#CELLS > 1/2\n", *s1_evol_steps);
    }else if(num_on < num_off){
        printf("AT T=%d: #ON/#CELLS < 1/2\n", *s1_evol_steps);
    }else{
        printf("AT T=%d: #ON/#CELLS = 1/2\n", *s1_evol_steps);
    }
    printf(THEEND);
}

/******************************************************************************/

/* updates the bin_rule array with the individual integers present in the 
reversed binary representation of the given rule */

void bin_rule_calc(int* bin_rule, int rule){
    
    int num = rule, x = NBRHDS - 1;
    /* runs until all 8 elements of bin_rule have been modified */
    while(x >= 0){
        /* checks if the 'x'-th bit of num is 1, in which case the value of 
        the 'x'-th bit is subtracted from num */
        if (num >= 1 << x){
            bin_rule[x] = 1;
            num -= 1 << x;
        }else{
            bin_rule[x] = 0;
        }
        x--;
    }
}

/******************************************************************************/
/* reads the characters present in the current line and splits them into two 
integers cell_pos and time_step which are stored in an array for future use */

void cell_pos_and_time_step(char* curr_line, int idx, int* cell_pos, 
                            int* time_step) {
    
    char cell_pos_str[idx], time_step_str[idx];
    
    for (int j = 0; j < idx; j++){
        /* if the string reprsenting cell_pos has been read completely */
        if (curr_line[j] == ','){
            cell_pos_str[j] = CH_NULL;
            
            /* loops through the characters after the commah to
            read the time_step string until the end of curr_line*/
            for(int k = j+1; k < idx; k++){
                time_step_str[k-(j+1)] = curr_line[k];
            }
            time_step_str[idx-j-1] = CH_NULL;
            break;
            
        /* appends the character to the string representing cell_pos */
        }else{
            cell_pos_str[j] = curr_line[j];
        }
    }
    
    /* converts the acquired strings to integers */
    *cell_pos = atoi(cell_pos_str);
    *time_step = atoi(time_step_str);
}

/******************************************************************************/
/* Computes the required number of cellular automaton 
evolutions by updating cells based on their neighborhood and the 
binary update rule array.These evolutions  are stored in the array all_evols 
and are then printed along with their time step */

void ca_evolution_update(CA_t* ca, int req_evol_steps, int start_t_step,
                         char all_nbrhds[][NBRHD_SIZE+1]){
    
    /* updates each evolution into all_evols until the required number of steps 
    to compute has been reached */
    for(; ca -> time < req_evol_steps; ca -> time++){
        /* updates the neighborhood given by the prior cell, current cell 
        and next cell into curr_nbrhd (accounting for wrapping if needed) */
        for(int i = 0; i < ca -> size; i++){
            char curr_nbrhd[NBRHD_SIZE+1];
            /* if the current cell is at the start of the cellular automaton */
            if (i == 0){
                curr_nbrhd[0] = ca -> all_evols[ca -> time-1][ca -> size-1];
                curr_nbrhd[1] = ca -> all_evols[ca -> time-1][i];
                curr_nbrhd[2] = ca -> all_evols[ca -> time-1][i+1];
                curr_nbrhd[3] = CH_NULL;
            /* if the current cell is at the end of the cellular automaton */
            }else if (i == ca -> size - 1){
                curr_nbrhd[0] = ca -> all_evols[ca -> time-1][i-1];
                curr_nbrhd[1] = ca -> all_evols[ca -> time-1][i];
                curr_nbrhd[2] = ca -> all_evols[ca -> time-1][0];
                curr_nbrhd[3] = CH_NULL;
            }else{
                curr_nbrhd[0] = ca -> all_evols[ca -> time-1][i-1];
                curr_nbrhd[1] = ca -> all_evols[ca -> time-1][i];
                curr_nbrhd[2] = ca -> all_evols[ca -> time-1][i+1];
                curr_nbrhd[3] = CH_NULL;
            }
            
            /* iterates through the array all_nbrhds until a neighbourhood
            that is the same as curr_nbrhd is reached */
            for (int j = 0; j < NBRHDS; j++){
                if (strcmp(curr_nbrhd, all_nbrhds[j]) == 0){
                    /* if the binary update rule corrsponding to the current 
                    neighborhood is 1, the middle cell of the neighbourhood 
                    will be on, otherwise it will be off. */
                    if (ca -> bin_rule[j]){
                        ca -> all_evols[ca -> time][i] = ON_CELL;
                    }else {
                        ca -> all_evols[ca -> time][i] = OFF_CELL;
                    }
                    break;
                }
            }
        }
    }
    
    /* prints the evolutions calculated by calling this function, 
    along with the corresponding time steps */
    for(int t = start_t_step; t < ca -> time; t++){
        printf("%4d: %s\n", t, ca -> all_evols[t]);
    }
}

/******************************************************************************/

/* counts the number of on and off states of a particular cell within the 
cellular automaton based on a given starting time step. These tallies are then 
printed along with the corresponding cell position and starting time step */
void on_off_states_calc(CA_t* ca, int cell_pos, int start_time){
    
    int num_on = 0, num_off = 0;
    /* tallies number of on and off states */
    for(int i = start_time; i < ca -> time; i++){
        if (ca -> all_evols[i][cell_pos] == ON_CELL){
            num_on++;
        }else{
            num_off++;
        }
    }
    printf("#ON=%d #OFF=%d CELL#%d START@%d\n", 
           num_on, num_off, cell_pos, start_time);
}
/* Algorithms are fun! */
