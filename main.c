#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define NO_CITIES 48

int start() {
    return 0;
}

int * getfield() {
    static int neim[NO_CITIES*NO_CITIES];
    char buffer[1000];
    int value;
    int i;
    int row = 0;
    FILE * f = fopen("data/att48.csv", "r");
    if(f == NULL) {
        printf("Error can't open file \n");
    }
    while (fgets(buffer,1000,f) != NULL) {
        char * p = strtok(buffer, ";");
        for(i=0;i<NO_CITIES;i++) {
            //printf("%s\n", p);
            p=strtok(NULL, ";");
            value = (int) strtol(p, (char**) NULL, 10);
            neim[i+row*NO_CITIES] = value;
            printf("%d\t%d\n", neim[i+row*NO_CITIES], value);
        }
        printf("\n");
        row++;
    }
    fclose(f);
    return neim;
}

int main() {
    int i;
    int * node_edge_incidence_mat;
    node_edge_incidence_mat = getfield();
    /*
    for (i=0;i<NO_CITIES*NO_CITIES;i++){
        printf("%d\n",*(node_edge_incidence_mat+i));
    }
     */
}
