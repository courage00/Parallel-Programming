#include "bfs.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <cstddef>
#include <omp.h>
#include <map>
using namespace std;

#include "../common/CycleTimer.h"
#include "../common/graph.h"

#define ROOT_NODE_ID 0
#define NOT_VISITED_MARKER -1

void vertex_set_clear(vertex_set *list)
{
    list->count = 0;
}

void vertex_set_init(vertex_set *list, int count)
{
    list->max_vertices = count;
    list->vertices = (int *)malloc(sizeof(int) * list->max_vertices);
    vertex_set_clear(list);
}

// Take one step of "top-down" BFS.  For each vertex on the frontier,
// follow all outgoing edges, and add all neighboring vertices to the
// new_frontier.
void top_down_step(
    Graph g,
    int fDistances,
    vertex_set *frontier,
    vertex_set *new_frontier,
    int *distances)
{
    int fDistancesPlus = fDistances+1;
    #pragma omp parallel for
    for (int i = 0; i < frontier->count; i++)
    {

        int node = frontier->vertices[i];

        int start_edge = g->outgoing_starts[node];
        int end_edge = (node == g->num_nodes - 1)
                           ? g->num_edges
                           : g->outgoing_starts[node + 1];

        for (int neighbor = start_edge; neighbor < end_edge; neighbor++)
        {
            int outgoing = g->outgoing_edges[neighbor];
            if (distances[outgoing] == -1)
            {
                distances[outgoing] = fDistancesPlus;
// #pragma omp critical
//                 {
//                     int index = new_frontier->count++;
//                     new_frontier->vertices[index] = outgoing;
//                 }
                  int index  = __sync_fetch_and_add(&new_frontier->count, 1);
                  //  int index = new_frontier->count++;
                    new_frontier->vertices[index] = outgoing;

            }
            //printf("count:%d \n",new_frontier->count);
        }
    }
}

// Implements top-down BFS.
//
// Result of execution is that, for each node in the graph, the
// distance to the root is stored in sol.distances.
void bfs_top_down(Graph graph, solution *sol)
{

    vertex_set list1;
    vertex_set list2;
    vertex_set_init(&list1, graph->num_nodes);
    vertex_set_init(&list2, graph->num_nodes);

    vertex_set *frontier = &list1;
    vertex_set *new_frontier = &list2;

    // initialize all nodes to NOT_VISITED
    #pragma omp parallel for
    for (int i = 0; i < graph->num_nodes; i++)
    {
        sol->distances[i] = NOT_VISITED_MARKER;
    }
    // setup frontier with the root node
    frontier->vertices[frontier->count++] = ROOT_NODE_ID;
    sol->distances[ROOT_NODE_ID] = 0;
    int fDistances =0;
    while (frontier->count != 0)
    {

#ifdef VERBOSE
        double start_time = CycleTimer::currentSeconds();
#endif

        vertex_set_clear(new_frontier);
        //printf("frontier:%d \n",*frontier);
        top_down_step(graph, fDistances, frontier, new_frontier, sol->distances);

#ifdef VERBOSE
        double end_time = CycleTimer::currentSeconds();
        printf("frontier=%-10d %.4f sec\n", frontier->count, end_time - start_time);
#endif

        // swap pointers
        // printf("frontier->count:%d \n",frontier->count );
        // printf("new_frontier->count:%d \n",new_frontier->count );
        vertex_set *tmp = frontier;
        frontier = new_frontier;
        new_frontier = tmp;
        fDistances++;
    }
    // for (int i = 0; i < graph->num_nodes; i++)
    //     printf("%dsol->distances:%d \n",i,sol->distances[i]);
}

void bottom_up_step(
    Graph g,
    int fDistances,
    bool *flag,
    int *distances)
{
    int fDistancesPlus = fDistances+1;
    #pragma omp parallel for
    for (int  i = 1; i < g->num_nodes-1 ; i++)
    {        
        if (distances[i] == NOT_VISITED_MARKER)
        {
            int start_edge = g->incoming_starts[i];
            int end_edge = g->incoming_starts[i + 1];
            for (int neighbor = start_edge; neighbor < end_edge; neighbor++)
            {
                int incoming = g->incoming_edges[neighbor];
                //printf("i:%d  incoming:%d distances[incoming]:%d flag:%d  \n",i,incoming,distances[incoming],*flag );
                if (distances[incoming] == fDistances)
                {
                    // #pragma omp critical
                    //{
                        distances[i] = fDistancesPlus;
                        // if(!*flag)
                        *flag = true;                            
                    // }
                    break;
                }
            }
       }
    }
    if (distances[g->num_nodes - 1] == NOT_VISITED_MARKER)
        {
            int start_edge = g->incoming_starts[g->num_nodes - 1];
            for (int neighbor = start_edge; neighbor < g->num_edges; neighbor++)
            {
                int incoming = g->incoming_edges[neighbor];
                //printf("i:%d  incoming:%d distances[incoming]:%d flag:%d  \n",i,incoming,distances[incoming],*flag );
                if (distances[incoming] == fDistances)
                {
                    // #pragma omp critical
                    //{
                    distances[g->num_nodes - 1] = fDistancesPlus;
                    *flag = true;
                    // }
                    break;
                }
            }
        }
}

void bfs_bottom_up(Graph graph, solution *sol)
{
    // initialize all nodes to NOT_VISITED
    #pragma omp parallel for
    for (int i = 0; i < graph->num_nodes; i++)
    {
        sol->distances[i] = NOT_VISITED_MARKER;
    }

    sol->distances[ROOT_NODE_ID] = 0;
    bool boolValue = true;
    bool *flag = &boolValue;

    int fDistances =0;
    while (*flag)
    {
        //printf("fDistances:%d \n",fDistances );
        *flag =false;
        bottom_up_step(graph,fDistances,flag, sol->distances);
        fDistances++;
    }
}
void bottom_up_step_hybrid_ver(
    Graph g,
    int fDistances,
    vertex_set *new_frontier,
    int *distances)
{
    int fDistancesPlus = fDistances+1;
    #pragma omp parallel for
    for (int i = 1; i < g->num_nodes-1; i++)
    {
        if (distances[i] == NOT_VISITED_MARKER)
        {
            int start_edge = g->incoming_starts[i];
            int end_edge = g->incoming_starts[i + 1];
            for (int neighbor = start_edge; neighbor < end_edge; neighbor++)
            {
                int incoming = g->incoming_edges[neighbor];
                if (distances[incoming] == fDistances)
                {

                    distances[i] = fDistancesPlus;
                    int index  = __sync_fetch_and_add(&new_frontier->count, 1);
                    new_frontier->vertices[index] = i;
                    break;
                }
            }
        }
    }
    if (distances[g->num_nodes - 1] == NOT_VISITED_MARKER)
    {
        int start_edge = g->incoming_starts[g->num_nodes - 1];
        for (int neighbor = start_edge; neighbor < g->num_edges; neighbor++)
        {
            int incoming = g->incoming_edges[neighbor];
            //printf("i:%d  incoming:%d distances[incoming]:%d flag:%d  \n",i,incoming,distances[incoming],*flag );
            if (distances[incoming] == fDistances)
            {    
                    distances[g->num_nodes - 1] = fDistancesPlus;
                    int index2  = __sync_fetch_and_add(&new_frontier->count, 1);
                    new_frontier->vertices[index2] = g->num_nodes - 1;
                break;
            }
        }
    }
}

    void bfs_hybrid(Graph graph, solution * sol)
    {
        // For PP students:
        //
        // You will need to implement the "hybrid" BFS here as
        // described in the handout.
        vertex_set list1;
        vertex_set list2;
        vertex_set_init(&list1, graph->num_nodes);
        vertex_set_init(&list2, graph->num_nodes);
        vertex_set *frontier = &list1;
        vertex_set *new_frontier = &list2;
        int gEdge = graph->num_edges;
        int gVertexBT = graph->num_nodes;
        // int gVertexBT = graph->num_nodes / 60;

        int flag = 0;

// initialize all nodes to NOT_VISITED
#pragma omp parallel for
        for (int i = 0; i < graph->num_nodes; i++)
        {
            sol->distances[i] = NOT_VISITED_MARKER;
        }
        //printf("start!!!!!!!!!!!!!!!!!!!!!!!!!\n");
        // setup frontier with the root node
        //printf("frontier->count++:%d \n",frontier->count++ ); //0
        frontier->vertices[frontier->count++] = ROOT_NODE_ID;
        sol->distances[ROOT_NODE_ID] = 0;

        int fDistances = 0;
        while (frontier->count != 0)
        {
            int numFEdge = 0;
            vertex_set_clear(new_frontier);
            if (flag == 0)
            {
                gVertexBT -= frontier->count;
                #pragma omp parallel for reduction(+:numFEdge)
                for (int i = 0; i < frontier->count; i++)
                {
                    int node = frontier->vertices[i];
                    numFEdge += outgoing_size(graph, node);
                }
                gEdge -= numFEdge;
                if (numFEdge > (gEdge / 8))
                {
                    flag = 1;
                    //printf("numFEdge:%d  gEdge:%d  gEdge / 8:%d \n", numFEdge, gEdge, gEdge / 8);
                }
            }
            if (flag == 1)
            {
                if (frontier->count < gVertexBT/30)
                {
                    //printf("gVertexBT:%d  frontier->count:%d \n", gVertexBT/35, frontier->count);
                    //printf("gVertexBT:%d  frontier->count:%d \n", gVertexBT, frontier->count);
                    flag=2;
                }
            }
            if (flag == 0 || flag == 2)
            {
                //printf("flag:%d  frontier->count:%d \n", flag, frontier->count);
                top_down_step(graph, fDistances, frontier, new_frontier, sol->distances);
            }
            else
            {
                //printf("flag:%d  frontier->count:%d \n", flag, frontier->count);
                bottom_up_step_hybrid_ver(graph, fDistances,new_frontier, sol->distances);
            }
            //printf("frontier:%d \n",*frontier);
            // printf("frontier->count:%d \n",frontier->count );
            //printf("mapSet.size():%d \n",mapSet.size());

            // swap pointers
            fDistances++;
            vertex_set *tmp = frontier;
            frontier = new_frontier;
            new_frontier = tmp;
        }
    }
