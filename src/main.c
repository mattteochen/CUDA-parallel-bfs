%%cu
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define USE_HOST 0
#define DEBUG 0
#define MAX_FILE_COLUMN_LEN 100
#define ERROR 1
#define REAL_NODES_NUM(n) n-1
#define REAL_NEIGHBOURS_NUM(n) REAL_NODES_NUM(n)

void parse(uint32_t* from, uint32_t* to, const char* line) {
  char from_str[MAX_FILE_COLUMN_LEN] = {0};
  char to_str[MAX_FILE_COLUMN_LEN] = {0};
  uint32_t from_str_index = 0;
  uint32_t to_str_index = 0;
  const uint32_t len = strlen(line);
  uint32_t i = 0;
  while (i < len && line[i] != ' ' && line[i] != '\0' && line[i] != '\n') {
    from_str[from_str_index++] = line[i++];
  }
  from_str[from_str_index] = '\0';
  i++;
  while (i < len && line[i] != ' ' && line[i] != '\0' && line[i] != '\n') {
    to_str[to_str_index++] = line[i++];
  }
  *from = atoi(from_str);
  *to = atoi(to_str);
}

#if (USE_HOST == 1)
void print_result(uint32_t* next_level_nodes, const uint32_t num_next_level_nodes) {
  if (!next_level_nodes) {
    printf("null node\n");
  }
  for (uint32_t i=0; i<num_next_level_nodes; i++) {
    printf("%u " ,next_level_nodes[i]);
  }
  printf("\n");
}

void host_queuing_kernel(uint32_t *node_ptr, uint32_t *node_neighbours,
                         uint32_t *node_visited, uint32_t *curr_level_nodes,
                         uint32_t *next_level_nodes,
                         uint32_t* prefix_sum,
                         const uint32_t num_curr_level_nodes,
                         uint32_t *num_next_level_nodes) {
  prefix_sum = 0;
  //iterate over neighbours
  for (uint32_t i=0; i<num_curr_level_nodes; ++i) {
    uint32_t node = curr_level_nodes[i];
    for (uint32_t j=node_ptr[node]; j<node_ptr[node+1]; ++j) {
      uint32_t n = node_neighbours[j];
      if (!node_visited[n]) {
        node_visited[n] = 1;
        next_level_nodes[*num_next_level_nodes] = n;
        (*num_next_level_nodes)++;
      }
    }
  }
#if (DEBUG == 1)
  print_result(next_level_nodes, *num_next_level_nodes);
#endif
}
#else
__device__ void print_result(uint32_t* next_level_nodes, const uint32_t num_next_level_nodes) {
  if (!next_level_nodes) {
    printf("null node\n");
  }
  for (uint32_t i=0; i<num_next_level_nodes; i++) {
    printf("%u " ,next_level_nodes[i]);
  }
  printf("\n");
}

__global__ void gpu_global_queuing_kernel(uint32_t *node_ptr, uint32_t *node_neighbours,
                         uint32_t *node_visited, uint32_t *curr_level_nodes,
                         uint32_t *next_level_nodes,
                         uint32_t* prefix_sum,
                         const uint32_t num_curr_level_nodes,
                         uint32_t *num_next_level_nodes) {
  //printf("block idX: %u block dimX: %u thread idX: %u\n", blockIdx.x, blockDim.x, threadIdx.x);
  //printf("block idY: %u block dimY: %u thread idY: %u\n", blockIdx.y, blockDim.y, threadIdx.y);
  prefix_sum = 0;
  //iterate over neighbours
  for (uint32_t i=0; i<num_curr_level_nodes; ++i) {
    uint32_t node = curr_level_nodes[i];
    for (uint32_t j=node_ptr[node]; j<node_ptr[node+1]; ++j) {
      uint32_t n = node_neighbours[j];
      if (!node_visited[n]) {
        node_visited[n] = 1;
        next_level_nodes[*num_next_level_nodes] = n;
        (*num_next_level_nodes)++;
      }
    }
  }
#if (DEBUG == 1)
  print_result(next_level_nodes, *num_next_level_nodes);
#endif
}
#endif

int main(int argc, char* argv[]) {
#if (USE_HOST == 0)
  // retrieve some info about the CUDA device
  int32_t num_devices;
  cudaGetDeviceCount(&num_devices);
  for (int i = 0; i < num_devices; i++) {
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, i);
    printf("Device Number: %d\n", i);
    printf("  Device name: %s\n", prop.name);
    printf("  max Blocks Per MultiProcessor: %d\n", prop.maxBlocksPerMultiProcessor);
    printf("  max Threads Per MultiProcessor: %d\n", prop.maxThreadsPerMultiProcessor);
    printf("  max Threads Per Block: %d\n", prop.maxThreadsPerBlock);
    printf("  num SM: %d\n", prop.multiProcessorCount);
    printf("  num bytes sharedMem Per Block: %d\n", prop.sharedMemPerBlock);
    printf("  num bytes sharedMem Per Multiprocessor: %d\n", prop.sharedMemPerMultiprocessor);
    printf("  Memory Clock Rate (KHz): %d\n",
         prop.memoryClockRate);
    printf("  Memory Bus Width (bits): %d\n",
         prop.memoryBusWidth);
    printf("  Peak Memory Bandwidth (GB/s): %f\n\n",
         2.0*prop.memoryClockRate*(prop.memoryBusWidth/8)/1.0e6);
  }
#endif
  uint32_t line_counter = 0;
  uint32_t num_nodes = 0;
  uint32_t total_neighbours = 0;
  uint32_t* node_ptr = 0;
  uint32_t* node_neighbours = 0;
  uint32_t* node_visited = 0;
  uint32_t* curr_level_nodes = 0;
  uint32_t* next_level_nodes = 0;
  uint32_t num_curr_level_nodes = 0;
#if (USE_HOST == 1)
  uint32_t num_next_level_nodes = 0;
#else
  uint32_t* num_next_level_nodes = 0;
  cudaMallocManaged((void **) &num_next_level_nodes, sizeof(uint32_t));                                
#endif
  FILE *file;
  uint32_t node_ptr_index = 0;
  uint32_t node_neighbours_index = 0;
  uint32_t last_node_val = 0;
  const char* file_name = "./3.txt";

  file = fopen(file_name, "r");
  if (file == NULL) {
    printf("Error opening the file.\n");
    return 1;
  }
  char line[MAX_FILE_COLUMN_LEN] = {0};
  while (fgets(line, sizeof(line), file) != NULL) {
    if (!line_counter) {
      (line_counter)++;
      num_nodes = atoi(line);
    }
    //counting the first line, yes one too much but then compensated with the virtual node added
    (total_neighbours)++;
  }
  fclose(file);
  //add virtual node to avoid index overflow in the computation later
  (num_nodes)++;
  printf("total nodes: %d - total neighbours: %d\n", REAL_NODES_NUM(num_nodes), REAL_NEIGHBOURS_NUM(total_neighbours));

  //allocate buffers
#if (USE_HOST == 1)
  node_ptr = (uint32_t*)malloc(sizeof(uint32_t) * num_nodes);
#else
  cudaMallocManaged((void **) &node_ptr, sizeof(uint32_t) * num_nodes);
#endif
  if (!node_ptr) {
    printf("failed malloc - node_ptr\n");
  }
#if (USE_HOST == 1)
  curr_level_nodes = (uint32_t*)malloc(sizeof(uint32_t) * num_nodes);
#else
  cudaMallocManaged((void **) &curr_level_nodes, sizeof(uint32_t) * num_nodes);
#endif
  if (!curr_level_nodes) {
    printf("failed malloc - curr_level_nodes\n");
  }
#if (USE_HOST == 1)
  node_neighbours = (uint32_t*)malloc(sizeof(uint32_t) * total_neighbours);
#else
  cudaMallocManaged((void **) &node_neighbours, sizeof(uint32_t) * total_neighbours);
#endif
  if (!node_neighbours) {
    printf("failed malloc - node_neighbours\n");
  }
#if (USE_HOST == 1)
  next_level_nodes = (uint32_t*)malloc(sizeof(uint32_t) * total_neighbours);
#else
  cudaMallocManaged((void **) &next_level_nodes, sizeof(uint32_t) * total_neighbours);
#endif
  if (!next_level_nodes) {
    printf("failed malloc - next_level_nodes\n");
  }

  file = fopen(file_name, "r");
  line_counter = 0;
  if (file == NULL) {
    printf("Can not open the file");
    return 1;
  }
  while (fgets(line, sizeof(line), file) != NULL) {
    if (!line_counter) {
      (line_counter)++;
      continue;
    }
    uint32_t from;
    uint32_t to;
    parse(&from, &to, line);
    if (!node_ptr_index || last_node_val != from) {
      last_node_val = from;
      (node_ptr)[node_ptr_index++] = node_neighbours_index;
    }
    (node_neighbours)[node_neighbours_index++] = to;
  }
  fclose(file);

  //populate the virtual node
  (node_ptr)[node_ptr_index] = node_neighbours_index;
  (node_neighbours)[node_neighbours_index] = UINT32_MAX;

  uint32_t max_node_val = num_nodes - 1;
  for (uint32_t i=0; i<node_neighbours_index; ++i) {
    if ((node_neighbours)[i] > max_node_val) {
      max_node_val = (node_neighbours)[i];
    }
  }
  max_node_val += 10; //overflow barrier
  printf("max node values: %u\n", max_node_val);

#if (USE_HOST == 1)
  node_visited = (uint32_t*)malloc(sizeof(uint32_t) * max_node_val);
#else
  cudaMallocManaged((void **) &node_visited, sizeof(uint32_t) * max_node_val);
#endif
  if (!node_visited) {
    printf("failed malloc - node_visited)\n");
  }
  memset(node_visited, 0, sizeof(uint32_t) * max_node_val);

#if (DEBUG == 1)
  printf("node ptr:\n");
  for (uint32_t i=0; i<num_nodes; i++) {
    printf("%u ", (node_ptr)[i]);
  }
  printf("\n");
  printf("node neighbours:\n");
  for (uint32_t i=0; i<total_neighbours; i++) {
    printf("%u ", (node_neighbours)[i]);
  }
  printf("\n");
#endif

  //populate_curr_level_nodes
  for (uint32_t i=0; i<REAL_NODES_NUM(num_nodes); ++i) {
    curr_level_nodes[i] = i;
  }
  num_curr_level_nodes = REAL_NODES_NUM(num_nodes);

#if (USE_HOST == 1)
  host_queuing_kernel(node_ptr, node_neighbours, node_visited, curr_level_nodes, next_level_nodes, 0, num_curr_level_nodes, &num_next_level_nodes);
#else
  float  naive_gpu_elapsed_time_ms;
  // some events to count the execution time
  //clock_t st, end;
  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  
  uint32_t grid_size = (num_nodes + 1024 - 1) / 1024;
  printf("num grid %u\n", grid_size);
  dim3 dim_block(1, 1024/16);
  cudaEventRecord(start, 0);
  gpu_global_queuing_kernel<<<grid_size, dim_block>>>(node_ptr, node_neighbours, node_visited, curr_level_nodes, next_level_nodes, 0, num_curr_level_nodes, num_next_level_nodes);
  cudaThreadSynchronize();
  
  cudaEventRecord(stop, 0);
  cudaEventSynchronize(stop);
  cudaEventElapsedTime(&naive_gpu_elapsed_time_ms, start, stop);
  printf("Time elapsed on global GPU queueing: %f ms\n", naive_gpu_elapsed_time_ms);

#endif

#if (USE_HOST == 1)
  free(node_ptr);
  free(node_neighbours);
  free(curr_level_nodes);
  free(next_level_nodes);
  free(node_visited);
#else
  cudaFree(node_ptr);
  cudaFree(node_neighbours);
  cudaFree(curr_level_nodes);
  cudaFree(next_level_nodes);
  cudaFree(node_visited);
#endif

  return 0;
}
