#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#define ADJ_MATRIX_ROW_SIZE 1000
#define DEBUG 0
#define DEBUG_HOST_KER 0
#define DEBUG_KER 0
#define DEBUG_KER_GRID 0
#define PRODUCE_OUT_FILE 1
#define ERROR 1
#define MAX_FILE_COLUMN_LEN 100
#define REAL_NEIGHBOURS_NUM(n) REAL_NODES_NUM(n)
#define REAL_NODES_NUM(n) n-1
#define STARTING_LEVEL_NUM 1
#define STARTING_NODE 0
#define USE_HOST 0
#define USE_PREFIX_SUM 0

#define DEVICE_SHARED_MEM_PER_BLOCK 65536/16

#if (USE_HOST == 1)
#warning "Using USE_HOST == 1 my NOT work on collab if binary is not called manually"
#endif

typedef struct Vector {
  int32_t* buff;
  int32_t size;
} Vector;

void print_adj_matrix(Vector* adj_matrix, const int32_t size) {
  for (int32_t i=0; i<size; i++) {
    printf("node: %u size: %u\n", i, adj_matrix[i].size);
    for (uint32_t j=0; j<adj_matrix[i].size; j++) {
      printf("%u ", adj_matrix[i].buff[j]);
    }
    printf("\n");
  }
}

uint8_t reallocate(int32_t** ptr, const uint64_t size, int32_t* vec_size) {
  *ptr = (int32_t*)realloc((void*)*ptr, size);
  if (!ptr) {
    return 1;
  }
  *vec_size = (int32_t)(size / sizeof(int32_t));
  return 0;
}

void adj_to_csr(Vector *adj_matrix, const int32_t size, int32_t *node_ptr,
                int32_t **node_neighbours, int32_t *node_neighbours_index,
                int32_t *total_neighbours) {
  for (int32_t i=0; i<size; i++) {
    const int32_t curr_node_len = adj_matrix[i].size;
    node_ptr[i] = *node_neighbours_index;
    if (*node_neighbours_index >= *total_neighbours) {
      uint8_t flag = reallocate(node_neighbours, sizeof(int32_t) * *node_neighbours_index * 2, total_neighbours);
      if (flag) {
        printf("failed to realloc - node_neighbours\n");
        exit(1);
      }
    }
    if (curr_node_len == 0) {
      (*node_neighbours)[*node_neighbours_index] = INT32_MAX;
      (*node_neighbours_index)++;
    } else {
      for (uint32_t j=0; j<curr_node_len; j++) {
        if (*node_neighbours_index >= *total_neighbours) {
          uint8_t flag = reallocate(node_neighbours, sizeof(int32_t) * *node_neighbours_index * 2, total_neighbours);
          if (flag) {
            printf("failed to realloc - node_neighbours\n");
            exit(1);
          }
        }
        (*node_neighbours)[*node_neighbours_index] = adj_matrix[i].buff[j];
        (*node_neighbours_index)++;
      }
    }
  }
}

void parse(int32_t* from, int32_t* to, const char* line) {
  char from_str[MAX_FILE_COLUMN_LEN] = {0};
  char to_str[MAX_FILE_COLUMN_LEN] = {0};
  int32_t from_str_index = 0;
  int32_t to_str_index = 0;
  const int32_t len = strlen(line);
  int32_t i = 0;
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

void print_result(int32_t* next_level_nodes, const int32_t num_next_level_nodes) {
  if (!next_level_nodes) {
    printf("null node\n");
  }
  for (int32_t i=0; i<num_next_level_nodes; i++) {
    printf("%u " ,next_level_nodes[i]);
  }
  printf("\n");
}

#if (USE_HOST == 1)
void host_queuing_kernel(int32_t *node_ptr, int32_t *node_neighbours,
                         int32_t *node_visited, int32_t *curr_level_nodes,
                         int32_t *next_level_nodes,
                         const int32_t num_curr_level_nodes,
                         const int32_t total_neighbours,
                         int32_t *num_next_level_nodes) {
  //iterate over neighbours
  for (int32_t i=0; i<num_curr_level_nodes; ++i) {
    int32_t node = curr_level_nodes[i];
#if (DEBUG_HOST_KERNEL == 1)
    printf("computing node: %u - neighbour index %u\n", node, node_ptr[node]);
#endif
    if (node_neighbours[node_ptr[node]] == INT32_MAX) {
      continue;;
    }
    for (int32_t j=node_ptr[node]; j<node_ptr[node+1]; ++j) {
      int32_t n = node_neighbours[j];
#if (DEBUG_HOST_KERNEL == 1)
      printf("j: %u - computing neighbour: %u\n", j, n);
#endif
      if (!node_visited[n]) {
        node_visited[n] = 1;
        next_level_nodes[*num_next_level_nodes] = n;
        (*num_next_level_nodes)++;
      }
    }
  }
#if (DEBUG == 1)
  // print_result(next_level_nodes, *num_next_level_nodes);
#endif
}
#else
__global__ void gpu_global_queuing_kernel(int32_t *node_ptr, int32_t *node_neighbours,
                         int32_t *node_visited, int32_t *curr_level_nodes,
                         int32_t *next_level_nodes,
                         const int32_t num_curr_level_nodes,
                         const int32_t total_neighbours,
                         int32_t *num_next_level_nodes) {
  int32_t node_idx = blockIdx.x * blockDim.x + threadIdx.x;
#if (DEBUG_KER == 1)
  printf("node idy %u node idx %u num_curr_level_nodes %u\n", node_idx, node_idxx, num_curr_level_nodes);
#endif
  if (node_idx >= num_curr_level_nodes) {
#if (DEBUG_KER == 1)
    printf("Skipping: %u\n", node_idx);
#endif
    return;
  }
#if (DEBUG_KER == 1)
  else {
    printf("Computing %u\n", node_idx);
  }
#endif
  //iterate over neighbours
  int32_t node = curr_level_nodes[node_idx];
  if (node_neighbours[node_ptr[node]] == INT32_MAX) {
    return;
  }
  for (int32_t j=node_ptr[node]; j<node_ptr[node+1]; ++j) {
    const int32_t n = node_neighbours[j];
    const int32_t visited = atomicCAS(&node_visited[n], 0, 1);
    if (visited == 0) {
      int32_t next_pos = atomicAdd(num_next_level_nodes, 1);
      next_level_nodes[next_pos] = n;
    }
  }
}

__global__ void gpu_block_queuing_kernel(int32_t *node_ptr, int32_t *node_neighbours,
                         int32_t *node_visited, int32_t *curr_level_nodes,
                         int32_t *next_level_nodes,
                         const int32_t num_curr_level_nodes,
                         const int32_t total_neighbours,
                         int32_t *num_next_level_nodes) {
  __shared__ int32_t shared_block_queue[DEVICE_SHARED_MEM_PER_BLOCK]; //max shared mem per multiprocessor / blocks per multiprocessor used (4KB)
  __shared__ int32_t shared_block_num_next_level_nodes;
  shared_block_num_next_level_nodes = 0;
  __syncthreads();

  int32_t node_idx = blockIdx.x * blockDim.x + threadIdx.x;
#if (DEBUG_KER == 1)
  printf("node idy %u node idx %u num_curr_level_nodes %u\n", node_idx, node_idxx, num_curr_level_nodes);
#endif
  if (node_idx >= num_curr_level_nodes) {
#if (DEBUG_KER == 1)
    printf("Skipping: %u\n", node_idx);
#endif
  } else {
    //iterate over neighbours
    int32_t node = curr_level_nodes[node_idx];
    if (node_neighbours[node_ptr[node]] < INT32_MAX) {
      for (int32_t j=node_ptr[node]; j<node_ptr[node+1]; ++j) {
        const int32_t n = node_neighbours[j];
#if (DEBUG_KER == 1)
        printf("computing node %u neighbours %u thread %u block %u\n", node, n, threadIdx.x, blockIdx.x * blockDim.x);
#endif
        const int32_t visited = atomicCAS(&node_visited[n], 0, 1);
        if (visited == 0) {
          int32_t next_pos_shared = atomicAdd(&shared_block_num_next_level_nodes, 1);
          if (next_pos_shared >= DEVICE_SHARED_MEM_PER_BLOCK) {
            printf("FATAL: next_pos_shared >= DEVICE_SHARED_MEM_PER_BLOCK\n");
            break;
          }
          shared_block_queue[next_pos_shared] = n;
        }
      }
    }
  }
  __syncthreads();
  if (shared_block_num_next_level_nodes < blockDim.x && threadIdx.x < shared_block_num_next_level_nodes) {
    int32_t next_pos = atomicAdd(num_next_level_nodes, 1);
    next_level_nodes[next_pos] = shared_block_queue[threadIdx.x];
  } else if (shared_block_num_next_level_nodes >= blockDim.x && threadIdx.x < blockDim.x) {
    const int32_t reminder = shared_block_num_next_level_nodes % blockDim.x;
    const int32_t write_per_thread = shared_block_num_next_level_nodes / blockDim.x;
    //printf("r %u w %u\n", reminder, write_per_thread);
    for (int32_t i=0; i<write_per_thread; i++) {
      int32_t next_pos = atomicAdd(num_next_level_nodes, 1);
      next_level_nodes[next_pos] = shared_block_queue[threadIdx.x * write_per_thread + i];
    }
    //threads lower than reminder writes one reminder each
    if (threadIdx.x < reminder) {
      int32_t next_pos = atomicAdd(num_next_level_nodes, 1);
      next_level_nodes[next_pos] = shared_block_queue[blockDim.x * write_per_thread + threadIdx.x];
    }
    /*
    if (threadIdx.x == blockDim.x-1) {
      for (int32_t i=0; i<reminder; i++) {
        int32_t next_pos = atomicAdd(num_next_level_nodes, 1);
        next_level_nodes[next_pos] = shared_block_queue[(threadIdx.x+1) * write_per_thread + i];
      }
    }
    */
  }
}
#endif

#if (USE_HOST == 1)
void launch_host_kernel(int32_t num_curr_level_nodes,
                        const int32_t total_neighbours, int32_t *node_ptr,
                        int32_t *node_neighbours, int32_t *node_visited,
                        int32_t *curr_level_nodes, int32_t *next_level_nodes,
                        FILE *out_next_level_nodes) {
  double total_elapsed_time = 0.0;
  int32_t level = 0;
  int8_t done = 0;
  int32_t* in_level;
  int32_t* out_level;
  int32_t num_out_level = num_curr_level_nodes;
  while (!done) {
    if (level == 0 || level%2 == 0) {
      in_level = curr_level_nodes;
      out_level = next_level_nodes;
    } else {
      in_level = next_level_nodes;
      out_level = curr_level_nodes;
    }
    num_curr_level_nodes = num_out_level;
    num_out_level = 0;
    clock_t start = clock();
    host_queuing_kernel(node_ptr, node_neighbours, node_visited, in_level,
                        out_level, num_curr_level_nodes, total_neighbours,
                        &num_out_level);
    clock_t end = clock();
    total_elapsed_time += (double)(end - start) * 1000.0 / CLOCKS_PER_SEC;
#if (DEBUG == 1)
    printf("next level nodes: %u:  ", num_out_level);
    print_result(out_level, num_out_level);
#else
    printf("next level nodes: %u\n", num_out_level);
#endif
#if (DEBUG >= 1 && PRODUCE_OUT_FILE == 1)
    if (num_out_level > 0) {
      for (int32_t i=0; i<num_out_level; i++) {
        fprintf(out_next_level_nodes, "%u\n", out_level[i]);
      }
      fprintf(out_next_level_nodes, "-------------\n");
    }
#endif
    if (num_out_level == 0) {
      done = 1;
    }
    level++;
  }
  printf("Time elapsed on host queueing: %f ms\n", total_elapsed_time);
}
#else
void launch_device_shared_queue_kernel(
    const int32_t threads_per_block, int32_t num_curr_level_nodes,
    const int32_t total_neighbours, int32_t *d_node_ptr, int32_t *d_node_neighbours,
    int32_t *d_node_visited, int32_t *d_curr_level_nodes, int32_t *d_next_level_nodes,
    FILE *out_next_level_nodes) {
  float total_global_queue_gpu_elapsed_time_ms = 0;
  int8_t done = 0;
  int32_t levels = 0;
  int32_t zero = 0;
  int32_t grid_size = 0;
  int32_t* in_level = 0;
  int32_t* out_level = 0;
  int32_t kernel_num_curr_level_nodes = 0;
  int32_t* num_out_level = 0;
  //initialise num_out_level
  cudaMallocManaged((void **) &num_out_level, sizeof(int32_t));
  cudaMemcpy(num_out_level, &num_curr_level_nodes, sizeof(int32_t), cudaMemcpyHostToDevice);
  while (!done) {
    cudaMemcpy(&kernel_num_curr_level_nodes, num_out_level, sizeof(int32_t), cudaMemcpyDeviceToHost);
    grid_size = (kernel_num_curr_level_nodes + threads_per_block - 1) / threads_per_block;
#if (DEBUG_KER_GRID == 1)
    printf("num grid %u with tpb %u tot_t %u\n", grid_size, threads_per_block, grid_size * threads_per_block);
#endif
    //grid_size = MAX_GRID_SIZE;
    //threads_per_block = (kernel_num_curr_level_nodes + grid_size - 1) / grid_size;
#if (DEBUG_KER_GRID == 1)
    //printf("optimised num grid %u with tpb %u tot_t %u\n", grid_size, threads_per_block, grid_size * threads_per_block);
#endif
    if (levels == 0 || levels%2 == 0) {
      in_level = d_curr_level_nodes;
      out_level = d_next_level_nodes;
    } else {
      in_level = d_next_level_nodes;
      out_level = d_curr_level_nodes;
    }
    cudaMemcpy(num_out_level, &zero, sizeof(int32_t), cudaMemcpyHostToDevice);
    float level_global_queue_gpu_elapsed_time_ms;
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start, 0);
    gpu_block_queuing_kernel<<<grid_size, threads_per_block>>>(
        d_node_ptr, d_node_neighbours, d_node_visited, in_level, out_level,
        kernel_num_curr_level_nodes, total_neighbours, num_out_level);
    cudaDeviceSynchronize();
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&level_global_queue_gpu_elapsed_time_ms, start, stop);
    total_global_queue_gpu_elapsed_time_ms += level_global_queue_gpu_elapsed_time_ms;

    //check stop criteria
    int32_t step_nodes;
    cudaMemcpy(&step_nodes, num_out_level, sizeof(int32_t), cudaMemcpyDeviceToHost);
    int32_t* step_nodes_buff = (int32_t*)malloc(sizeof(int32_t) * step_nodes);
    if (step_nodes == 0) {
      done = 1;
    }
#if (DEBUG < 1)
    printf("next level nodes: %u\n", step_nodes);
#endif

    //debug
#if (DEBUG == 1)
    if (!step_nodes_buff) {
      printf("malloc failed - step_nodes_buff - requested size: %u B\n", step_nodes * sizeof(uint32_t));
    } else if (step_nodes > 0) {
      cudaMemcpy(step_nodes_buff, out_level, sizeof(int32_t) * step_nodes, cudaMemcpyDeviceToHost);
      printf("next level nodes: %u:  ", step_nodes);
      print_result(step_nodes_buff, step_nodes);
    }
#endif
#if (DEBUG >= 1 && PRODUCE_OUT_FILE == 1)
    //save output
    if (step_nodes > 0) {
      for (int32_t i=0; i<step_nodes; i++) {
        fprintf(out_next_level_nodes, "%u\n", step_nodes_buff[i]);
      }
      fprintf(out_next_level_nodes, "-------------\n");
    }
#endif
    levels++;
  }
  printf("Time elapsed on block GPU queueing: %f ms\n",
         total_global_queue_gpu_elapsed_time_ms);
  cudaFree(num_out_level);
}

void launch_device_global_queue_kernel(
    const int32_t threads_per_block, int32_t num_curr_level_nodes,
    const int32_t total_neighbours, int32_t *d_node_ptr, int32_t *d_node_neighbours,
    int32_t *d_node_visited, int32_t *d_curr_level_nodes, int32_t *d_next_level_nodes,
    FILE *out_next_level_nodes) {
  float total_global_queue_gpu_elapsed_time_ms = 0;
  int8_t done = 0;
  int32_t levels = 0;
  int32_t zero = 0;
  int32_t grid_size = 0;
  int32_t* in_level = 0;
  int32_t* out_level = 0;
  int32_t kernel_num_curr_level_nodes = 0;
  int32_t* num_out_level = 0;
  //initialise num_out_level
  cudaMallocManaged((void **) &num_out_level, sizeof(int32_t));
  cudaMemcpy(num_out_level, &num_curr_level_nodes, sizeof(int32_t), cudaMemcpyHostToDevice);
  while (!done) {
    cudaMemcpy(&kernel_num_curr_level_nodes, num_out_level, sizeof(int32_t), cudaMemcpyDeviceToHost);
    grid_size = (kernel_num_curr_level_nodes + threads_per_block - 1) / threads_per_block;
#if (DEBUG_KER_GRID == 1)
    printf("non optimised num grid %u with tpb %u tot_t %u\n", grid_size, threads_per_block, grid_size * threads_per_block);
#endif
    //grid_size = MAX_GRID_SIZE;
    //threads_per_block = (kernel_num_curr_level_nodes + grid_size - 1) / grid_size;
#if (DEBUG_KER_GRID == 1)
    //printf("optimised num grid %u with tpb %u tot_t %u\n", grid_size, threads_per_block, grid_size * threads_per_block);
#endif
    if (levels == 0 || levels%2 == 0) {
      in_level = d_curr_level_nodes;
      out_level = d_next_level_nodes;
    } else {
      in_level = d_next_level_nodes;
      out_level = d_curr_level_nodes;
    }
    cudaMemcpy(num_out_level, &zero, sizeof(int32_t), cudaMemcpyHostToDevice);
    float level_global_queue_gpu_elapsed_time_ms;
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start, 0);
    gpu_global_queuing_kernel<<<grid_size, threads_per_block>>>(
        d_node_ptr, d_node_neighbours, d_node_visited, in_level, out_level,
        kernel_num_curr_level_nodes, total_neighbours, num_out_level);
    cudaDeviceSynchronize();
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&level_global_queue_gpu_elapsed_time_ms, start, stop);
    total_global_queue_gpu_elapsed_time_ms += level_global_queue_gpu_elapsed_time_ms;

    //check stop criteria
    int32_t step_nodes;
    cudaMemcpy(&step_nodes, num_out_level, sizeof(int32_t), cudaMemcpyDeviceToHost);
    int32_t* step_nodes_buff = (int32_t*)malloc(sizeof(int32_t) * step_nodes);
    if (step_nodes == 0) {
      done = 1;
    }
#if (DEBUG < 1)
    printf("next level nodes: %u\n", step_nodes);
#endif

    //debug
#if (DEBUG == 1)
    if (!step_nodes_buff) {
      printf("malloc failed - step_nodes_buff - requested size: %u B\n", step_nodes * sizeof(uint32_t));
    } else if (step_nodes > 0) {
      cudaMemcpy(step_nodes_buff, out_level, sizeof(int32_t) * step_nodes, cudaMemcpyDeviceToHost);
      printf("next level nodes: %u:  ", step_nodes);
      print_result(step_nodes_buff, step_nodes);
    }
#endif
#if (DEBUG >= 1 && PRODUCE_OUT_FILE == 1)
    //save output
    if (step_nodes > 0) {
      for (int32_t i=0; i<step_nodes; i++) {
        fprintf(out_next_level_nodes, "%u\n", step_nodes_buff[i]);
      }
      fprintf(out_next_level_nodes, "-------------\n");
    }
#endif
    levels++;
  }
  printf("Time elapsed on global GPU queueing: %f ms\n",
         total_global_queue_gpu_elapsed_time_ms);
  cudaFree(num_out_level);
}
#endif

int prepare_and_spawn(const char* input_file, const char* next_level_out_file, const char* visited_out_file) {
#if (USE_HOST == 0)
  // retrieve some info abfile_out_next_level_node the CUDA device
  int32_t num_devices;
  cudaGetDeviceCount(&num_devices);
  cudaDeviceProp main_device_prop;
  for (int i = 0; i < num_devices; i++) {
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, i);
    memcpy(&main_device_prop, &prop, sizeof(cudaDeviceProp));
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
  //device params
#if (USE_HOST == 0)
  int32_t* d_curr_level_nodes = 0;
  int32_t* d_next_level_nodes = 0;
  int32_t* d_node_neighbours = 0;
  int32_t* d_node_ptr = 0;
  int32_t* d_node_visited = 0;
  int32_t* num_next_level_nodes = 0;
  cudaMallocManaged((void **) &num_next_level_nodes, sizeof(int32_t));
#endif

  //host params
  FILE *file;
  FILE *file_out_next_level_node;
  FILE *file_out_visited_node;
  Vector* adj_matrix = 0;
  int32_t line_counter = 0;
  int32_t max_node_val = 0;
  int32_t node_neighbours_index = 0;
  int32_t num_curr_level_nodes = 0;
  int32_t num_nodes = 0;
  int32_t total_neighbours = 0;
  int32_t* curr_level_nodes = 0;
  int32_t* next_level_nodes = 0;
  int32_t* node_neighbours = 0;
  int32_t* node_ptr = 0;
  int32_t* node_visited = 0;
  file = fopen(input_file, "r");
  if (file == NULL) {
    printf("Error opening the file.\n");
    return 1;
  }
  file_out_next_level_node = fopen(next_level_out_file, "w");
  if (file_out_next_level_node == NULL) {
    printf("Error opening the file_out_next_level_node file.\n");
    return 1;
  }
  file_out_visited_node = fopen(visited_out_file, "w");
  if (file_out_visited_node == NULL) {
    printf("Error opening the file_out_next_level_node file.\n");
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
  //heuristic initial allocation
  total_neighbours += 2 * num_nodes;
  printf("total nodes: %d\n", REAL_NODES_NUM(num_nodes));

  //allocate buffers
  node_ptr = (int32_t*)malloc(sizeof(int32_t) * num_nodes);
  if (!node_ptr) {
    printf("failed malloc - node_ptr\n");
    return 1;
  }
  for (int32_t i=0; i<num_nodes; i++) {
    node_ptr[i] = INT32_MAX;
  }
#if (USE_HOST == 0)
  cudaMallocManaged((void **) &d_node_ptr, sizeof(int32_t) * num_nodes);
  if (!d_node_ptr) {
    printf("failed malloc - d_node_ptr\n");
  }
#endif
  curr_level_nodes = (int32_t*)malloc(sizeof(int32_t) * num_nodes);
#if (USE_HOST == 0)
  cudaMallocManaged((void **) &d_curr_level_nodes, sizeof(int32_t) * num_nodes);
  if (!d_curr_level_nodes) {
    printf("failed malloc - d_curr_level_nodes\n");
  }
#endif
  if (!curr_level_nodes) {
    printf("failed malloc - curr_level_nodes\n");
  }
  node_neighbours = (int32_t*)malloc(sizeof(int32_t) * total_neighbours);
  //cuda allocation happens later
  if (!node_neighbours) {
    printf("failed malloc - node_neighbours\n");
  }
  next_level_nodes = (int32_t*)malloc(sizeof(int32_t) * total_neighbours);
#if (USE_HOST == 0)
  cudaMallocManaged((void **) &d_next_level_nodes, sizeof(int32_t) * total_neighbours);
  if (!d_next_level_nodes) {
    printf("failed malloc - d_next_level_nodes\n");
  }
#endif
  if (!next_level_nodes) {
    printf("failed malloc - next_level_nodes\n");
  }

  file = fopen(input_file, "r");
  line_counter = 0;
  if (file == NULL) {
    printf("Can not open the file");
    return 1;
  }

  adj_matrix = (Vector*)malloc(sizeof(Vector) * num_nodes);
  if (!adj_matrix) {
    printf("malloc failed - adj_matrix\n");
    return 1;
  }
  for (int32_t i=0; i<num_nodes; i++) {
    adj_matrix[i].buff = (int32_t*)malloc(sizeof(int32_t) * ADJ_MATRIX_ROW_SIZE);
    if (!adj_matrix[i].buff) {
      printf("malloc failed - adj_matrix[%u]\n", i);
      return 1;
    }
    adj_matrix[i].size = 0;
  }
  while (fgets(line, sizeof(line), file) != NULL) {
    if (!line_counter) {
      (line_counter)++;
      continue;
    }
    int32_t from;
    int32_t to;
    parse(&from, &to, line);
    adj_matrix[from].buff[adj_matrix[from].size++] = to;
    adj_matrix[to].buff[adj_matrix[to].size++] = from;
  }
  fclose(file);

  //print adj matrix
#if (DEBUG > 2)
  print_adj_matrix(adj_matrix, num_nodes);
#endif

  //move
  adj_to_csr(adj_matrix, num_nodes, node_ptr, &node_neighbours, &node_neighbours_index, &total_neighbours);
#if (USE_HOST == 0)
  cudaMallocManaged((void **) &d_node_neighbours, sizeof(int32_t) * total_neighbours);
  if (!d_node_neighbours) {
    printf("failed malloc - node_neighbours\n");
  }
#endif
  total_neighbours = node_neighbours_index;

  max_node_val = num_nodes;
  node_visited = (int32_t*)malloc(sizeof(int32_t) * max_node_val);
  if (!node_visited) {
    printf("failed malloc - node_visited)\n");
  }
  memset(node_visited, 0, sizeof(int32_t) * max_node_val);
#if (USE_HOST == 0)
  cudaMallocManaged((void **) &d_node_visited, sizeof(int32_t) * max_node_val);
  if (!d_node_visited) {
    printf("failed malloc - d_node_visited)\n");
  }
#endif

#if (DEBUG > 1)
  printf("node ptr:\n");
  for (int32_t i=0; i<num_nodes; i++) {
    printf("%u ", (node_ptr)[i]);
  }
  printf("\n");
  printf("node neighbours:\n");
  for (int32_t i=0; i<total_neighbours; i++) {
    printf("%u ", (node_neighbours)[i]);
  }
  printf("\n");
#endif

#if (USE_HOST == 0 && USE_PREFIX_SUM == 1)
#error "Prefix sum not supported"
  //yet not supported
  //compute prefix sum
  int32_t* prefix_sum = 0;
  cudaMallocManaged((void **) &prefix_sum, sizeof(int32_t) * num_nodes);
  int32_t count = 0;
  for (int32_t i=0; i<num_nodes-1; i++) {
    prefix_sum[i] = count;
    count += node_ptr[i+1] - node_ptr[i];
  }
  prefix_sum[num_nodes-1] = count;
  printf("prefix sum\n");
  for (int32_t i=0; i<num_nodes; i++) {
    printf("%u ", prefix_sum[i]);
  }
  printf("\n");
  //assign next level nodes count
  *num_next_level_nodes = count;
#endif

#if (USE_HOST == 1)
  //populate starting curr_level_nodes
  curr_level_nodes[0] = STARTING_NODE;
  num_curr_level_nodes = STARTING_LEVEL_NUM;
  node_visited[0] = 1;
  printf("\nlaunching host kernel with initial level nodes size: %u\n", num_curr_level_nodes);
  launch_host_kernel(num_curr_level_nodes, total_neighbours, node_ptr,
                     node_neighbours, node_visited, curr_level_nodes,
                     next_level_nodes, file_out_next_level_node);
#else
  //populate starting curr_level_nodes
  curr_level_nodes[0] = STARTING_NODE;
  num_curr_level_nodes = STARTING_LEVEL_NUM;
  node_visited[0] = 1;
  cudaMemcpy(d_node_visited, node_visited, sizeof(int32_t), cudaMemcpyHostToDevice); //copy first 4 bytes
  printf("\nlaunching device kernel (global queue) with initial level nodes size: %u\n", num_curr_level_nodes);
  //copy resources
  cudaMemcpy(d_node_neighbours, node_neighbours, total_neighbours * sizeof(int32_t), cudaMemcpyHostToDevice);
  cudaMemcpy(d_node_ptr, node_ptr, num_nodes * sizeof(int32_t), cudaMemcpyHostToDevice);
  cudaMemcpy(d_curr_level_nodes, curr_level_nodes, num_nodes * sizeof(int32_t), cudaMemcpyHostToDevice);
  cudaMemcpy(d_node_visited, node_visited, max_node_val * sizeof(int32_t), cudaMemcpyHostToDevice);

  const int32_t threads_per_block = main_device_prop.maxThreadsPerBlock/main_device_prop.maxBlocksPerMultiProcessor;

  launch_device_global_queue_kernel(threads_per_block, num_curr_level_nodes, total_neighbours, d_node_ptr,
                     d_node_neighbours, d_node_visited, d_curr_level_nodes,
                     d_next_level_nodes, file_out_next_level_node);

  //copy resources
  cudaMemcpy(d_node_neighbours, node_neighbours, total_neighbours * sizeof(int32_t), cudaMemcpyHostToDevice);
  cudaMemcpy(d_node_ptr, node_ptr, num_nodes * sizeof(int32_t), cudaMemcpyHostToDevice);
  cudaMemcpy(d_curr_level_nodes, curr_level_nodes, num_nodes * sizeof(int32_t), cudaMemcpyHostToDevice);
  memset(node_visited, 0, sizeof(int32_t) * max_node_val);
  cudaMemcpy(d_node_visited, node_visited, max_node_val * sizeof(int32_t), cudaMemcpyHostToDevice);
  //populate starting curr_level_nodes
  curr_level_nodes[0] = STARTING_NODE;
  num_curr_level_nodes = STARTING_LEVEL_NUM;
  node_visited[0] = 1;
  cudaMemcpy(d_node_visited, node_visited, sizeof(int32_t), cudaMemcpyHostToDevice); //copy first 4 bytes
  printf("\nlaunching device kernel (shared queue) with initial level nodes size: %u\n", num_curr_level_nodes);

  launch_device_shared_queue_kernel(threads_per_block, num_curr_level_nodes, total_neighbours, d_node_ptr,
                     d_node_neighbours, d_node_visited, d_curr_level_nodes,
                     d_next_level_nodes, file_out_next_level_node);
#endif

#if (USE_HOST == 0)
#if (DEBUG >= 1 && PRODUCE_OUT_FILE == 1)
  cudaMemcpy(node_visited, d_node_visited, max_node_val * sizeof(int32_t), cudaMemcpyDeviceToHost);
  for (int32_t i=0; i<max_node_val; i++) {
    fprintf(file_out_visited_node, "%u\n", node_visited[i]);
  }
#endif
#else
#if (DEBUG >= 1 && PRODUCE_OUT_FILE == 1)
  for (int32_t i=0; i<max_node_val; i++) {
    fprintf(file_out_visited_node, "%u\n", node_visited[i]);
  }
#endif
#endif
  fclose(file_out_next_level_node);
  fclose(file_out_visited_node);

  free(node_ptr);
  free(node_neighbours);
  free(curr_level_nodes);
  free(next_level_nodes);
  free(node_visited);
  for (int32_t i=0; i<num_nodes; i++) {
    if (adj_matrix[i].buff) {
      free(adj_matrix[i].buff);
    }
  }
  free(adj_matrix);
#if (USE_HOST == 0)
  cudaFree(d_node_ptr);
  cudaFree(d_node_neighbours);
  cudaFree(d_curr_level_nodes);
  cudaFree(d_next_level_nodes);
  cudaFree(d_node_visited);
  cudaFree(num_next_level_nodes);
#endif

  return 0;
}

int main(int argc, char* argv[]) {
  int in_count;
#if (USE_HOST == 1)
  char** in_values;
  in_count = argc;
  in_values = (char**)malloc(sizeof(char*) * in_count);
  if (!in_values) {
    printf("malloc failed - in_values\n");
    return 1;
  }
  for (int32_t i=0; i<in_count; i++) {
    in_values[i] = (char*)malloc(sizeof(char) * (strlen(argv[i]) + 1));
    if (!in_values[i]) {
      printf("malloc failed - in_values[%u]\n", i);
      return 1;
    }
    memcpy(in_values[i], argv[i], strlen(argv[i]) + 1);
  }
  if (argc == 0) {
    printf("no input file specified\n");
    return 0;
  }
#else
  in_count = 2; //to complete (set the value to input file num + 1)
  char in_values[][100] =  {
    "standard6.txt", "roadNet-CA.txt"
  }; 
  if (in_count == 0) {
    printf("no input file specified. please set input files at line %d\n", __LINE__);
    return 0;
  }
#endif
#if (USE_HOST == 1)
  for (int32_t i=1; i<in_count; i++) {
#else
  for (int32_t i=0; i<in_count; i++) {
#endif
    printf("computing: %s\n", in_values[i]);
    const size_t file_name_len = strlen(in_values[i]);
    char* level_out_file_name = (char*)malloc((file_name_len + 15) * sizeof(char));
    char* visited_out_file_name = (char*)malloc((file_name_len + 15) * sizeof(char));
    if (!level_out_file_name) {
      printf("malloc failed - level_out_file_name\n");
      return 1;
    }
    if (!visited_out_file_name) {
      printf("malloc failed - visited_out_file_name\n");
      return 1;
    }
    memcpy(level_out_file_name, in_values[i], sizeof(char) * (file_name_len-4));
    memcpy(visited_out_file_name, in_values[i], sizeof(char) * (file_name_len-4));
    strcat(level_out_file_name, "_next_level_out.txt");
    strcat(visited_out_file_name, "_visited_out.txt");
    if(prepare_and_spawn(in_values[i], level_out_file_name, visited_out_file_name)) {
      printf("kernel failed for %s\n", in_values[i]);
    } else {
      printf("kernel ended for %s\n\n", in_values[i]);
    }
    free(level_out_file_name);
    free(visited_out_file_name);
  }
#if (USE_HOST == 1)
  for (int32_t i=0; i<in_count; i++) {
    free(in_values[i]);
  }
  free(in_values);
#endif
  return 0;
}
