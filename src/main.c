#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define MAX_FILE_COLUMN_LEN 100
#define SPACE ' '

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

uint8_t load_and_allocate(const char* file_name, uint32_t* line_counter, uint32_t *num_nodes, uint32_t* total_neighbours, uint32_t* node_ptr, uint32_t* node_neighbours) {
  FILE *file;
  uint32_t node_ptr_index = 0;
  uint32_t node_neighbours_index = 0;
  file = fopen(file_name, "r");
  if (file == NULL) {
    printf("Error opening the file.\n");
    return 1;
  }
  char line[MAX_FILE_COLUMN_LEN] = {0};
  while (fgets(line, sizeof(line), file) != NULL) {
    if (!*line_counter) {
      (*line_counter)++;
      *num_nodes = atoi(line);
    }
    //counting the first line, yes one too much but then compensated with the virtual node added
    (*total_neighbours)++;
  }
  fclose(file);
  //add virtual node to avoid index overflow
  (*num_nodes)++;
  printf("total nodes: %d - total neighbours: %d\n", *num_nodes, *total_neighbours);
  node_ptr = (uint32_t*)malloc(sizeof(uint32_t) * *num_nodes);
  node_neighbours = (uint32_t*)malloc(sizeof(uint32_t) * *total_neighbours);


  file = fopen(file_name, "r");
  *line_counter = 0;
  if (file == NULL) {
    printf("Error opening the file.\n");
    return 1;
  }
  while (fgets(line, sizeof(line), file) != NULL) {
    if (!*line_counter) {
      (*line_counter)++;
      continue;
    }
    uint32_t from;
    uint32_t to;
    parse(&from, &to, line);
    // printf("from %d to %d\n", from, to);
    if (!node_ptr_index || node_ptr[node_ptr_index-1] != from) {
      node_ptr[node_ptr_index++] = from;
    }
    node_neighbours[node_neighbours_index++] = to;
  }
  fclose(file);

  printf("node ptr:\n");
  for (uint32_t i=0; i<node_ptr_index; i++) {
    printf("%u ", node_ptr[i]);
  }
  printf("\n");
  printf("node neighbours:\n");
  for (uint32_t i=0; i<node_neighbours_index; i++) {
    printf("%u ", node_neighbours[i]);
  }
  printf("\n");

  return 0;
}

int main(int argc, char* argv[]) {
  uint32_t line_counter;
  uint32_t num_nodes = 0;
  uint32_t total_neighbours = 0;
  uint32_t* node_ptr = 0;
  uint32_t* node_neighbours = 0;

  if (!load_and_allocate("../tests/1.txt", &line_counter, &num_nodes, &total_neighbours, node_ptr, node_neighbours)) {
    return 1;
  }

  free(node_ptr);
  free(node_neighbours);

  return 0;
}
